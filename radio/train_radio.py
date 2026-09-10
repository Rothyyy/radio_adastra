import contextlib
import datetime
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from radio.logger import TrainingLogger
from radio.model_teacher import Teachers
from radio.train_utils import plot_training_loss, setup_scheduler, update_best_metric

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Distillation loss  (RADIOv2.5)
# ---------------------------------------------------------------------------
#
#   summary / CLS token : cosine distance  1 - cos(student, teacher)
#   spatial / patch      : MSE(student, PHI-S(teacher))
#
# PHI-S standardizes each teacher's spatial features to ~unit per-channel
# variance via an orthogonal rotation + a scalar (fit once by
# calibrate_teachers.py over the frozen teachers). Because it's a rotation + a
# scalar, MSE in PHI-S space is proportional to MSE in raw feature space, just
# balanced across channels and teachers - so teachers can be equally weighted.

TEACHER_NAMES = ("seg", "curia", "vlm")
TEACHER_DIMS = {"seg": 768, "curia": 768, "vlm": 1152}


class RadioLoss(nn.Module):
    """RADIOv2.5-style multi-teacher distillation loss.

    ``phis_path`` is the ``teacher_phis.pt`` produced by ``calibrate_teachers.py``
    (mean / rotation / scale per teacher). Without it the spatial term falls back
    to a plain MSE on raw features (unbalanced - only for quick evals).

    ``last_raw`` / ``last_components`` expose the latest per-teacher values for
    logging. Transform buffers ride along in checkpoints.
    """

    def __init__(self, phis_path: str = None, summary_weight: float = 1.0,
                 spatial_weight: float = 1.0):
        super().__init__()
        self.summary_weight = summary_weight
        self.spatial_weight = spatial_weight
        self.has_phis = bool(phis_path) and os.path.isfile(phis_path)

        # bump when the loss formula changes; a resume with a different version
        # re-inits the criterion state instead of loading a stale one
        self.register_buffer("_loss_ver", torch.tensor(4, dtype=torch.long))
        self.register_buffer("_steps", torch.tensor(0, dtype=torch.long))

        st = torch.load(phis_path, map_location="cpu") if self.has_phis else {}
        for name, d in TEACHER_DIMS.items():
            if name in st:
                self.register_buffer(f"mean_{name}", st[name]["mean"].float())
                self.register_buffer(f"rot_{name}", st[name]["rot"].float())
                self.register_buffer(f"scale_{name}", st[name]["scale"].float().reshape(()))
            else:
                self.register_buffer(f"mean_{name}", torch.zeros(d))
                self.register_buffer(f"rot_{name}", torch.eye(d))
                self.register_buffer(f"scale_{name}", torch.tensor(1.0))
        if not self.has_phis:
            print("[RadioLoss] no PHI-S file - spatial term is unbalanced raw MSE", flush=True)

        self.last_raw, self.last_components = {}, {}

    def _phis(self, name, feat):                       # feat: [..., D] -> [..., D]
        m = getattr(self, f"mean_{name}")
        r = getattr(self, f"rot_{name}")
        s = getattr(self, f"scale_{name}")
        return torch.matmul(feat - m, r.t()) / s.clamp_min(1e-6)

    def _teacher_loss(self, name, s_cls, t_cls, s_patch, t_patch):
        parts = {}
        loss = s_patch.new_zeros(())

        if s_cls is not None and t_cls is not None:
            cos = 1.0 - F.cosine_similarity(s_cls.float(), t_cls.float(), dim=-1).mean()
            parts["summary_cos"] = float(cos.detach())
            loss = loss + self.summary_weight * cos

        t_std = self._phis(name, t_patch.detach().float())
        mse = F.mse_loss(s_patch.float(), t_std)
        parts["spatial_mse"] = float(mse.detach())
        loss = loss + self.spatial_weight * mse
        return loss, parts

    def radio_loss(self, tokens_student, tokens_teachers):
        """tokens_* = [cls_seg, patch_seg, cls_curia, patch_curia, cls_vlm, patch_vlm]."""
        total = tokens_student[1].new_zeros(())
        n_active = 0
        self.last_raw, self.last_components = {}, {}

        for i, name in enumerate(TEACHER_NAMES):
            s_cls, s_patch = tokens_student[2 * i], tokens_student[2 * i + 1]
            t_cls, t_patch = tokens_teachers[2 * i], tokens_teachers[2 * i + 1]
            if t_patch is None:            # teacher disabled
                continue
            n_active += 1

            raw, parts = self._teacher_loss(name, s_cls, t_cls, s_patch, t_patch)
            self.last_raw[name] = float(raw.detach())
            self.last_components.update({f"{name}_{k}": v for k, v in parts.items()})
            total = total + raw

        if self.training:
            self._steps += 1
        return total / max(n_active, 1)

    def forward(self, tokens_student, tokens_teachers):
        return self.radio_loss(tokens_student, tokens_teachers)


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------
def dataloader_loop_radio(
    model,
    teachers,
    dataloader,
    criterion,
    epoch,
    optimizer=None,
    scheduler=None,
    accum_steps: int = 1,
    grad_clip: float = 1.0,
    log_every: int = 50,
    device=device,
):
    is_train = optimizer is not None
    criterion.train(is_train)
    loss_sum, n = 0.0, 0
    raw_sum = {name: 0.0 for name in TEACHER_NAMES}
    rank = int(os.environ.get("RANK", 0))

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    n_batches = len(dataloader)
    epoch_start = step_start = time.perf_counter()
    with torch.set_grad_enabled(is_train):
        for step, x in enumerate(dataloader):
            x = x.to(device, dtype=torch.float32, non_blocking=True)

            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
                tokens_student = model(x)
                with torch.no_grad():
                    tokens_teachers = teachers(x)
                loss = criterion(tokens_student, tokens_teachers)

            if is_train:
                # boundary = last micro-step of an accumulation window, or of the epoch
                at_boundary = (step + 1) % accum_steps == 0 or (step + 1) == n_batches
                sync_ctx = (model.no_sync() if not at_boundary and hasattr(model, "no_sync")
                            else contextlib.nullcontext())
                with sync_ctx:
                    (loss / accum_steps).backward()
                if at_boundary:
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            bs = x.size(0)
            loss_sum += loss.detach().float().item() * bs
            for name, v in criterion.last_raw.items():
                raw_sum[name] += v * bs
            n += bs

            if rank == 0 and log_every and step % log_every == 0 and step > 0:
                now = time.perf_counter()
                dt = (now - step_start) / log_every
                # ETA from the average step time over the whole epoch so far
                avg = (now - epoch_start) / (step + 1)
                eta_s = avg * (n_batches - step - 1)
                eta = datetime.timedelta(seconds=round(eta_s))
                finish = (datetime.datetime.now()
                          + datetime.timedelta(seconds=eta_s)).strftime("%H:%M")
                if torch.cuda.is_available():
                    mem = (f" peak_alloc={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
                           f" peak_resv={torch.cuda.max_memory_reserved() / 1e9:.1f}GB")
                else:
                    mem = ""
                raw_str = " ".join(f"{k}={v / max(n, 1):.3f}" for k, v in raw_sum.items())
                print(f"epoch {epoch} step {step}/{n_batches}: {dt:.3f}s/step "
                      f"({1 / dt:.2f} it/s) loss={loss_sum / max(n, 1):.4f} [{raw_str}] "
                      f"eta={eta} (~{finish}){mem}", flush=True)
                step_start = time.perf_counter()

    mean = loss_sum / max(n, 1)
    raw_mean = {name: raw_sum[name] / max(n, 1) for name in TEACHER_NAMES}
    return mean, raw_mean


def train_radio_amp(
    model,
    train_dataloader,
    valid_dataloader,
    num_epoch: int = 100,
    lr: float = 1e-4,
    w_decay: float = 1e-3,
    accum_steps: int = 1,
    grad_clip: float = 1.0,
    head_lr_mult: float = 1.0,
    find_unused_parameters: bool = False,
    loss_fn: nn.Module = None,
    optimizer: str = "AdamW",
    model_save_path: str = "save_model/model_checkpoint",
    plot_save_path: str = "plot_train/training_loss.pdf",
    train_logger: TrainingLogger = None,
    resume: str = None,
    device=device,
    rank: int = 0,
    is_distributed: bool = False,
):
    device = torch.device(device)
    model = model.to(device)
    teachers = Teachers().to(device).eval()
    criterion = (loss_fn if loss_fn is not None else RadioLoss()).to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index],
            find_unused_parameters=find_unused_parameters)
    raw_model = model.module if is_distributed else model

    # two param groups: the backbone at `lr`, the fresh projection heads at
    # `lr * head_lr_mult` (heads start from scratch even when the backbone is a
    # pretrained foundation model).
    backbone_params, head_params = [], []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else head_params).append(p)
    groups = [
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ]
    if optimizer == "AdamW":
        opt = optim.AdamW(groups, lr=lr, weight_decay=w_decay)
    else:
        opt = optim.SGD(groups, lr=lr, weight_decay=w_decay, momentum=0.9, nesterov=True)
    scheduler = setup_scheduler(opt, num_epoch)

    start_epoch = 1
    best_valid_loss = float("inf")
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        csd = ckpt.get("criterion_state_dict")
        if csd is not None:
            same_loss_ver = "_loss_ver" in csd and torch.equal(
                csd["_loss_ver"].cpu(), criterion._loss_ver.cpu())
            if same_loss_ver:
                criterion.load_state_dict(csd, strict=False)  # PHI-S buffers come from file
            elif rank == 0:
                print("criterion loss version changed - using fresh loss state", flush=True)
        start_epoch = ckpt["epoch"] + 1
        best_valid_loss = ckpt.get("best_valid_loss", best_valid_loss)
        if rank == 0:
            print(f"Resumed from {resume} at epoch {start_epoch}", flush=True)

    if rank == 0:
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    train_loss_epoch, valid_loss_epoch = [], []

    iterator = range(start_epoch, num_epoch + 1)
    if rank == 0:
        iterator = tqdm(iterator, desc="Training", file=sys.stdout)
    for epoch in iterator:
        model.train()
        train_mean, train_raw = dataloader_loop_radio(
            model, teachers, train_dataloader, criterion, epoch,
            optimizer=opt, scheduler=scheduler, accum_steps=accum_steps,
            grad_clip=grad_clip, device=device,
        )
        train_loss_epoch.append(train_mean)

        model.eval()
        valid_mean, valid_raw = dataloader_loop_radio(
            model, teachers, valid_dataloader, criterion, epoch, optimizer=None,
            device=device,
        )
        valid_loss_epoch.append(valid_mean)

        scheduler.step()
        new_best, best_valid_loss = update_best_metric(valid_mean, best_valid_loss, mode="min")

        if rank != 0:
            continue

        metric_dict = {
            "Epoch": epoch, "Train_Loss": round(train_mean, 4),
            "Valid_Loss": round(valid_mean, 4),
            "LR": f"{opt.param_groups[0]['lr']:.3e}",
            "Head_LR": f"{opt.param_groups[1]['lr']:.3e}",
            **{f"tr_{k}": round(v, 4) for k, v in train_raw.items()},
            **{f"va_{k}": round(v, 4) for k, v in valid_raw.items()},
        }
        if train_logger is not None:
            train_logger.log(metric_dict)
        iterator.set_postfix({k: metric_dict[k] for k in ("Epoch", "Train_Loss", "Valid_Loss")})

        ckpt = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": {  # static PHI-S buffers come from the file, skip them
                k: v for k, v in criterion.state_dict().items()
                if k in ("_loss_ver", "_steps")},
            "best_valid_loss": best_valid_loss,
        }
        torch.save(ckpt, model_save_path + "_last.pth")
        if new_best:
            torch.save(ckpt, model_save_path + "_best_valid.pth")
            print(f"Saved best valid model at epoch {epoch} (valid_loss={valid_mean:.4f})", flush=True)
        plot_training_loss(train_loss_epoch, valid_loss_epoch, plot_save_path)

    if rank == 0:
        print(f"End of training. Best valid loss = {best_valid_loss:.4f}", flush=True)
        if train_logger is not None:
            train_logger.close()
