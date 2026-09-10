"""RADIO pretraining entry point.

Distils MedSAM + Curia + MedSigLIP into a single CT student backbone.
No classification here — this is representation pretraining; a downstream head
is trained separately later. 

Example:
    python -m radio.train_radio_script --num_exp 1 --backbone curia \
        --lr 1e-4 --w_decay 1e-3 --batch_size 2 --num_slices 10 \
        --accum_steps 2 --num_epoch 80 --save_path train_2d_models
"""

import argparse
import os
from datetime import timedelta

import pandas as pd
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler

from radio.dataset_radio import DatasetRadio2D, make_collate_radio2D
from radio.logger import TrainingLogger
from radio.model_radio import Radio
from radio.train_radio import RadioLoss, train_radio_amp

SAVE_ROOT = os.environ.get("RADIO_SAVE_ROOT", "/lus/work/CT3/cad17796/rsochet/save_model")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_exp", type=int, default=0, help="Experiment id for the save folder")
    p.add_argument("--csv", type=str, default="dataset_radio.csv")
    p.add_argument("--save_path", type=str, default="train_2d_models")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Fixed output dir (name under RADIO_SAVE_ROOT, or absolute path). "
                        "Auto-resumes from <run_dir>/model_last.pth; writes COMPLETED when "
                        "all epochs are done. Use this for resubmission chains.")
    p.add_argument("--backbone", type=str, default="curia",
                   help="scratch | curia | vit_b16 | vit_s16 | timm:<name>")
    p.add_argument("--freeze_blocks", type=int, default=0,
                   help="Freeze the first N transformer blocks of the student")
    p.add_argument("--batch_size", type=int, default=2, help="Volumes per step")
    p.add_argument("--num_slices", type=int, default=10, help="Axial slices sampled per volume")
    p.add_argument("--pre_cleaned", action="store_true",
                   help="CSV is from preprocess_dataset.py: skip the per-step air-slice filter")
    p.add_argument("--phis_path", type=str, default="models/teacher_phis.pt",
                   help="PHI-S transform from calibrate_teachers.py (spatial loss standardization)")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--num_epoch", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4, help="Backbone learning rate")
    p.add_argument("--head_lr_mult", type=float, default=1.0,
                   help="Projection-head LR = lr * this (heads start from scratch)")
    p.add_argument("--w_decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--ddp_find_unused", action="store_true",
                   help="DDP find_unused_parameters (safety net; small per-step overhead)")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main():
    args = get_args()

    # ---- distributed setup (torchrun sets these env vars; absent => single process) ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        # generous timeout: with whole-volume reads off Lustre a straggler rank
        # can be minutes behind; the default 600 s aborts the whole job.
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if distributed else 0

    if rank == 0:
        print(f"RADIO pretraining | backbone={args.backbone} | world_size={world_size} | "
              f"teachers=[MedSAM, Curia, MedSigLIP] | lr={args.lr} wd={args.w_decay} "
              f"epochs={args.num_epoch} bs={args.batch_size}x{args.num_slices} slices/rank | AMP bf16")

    df = pd.read_csv(args.csv)
    train_df, valid_df = train_test_split(df, test_size=args.val_frac, random_state=42)
    if rank == 0:
        print(f"Volumes: {len(train_df)} train / {len(valid_df)} valid")

    # --- output folder ---
    if args.run_dir:
        # fixed dir: resumable, shared across a resubmission chain
        log_dir = (args.run_dir if os.path.isabs(args.run_dir)
                   else os.path.join(SAVE_ROOT, args.run_dir))
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
    else:
        # legacy: a fresh auto-incrementing folder per run
        if rank == 0:
            k = 0
            while True:
                log_dir = os.path.join(SAVE_ROOT, f"{args.num_exp + k}_radio_{args.save_path}")
                try:
                    os.makedirs(log_dir, exist_ok=False)
                    break
                except FileExistsError:
                    k += 1
            box = [log_dir]
        else:
            box = [None]
        if distributed:
            dist.broadcast_object_list(box, src=0)
        log_dir = box[0]

    model_save_path = os.path.join(log_dir, "model")
    plot_save_path = os.path.join(log_dir, "train_plot.pdf")

    # auto-resume from the last checkpoint in this dir unless --resume was given
    resume = args.resume
    if resume is None and os.path.isfile(model_save_path + "_last.pth"):
        resume = model_save_path + "_last.pth"
    if rank == 0:
        print("Save path:", log_dir, "| resume:", resume or "none")

    model = Radio(dropout_prob=args.dropout, backbone=args.backbone,
                  freeze_blocks=args.freeze_blocks).to(device)
    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable student params: {n_train / 1e6:.1f}M")

    loss_fn = RadioLoss(phis_path=args.phis_path)
    if rank == 0 and not loss_fn.has_phis:
        print(f"WARNING: {args.phis_path} not found - run calibrate_teachers.py first", flush=True)
    collate_fn = make_collate_radio2D(args.num_slices, check_air=not args.pre_cleaned)
    train_set = DatasetRadio2D(train_df)
    valid_set = DatasetRadio2D(valid_df)

    common = dict(batch_size=args.batch_size, collate_fn=collate_fn)
    if torch.cuda.is_available():
        common.update(num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if distributed else None
    valid_sampler = DistributedSampler(valid_set, shuffle=False) if distributed else None
    train_dataloader = DataLoader(train_set, sampler=train_sampler,
                                  shuffle=(train_sampler is None), drop_last=True, **common)
    valid_dataloader = DataLoader(valid_set, sampler=valid_sampler, shuffle=False, **common)

    train_logger = None
    if rank == 0:
        train_logger = TrainingLogger(
            log_dir=log_dir, filename="train_log.csv",
            fieldnames=["Timestamp", "Epoch", "Train_Loss", "Valid_Loss", "LR", "Head_LR",
                        "tr_seg", "tr_curia", "tr_vlm", "va_seg", "va_curia", "va_vlm"])

    train_radio_amp(
        model, train_dataloader, valid_dataloader,
        num_epoch=args.num_epoch, lr=args.lr, w_decay=args.w_decay,
        accum_steps=args.accum_steps, head_lr_mult=args.head_lr_mult,
        find_unused_parameters=args.ddp_find_unused,
        loss_fn=loss_fn, optimizer="AdamW",
        model_save_path=model_save_path, plot_save_path=plot_save_path,
        train_logger=train_logger, resume=resume,
        device=device, rank=rank, is_distributed=distributed,
    )

    # train_radio_amp only returns once every epoch is done -> mark it, so a
    # resubmission chain knows to stop.
    if rank == 0:
        open(os.path.join(log_dir, "COMPLETED"), "w").close()
        print("COMPLETED", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
