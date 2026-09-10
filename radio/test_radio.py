"""Evaluate a pretrained RADIO student: mean distillation loss on a held-out
set, plus a per-teacher / per-feature breakdown so you can see which teacher
the student fits best."""

import torch
import torch.nn.functional as F
from torch.nn.functional import normalize

from radio.model_teacher import Teachers
from radio.train_radio import RadioLoss, TEACHER_NAMES

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def test_radio_amp(model, test_dataloader, loss_fn=None):
    criterion = (loss_fn or RadioLoss()).to(device).eval()
    model = model.to(device).eval()
    teachers = Teachers().to(device)

    total, n = 0.0, 0
    breakdown = {f"{name}_{k}": 0.0 for name in TEACHER_NAMES for k in ("cls", "patch")}

    for x in test_dataloader:
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            zs = model(x)
            zt = teachers(x)
            loss = criterion(zs, zt)

        bs = x.size(0)
        total += loss.float().item() * bs
        n += bs

        for i, name in enumerate(TEACHER_NAMES):
            s_cls, s_patch = zs[2 * i], zs[2 * i + 1]
            t_cls, t_patch = zt[2 * i], zt[2 * i + 1]
            if t_cls is not None and s_cls is not None:
                breakdown[f"{name}_cls"] += (
                    1 - F.cosine_similarity(normalize(s_cls.float(), -1),
                                            normalize(t_cls.float(), -1), dim=-1).mean()
                ).item() * bs
            breakdown[f"{name}_patch"] += F.mse_loss(
                normalize(s_patch.float(), -1), normalize(t_patch.float(), -1)
            ).item() * bs

    mean = total / max(n, 1)
    breakdown = {k: v / max(n, 1) for k, v in breakdown.items()}
    print(f"Test distillation loss: {mean:.4f}")
    for k, v in breakdown.items():
        print(f"  {k:16s} {v:.4f}")
    return mean, breakdown
