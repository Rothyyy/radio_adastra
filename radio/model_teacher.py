"""Frozen foundation-model teachers for RADIO distillation on CT.

Each teacher takes raw CT slices ``x`` of shape ``[N, H, W]`` in Hounsfield
units and returns ``(summary_token, patch_tokens)``:

    * summary_token : ``[N, C_t]`` or ``None`` (SAM has no CLS token)
    * patch_tokens  : ``[N, 1024, C_t]``  (32x32 grid, matching a 512px / 16px
      student so the spatial distillation targets line up 1:1)

Teachers are always run under ``no_grad`` in bf16 autocast and never receive
gradients.  Preprocessing is done explicitly here (windowing + resize +
normalization) instead of relying on the HF processors, which behave
inconsistently when handed a batched tensor.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForMaskGeneration,
    AutoModelForZeroShotImageClassification,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ROOT = os.environ.get("RADIO_MODEL_ROOT", "/lus/work/CT3/cad17796/rsochet/models")
SEG_PATH = os.path.join(MODEL_ROOT, "medsam")
CURIA_PATH = os.path.join(MODEL_ROOT, "curia")
VLM_PATH = os.path.join(MODEL_ROOT, "medsiglip")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def window_ct(x, center=40.0, width=400.0):
    """Apply a CT window and scale to [0, 1]. Default = soft-tissue window."""
    lo, hi = center - width / 2.0, center + width / 2.0
    return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)


def to_3ch_resized(x01, size):
    """[N, H, W] in [0,1] -> [N, 3, size, size]."""
    x = x01.unsqueeze(1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.repeat(1, 3, 1, 1)


class Teachers(nn.Module):
    def __init__(self, use_seg=True, use_curia=True, use_vlm=True):
        super().__init__()
        self.Seg = SegTeacher() if use_seg else None
        self.Curia = CuriaTeacher() if use_curia else None
        self.VLM = VLMTeacher() if use_vlm else None
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        seg_cls, seg_tok = self.Seg(x) if self.Seg is not None else (None, None)
        curia_cls, curia_tok = self.Curia(x) if self.Curia is not None else (None, None)
        vlm_cls, vlm_tok = self.VLM(x) if self.VLM is not None else (None, None)
        return [seg_cls, seg_tok, curia_cls, curia_tok, vlm_cls, vlm_tok]


class SegTeacher(nn.Module):
    """MedSAM ViT-B image encoder (1024px, 64x64 tokens -> pooled to 32x32)."""

    def __init__(self):
        super().__init__()
        model = AutoModelForMaskGeneration.from_pretrained(SEG_PATH, local_files_only=True)
        self.backbone = model.vision_encoder.eval()
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)

    @torch.no_grad()
    def forward(self, x):
        x = window_ct(x)
        x = to_3ch_resized(x, 1024).to(self.mean.dtype)
        x = (x - self.mean) / self.std
        x = x.to(self.mean.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(x, output_hidden_states=True)
        tokens = out["hidden_states"][-1]          # [N, 64, 64, 768]
        tokens = tokens.permute(0, 3, 1, 2)
        tokens = F.avg_pool2d(tokens, kernel_size=2, stride=2)   # [N, 768, 32, 32]
        tokens = tokens.flatten(2).transpose(1, 2).contiguous()  # [N, 1024, 768]
        return None, tokens


class CuriaTeacher(nn.Module):
    """Curia DINOv2 CT foundation model (512px, 32x32 tokens, 768-d)."""

    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(
            CURIA_PATH, local_files_only=True, trust_remote_code=True
        )
        self.backbone = AutoModel.from_pretrained(
            CURIA_PATH, local_files_only=True, trust_remote_code=True
        ).eval()

    @torch.no_grad()
    def forward(self, x):
        # Curia's own processor does per-image z-score on raw HU; feed it [H, W, N].
        pixel_values = self.processor(x.permute(1, 2, 0).cpu().numpy())["pixel_values"]
        pixel_values = pixel_values.squeeze(0).to(next(self.backbone.parameters()).device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(pixel_values)["last_hidden_state"]
        return out[:, 0, :], out[:, 1:, :]


class VLMTeacher(nn.Module):
    """MedSigLIP-448 vision tower (448px / 14px -> 32x32 tokens, 1152-d)."""

    def __init__(self):
        super().__init__()
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            VLM_PATH, local_files_only=True
        )
        self.backbone = model.vision_model.eval()

    @torch.no_grad()
    def forward(self, x):
        x = window_ct(x)
        x = to_3ch_resized(x, 448)
        x = (x - 0.5) / 0.5                        # SigLIP normalization -> [-1, 1]
        x = x.to(next(self.backbone.parameters()).device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(x)
        return out["pooler_output"], out["last_hidden_state"]


if __name__ == "__main__":
    print("Loading teachers...")
    t = Teachers().to(device)
    dummy = torch.randn(2, 512, 512, device=device) * 200
    outs = t(dummy)
    for name, o in zip(["seg_cls", "seg_tok", "curia_cls", "curia_tok", "vlm_cls", "vlm_tok"], outs):
        print(name, None if o is None else tuple(o.shape))
