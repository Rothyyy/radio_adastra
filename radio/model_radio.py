"""RADIO student: one trainable backbone + per-teacher projection heads.

This is the *pretraining* model only — it learns to reproduce the features of
several frozen CT foundation models (the teachers). No classification head; a
downstream head is trained separately later on top of ``backbone``.

The student backbone can be:
  * ``scratch``       - a small timm ViT, random init
  * ``curia``         - the Curia CT foundation model (DINOv2 ViT-B, CT)
  * ``vit_b16`` / ``vit_s16`` - ImageNet-pretrained timm ViT-B/16 or ViT-S/16
  * ``timm:<name>``   - any timm ViT by name

Random init does not converge on this data; use a pretrained backbone.

``forward(x)`` returns ``[cls_seg, patch_seg, cls_curia, patch_curia, cls_vlm,
patch_vlm]`` — the student's prediction of each teacher's features
(``cls_seg`` is ``None``: SAM has no CLS token).
"""

import timm
import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from transformers import AutoModel

from radio.model_mlp import MLP
from radio.model_teacher import CURIA_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"

# Teacher feature widths the projection heads must hit.
SEG_DIM = 768
CURIA_DIM = 768
VLM_DIM = 1152


def zscore_per_image(x, eps=1e-6):
    """[N, H, W] -> per-slice z-score (matches Curia's preprocessing)."""
    m = x.mean(dim=(-2, -1), keepdim=True)
    s = x.std(dim=(-2, -1), keepdim=True)
    return (x - m) / s.clamp_min(eps)


def window01(x, a_min=-125.0, a_max=225.0):
    return ((x - a_min) / (a_max - a_min)).clamp(0.0, 1.0)


class ScratchViT(nn.Module):
    def __init__(self, img_size=512, patch_size=16, embed_dim=640, depth=14,
                 num_heads=10, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=1, num_classes=0,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True,
        )

    def preprocess(self, x):
        return window01(x)

    def forward(self, x):
        out = self.backbone.forward_features(x.unsqueeze(1))  # [N, 1+P, D]
        return out[:, 0], out[:, 1:]


class TimmBackbone(nn.Module):
    """An ImageNet-pretrained timm ViT as the student, adapted to 1-channel / 512 px.

    timm does the adaptation at build time: it sums the RGB patch-embed conv
    weights into one channel (``in_chans=1``) and bicubic-resamples the
    positional embeddings from the checkpoint's native grid to 32x32
    (``img_size=512``). No per-image resizing - your 512 px slices go straight in.
    """

    def __init__(self, name, img_size=512, freeze_blocks=0):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=True, num_classes=0, img_size=img_size, in_chans=1,
        )
        self.embed_dim = self.backbone.embed_dim
        self.n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))

        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        mean = sum(cfg.get("mean", (0.485, 0.456, 0.406))) / 3.0
        std = sum(cfg.get("std", (0.229, 0.224, 0.225))) / 3.0
        self.register_buffer("norm_mean", torch.tensor(float(mean)))
        self.register_buffer("norm_std", torch.tensor(float(std)))

        # classifier-side modules: we only call forward_features, so freeze them
        # (avoids DDP "unused parameter" errors on variants that have real params here)
        for attr in ("head", "head_drop", "fc_norm", "attn_pool", "pre_logits"):
            m = getattr(self.backbone, attr, None)
            if isinstance(m, nn.Module):
                m.requires_grad_(False)

        if freeze_blocks > 0:
            self.backbone.patch_embed.requires_grad_(False)
            self.backbone.cls_token.requires_grad_(False)
            self.backbone.pos_embed.requires_grad_(False)
            for blk in self.backbone.blocks[:freeze_blocks]:
                blk.requires_grad_(False)

    def preprocess(self, x):
        x = window01(x)
        return (x - self.norm_mean) / self.norm_std

    def forward(self, x):
        feats = self.backbone.forward_features(x.unsqueeze(1))  # [N, n_prefix + P, D]
        return feats[:, 0], feats[:, self.n_prefix:]


class CuriaBackbone(nn.Module):
    """Curia (DINOv2) used as a *trainable* student, initialised from CT weights."""

    def __init__(self, freeze_blocks=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CURIA_PATH, local_files_only=True,
                                                  trust_remote_code=True)
        self.embed_dim = self.backbone.config.hidden_size  # 768

        # DINOv2's mask_token is only used for masked-image pretraining; here it
        # never touches the loss, which trips DDP ("parameters that were not used
        # in producing loss"). Freeze it so DDP ignores it.
        mask_token = getattr(self.backbone.embeddings, "mask_token", None)
        if mask_token is not None:
            mask_token.requires_grad_(False)

        if freeze_blocks > 0:
            self.backbone.embeddings.requires_grad_(False)
            for blk in self.backbone.encoder.layer[:freeze_blocks]:
                blk.requires_grad_(False)

    def preprocess(self, x):
        return zscore_per_image(x)

    def forward(self, x):
        out = self.backbone(x.unsqueeze(1))["last_hidden_state"]  # [N, 1025, 768]
        return out[:, 0], out[:, 1:]


TIMM_ALIASES = {
    "vit_b16": "vit_base_patch16_384.augreg_in21k_ft_in1k",
    "vit_s16": "vit_small_patch16_384.augreg_in21k_ft_in1k",
}


class Radio(nn.Module):
    def __init__(self, dropout_prob=0.2, backbone="curia", freeze_blocks=0, img_size=512):
        super().__init__()
        if backbone == "curia":
            self.backbone = CuriaBackbone(freeze_blocks=freeze_blocks)
        elif backbone == "scratch":
            self.backbone = ScratchViT(img_size=img_size)
        elif backbone in TIMM_ALIASES or backbone.startswith("timm:"):
            name = TIMM_ALIASES.get(backbone) or backbone.split("timm:", 1)[1]
            self.backbone = TimmBackbone(name, img_size=img_size, freeze_blocks=freeze_blocks)
        else:
            raise ValueError(f"unknown backbone {backbone!r} "
                             f"(use scratch | curia | vit_b16 | vit_s16 | timm:<name>)")
        d = self.backbone.embed_dim

        self.proj_seg_patch = MLP(d, d, SEG_DIM, dropout_prob)
        self.proj_curia_cls = MLP(d, d, CURIA_DIM, dropout_prob)
        self.proj_curia_patch = MLP(d, d, CURIA_DIM, dropout_prob)
        self.proj_vlm_cls = MLP(d, d, VLM_DIM, dropout_prob)
        self.proj_vlm_patch = MLP(d, d, VLM_DIM, dropout_prob)

    def forward(self, x):
        x = self.backbone.preprocess(x)
        z_cls, z_patch = self.backbone(x)
        return [
            None,
            self.proj_seg_patch(z_patch),
            self.proj_curia_cls(z_cls),
            self.proj_curia_patch(z_patch),
            self.proj_vlm_cls(z_cls),
            self.proj_vlm_patch(z_patch),
        ]

    @torch.no_grad()
    def encode(self, x):
        """Downstream helper: return the raw backbone (CLS, patch) features."""
        return self.backbone(self.backbone.preprocess(x))
