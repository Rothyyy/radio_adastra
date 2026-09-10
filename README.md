# RADIO-CT — agglomerative distillation of medical foundation models into one CT encoder

This repo adapts **RADIOv2.5** ([arxiv 2412.07679](https://arxiv.org/abs/2412.07679),
itself building on [AM-RADIO](https://arxiv.org/abs/2312.06709)) to CT imaging.

RADIO trains a **single student vision encoder** to reproduce, in one forward
pass, the features of several **frozen "teacher"** foundation models. The
original paper distils natural-image models (DINOv2, CLIP, SAM). Here the
teachers are **medical/CT models** — MedSAM, Curia, MedSigLIP — so the student
becomes a general-purpose CT slice encoder that inherits segmentation-quality
dense features, DINO-style semantic features, and vision-language semantics at
once.

This repo does **representation pretraining only**. There is no classification
head — downstream heads are trained separately on top of the pretrained
backbone.

---

## 1. Architecture

```
          raw CT slices  x : [N, 512, 512]  (Hounsfield units)
                 │
     ┌───────────┴─────────────┐
     ▼                         ▼
  STUDENT  (trainable)      TEACHERS  (frozen, no_grad, bf16)
  backbone + 5 MLP heads    MedSAM · Curia · MedSigLIP
     │                         │
     ▼                         ▼
  predicted teacher features   real teacher features
     └──────────┬──────────────┘
                ▼
       RadioLoss  (cosine on summary tokens, MSE on PHI-S-standardized patch tokens)
```

Every teacher emits a **32×32 = 1024 patch-token grid**, which is why the
student runs at **512 px with patch 16** (512/16 = 32). The spatial distillation
targets then line up 1:1 with no interpolation.

### 1.1 Teachers — `radio/model_teacher.py`

| key | model | HF class | role | summary token | patch tokens | input prep |
|---|---|---|---|---|---|---|
| `seg` | **MedSAM** ViT-B | `SamModel.vision_encoder` | dense / segmentation | – (SAM has no CLS) | 64×64×768 → avg-pool 2× → **1024×768** | soft-tissue window → 1024 px → 3-ch → ImageNet norm |
| `curia` | **Curia** (DINOv2 ViT-B, CT) | `Dinov2Model` | general CT features | CLS **768** | **1024×768** | Curia's own processor (per-image z-score on raw HU) |
| `vlm` | **MedSigLIP-448** vision tower | `SiglipModel.vision_model` | vision–language semantics | pooler **1152** | **1024×1152** | soft-tissue window → 448 px → 3-ch → SigLIP norm `[-1,1]` |

`Teachers(use_seg=…, use_curia=…, use_vlm=…)` lets you disable any teacher.
Weights are loaded from `$RADIO_MODEL_ROOT` (default
`/lus/work/CT3/cad17796/rsochet/models/{medsam,curia,medsiglip}`), `local_files_only`.

### 1.2 Student — `radio/model_radio.py`

`Radio(backbone=…)` = **one trainable backbone** + **five MLP projection heads**
(`radio/model_mlp.py`, `Linear→LayerNorm→ReLU→Dropout→Linear`):

```
proj_seg_patch   : d → 768     proj_curia_cls   : d → 768
proj_curia_patch : d → 768     proj_vlm_cls     : d → 1152
proj_vlm_patch   : d → 1152
```

`forward(x)` → `[None, patch_seg, cls_curia, patch_curia, cls_vlm, patch_vlm]`
(the student's *predictions* of each teacher feature).

Backbone options (`--backbone`):

| value | backbone | init | params (backbone) | notes |
|---|---|---|---|---|
| `scratch` | timm `VisionTransformer` 640-d / depth 14 | **random** | ~70 M | does **not** converge on this data — kept for comparison only |
| `curia` | Curia DINOv2 ViT-B (768-d / 12) | **CT foundation model** | ~86 M | strongest prior; also one of the teachers |
| `vit_b16` | timm `vit_base_patch16_384.augreg_in21k_ft_in1k` | **ImageNet-21k→1k** | ~86 M | generic pretrained, 768-d matches teachers |
| `vit_s16` | timm `vit_small_patch16_384.augreg_in21k_ft_in1k` | **ImageNet-21k→1k** | ~22 M | 3× cheaper, good for iterating |
| `timm:<name>` | any timm ViT | pretrained | – | e.g. `timm:vit_base_patch16_224.mae` |

For the timm backbones, timm adapts at build time: sums the RGB patch-embed conv
weights into **1 channel** (`in_chans=1`) and bicubic-resamples the positional
embeddings to the 32×32 grid (`img_size=512`). No per-image resizing.

Each backbone owns its `preprocess()` (raw HU → model's expected input) so the
student only ever sees raw Hounsfield slices.

---

## 2. The loss — `radio/train_radio.py :: RadioLoss`

Follows the **RADIOv2.5** formulation (paper Table A9):

```
per teacher t:
    summary term :  1 - cos( student_cls ,  teacher_cls )          # skipped for MedSAM
    spatial term :  MSE( student_patch ,  PHI-S_t( teacher_patch ) )
total = mean_t ( summary_weight · summary  +  spatial_weight · spatial )
```

### PHI-S standardization

Raw teacher features sit on very different scales, so a naive MSE would let one
teacher dominate. **PHI-S** fixes this at the *feature* level: per teacher,

```
X' = R (X - μ) / s        R orthogonal  (D×D),   s scalar
```

- `μ` — per-channel mean
- `R = R_random · Uᵀ` where `Σ = U Λ Uᵀ` — PCA rotation (decorrelate) then a fixed
  random rotation (spread variance evenly across channels)
- `s = √(mean eigenvalue)` — one scalar → every channel ends at ~unit variance

Because `R` is orthogonal and `s` scalar, `‖X'ₐ − X'_b‖² = ‖Xₐ − X_b‖²/s²` — the
feature geometry is preserved, just re-expressed in a balanced basis. After this,
a **plain equally-weighted sum of MSEs is already balanced** (paper's `λ_t = 1`),
and the logged per-teacher losses are all comparable and can be read directly.

> Deviation from the paper: it uses a Hadamard matrix for the variance-spreading
> rotation; we use a random orthogonal one because D = 768 / 1152 aren't powers
> of 2. The PHI-S paper evaluates this "PHI-random" variant and shows it captures
> nearly all the benefit.

The transform (`μ`, `R`, `s` per teacher) is fitted **once, offline** by
`calibrate_teachers.py` → `models/teacher_phis.pt`, and loaded as buffers. If the
file is missing, `RadioLoss` falls back to raw unbalanced MSE with a warning
(fine for a quick eval, not for real training).

`_loss_ver` bumps whenever the loss formula changes so a resumed run re-inits the
loss state instead of loading a stale one.

---

## 3. Data

`dataset_radio_clean.csv` — two columns, `id` and `nifti_path` — one row per CT
volume. Volumes are `.npy` arrays, `[H, W, D]` float16, **Hounsfield units**
(the Merlin extraction; mostly already 512×512). No labels.

`radio/dataset_radio.py`:
- **`DatasetRadio2D`** yields a whole volume `[D, 512, 512]` per item (resizes
  H×W to 512 if needed). A file that fails to load is skipped (random resample)
  so one corrupt `.npy` can't kill a multi-hour run.
- **`make_collate_radio2D(num_slices, check_air=True)`** samples `num_slices`
  informative axial slices per volume (drops near-air slices) and stacks them:
  `[B·num_slices, 512, 512]`.

The train/valid split is done in `train_radio_script.py` with
`sklearn.train_test_split(random_state=42, test_size=--val_frac)` — deterministic,
so the split is identical across every job in a resubmission chain.

---

## 4. Pipeline

```
 (0)  download teacher weights           download_models.py        (once, login node)
 (0)  download timm backbone weights     download_backbone.py      (once, if using vit_b16/vit_s16)
 ──────────────────────────────────────────────────────────────────
 (1)  validate the dataset CSV           preprocess_dataset.sh  →  dataset_radio_clean.csv
 (2)  fit PHI-S transform                calibrate_teachers.sh  →  models/teacher_phis.pt
 (3)  train                              train_script_multi_GPU.sh
 ──────────────────────────────────────────────────────────────────
 (opt) shape audit                       check_shape.py
 (opt) validate the calibration          test_calibration.py
 (opt) evaluate a checkpoint             radio/test_radio.py
```

### (1) Dataset validation — `preprocess_dataset.py` / `.sh`
Parallel-loads every `.npy` header, splits the CSV into a clean file and a
`.bad.csv` (with reasons: pickled/object array, wrong ndim, truncated). Run once
after building the raw CSV.

### (2) PHI-S calibration — `calibrate_teachers.py` / `.sh`
One pass over the frozen teachers. Streams patch features, accumulates a
per-teacher mean + covariance (float64, GPU, mean-shifted for stability),
eigendecomposes, and writes `models/teacher_phis.pt`. Sizing rule of thumb:
**≥ ~1000·D tokens and wide scan coverage** — slices within a scan are highly
correlated, so *many scans* matters more than *many tokens*. The sbatch default
(`--n_batches 1500 --batch_size 8 --num_slices 4`) covers ~12 k scans / ~49 M
tokens. The log prints per-teacher standardized variance — should be ≈ 1.0.
Re-run only if the teachers or data distribution change.

### (3) Training — `train_radio_script.py` + `train_script_multi_GPU.sh`

- **Launch**: one `torchrun` per node, **8 ranks / node = 1 rank per MI250 GCD**,
  `c10d` rendezvous. Bump `#SBATCH --nodes` to scale to 8·N ranks.
- **DDP**: `DistributedSampler` + `DistributedDataParallel`. `--batch_size` is
  **per-GPU**; global batch = `batch_size · 8 · nodes`. `no_sync()` skips the
  all-reduce on gradient-accumulation micro-steps.
- **Optimizer**: AdamW, **two param groups** — backbone at `--lr`, the five fresh
  projection heads at `--lr · --head_lr_mult`. Cosine schedule with linear
  warmup (`radio/train_utils.py`). `grad_clip=1.0`. bf16 autocast.
- **Checkpointing** (rank 0): `model_last.pth` every epoch, `model_best_valid.pth`
  on improvement, `train_log.csv`, `train_plot.pdf`.
- **Resumable chain**: `--run_dir` fixes the output folder. On start the job
  auto-resumes from `model_last.pth`; when all epochs finish it writes a
  `COMPLETED` sentinel. `train_script_multi_GPU.sh`:
  - queues its own successor with `--dependency=afterany` (survives timeout,
    crash, node failure),
  - the successor exits immediately if `COMPLETED` exists,
  - `.chain_count` caps the chain at `MAX_RESUBMITS` (default 30).
  Stop a chain: `scancel --me --name PretrainRadio`, or `touch <run_dir>/COMPLETED`.

Experiment knobs live in the config block at the top of
`train_script_multi_GPU.sh` (`BACKBONE`, `LR`, `HEAD_LR_MULT`, `NUM_EPOCH`).
`RUN_NAME` is derived from `BACKBONE`, so each backbone gets its own
`save_model/pretrain_<backbone>/` — no collisions.

---

## 5. File map

| file | purpose |
|---|---|
| **`radio/model_teacher.py`** | the 3 frozen teachers + their CT preprocessing |
| **`radio/model_radio.py`** | student: backbone variants + projection heads (`Radio`) |
| **`radio/model_mlp.py`** | the projection-head MLP |
| **`radio/train_radio.py`** | `RadioLoss` (PHI-S), the train/eval loop, `train_radio_amp` (DDP, checkpointing, resume) |
| **`radio/train_radio_script.py`** | CLI entry point: arg parsing, DDP init, data loaders, run-dir logic |
| **`radio/dataset_radio.py`** | `DatasetRadio2D`, slice-sampling collate |
| **`radio/train_utils.py`** | LR scheduler, best-metric tracking, loss plot |
| **`radio/logger.py`** | append-mode CSV metric logger |
| **`radio/test_radio.py`** | held-out distillation-loss eval with per-teacher breakdown |
| `calibrate_teachers.py` / `.sh` | fit the PHI-S transform → `models/teacher_phis.pt` |
| `test_calibration.py` | validate a PHI-S file (structure + held-out unit-variance check) |
| `preprocess_dataset.py` / `.sh` | CSV health check → `dataset_radio_clean.csv` (+ `.bad.csv`) |
| `check_shape.py` | fast per-volume shape table (mmap headers only) |
| `download_models.py` | reference snippets (commented) for fetching the 3 teacher checkpoints — the weights are already in `models/` |
| `download_backbone.py` | pre-fetch timm backbone weights into the HF cache (login node) |
| `train_script_multi_GPU.sh` | **main** sbatch: calibration fallback + torchrun training + resubmission chain |
| `train_script.sh` | single-GPU sbatch (debugging) |
| `prepare_dataset.py` | build the initial raw CSV from a directory of `.npy` files |

---

## 6. Quick start (Adastra / MI250)

```bash
source /lus/work/CT3/cad17796/rsochet/.rs_env/bin/activate

# --- once, on the login node ---
# teacher weights are already in models/{medsam,curia,medsiglip}.
python download_backbone.py                      # timm ViT-B/16 + ViT-S/16 (only if using them)

# --- prepare ---
sbatch preprocess_dataset.sh                     # GENOA CPU node -> dataset_radio_clean.csv (+ .bad.csv)
sbatch calibrate_teachers.sh                     # 1 MI250 node   -> models/teacher_phis.pt
python test_calibration.py                       # sanity check (optional)

# --- train ---
# edit BACKBONE / LR / NUM_EPOCH at the top of train_script_multi_GPU.sh
sbatch train_script_multi_GPU.sh                 # 8-GPU DDP, self-resubmitting
```

Outputs land in `save_model/pretrain_<backbone>/`:
`model_last.pth`, `model_best_valid.pth`, `train_log.csv`, `train_plot.pdf`,
`COMPLETED`, `.chain_count`.

`train_log.csv` columns: `Epoch, Train_Loss, Valid_Loss, LR, Head_LR,
tr_{seg,curia,vlm}, va_{seg,curia,vlm}` — the last six are the per-teacher raw
losses (`summary_cos + spatial_mse`), all on a comparable scale thanks to PHI-S.

---

## 7. Environment

- **Cluster**: CINES Adastra, MI250 nodes (8 GCDs = 8 "GPUs"), ROCm / `torch` ROCm build in venv `.rs_env`.
- Teacher + backbone weights are local; compute nodes run **offline** (`HF_HUB_OFFLINE=1`).
- `RADIO_MODEL_ROOT` — teacher weights dir (default `…/rsochet/models`).
- `RADIO_SAVE_ROOT` — checkpoint dir (default `…/rsochet/save_model`).

## 8. Known limitations / gotchas

- **Data loading is the bottleneck.** `DatasetRadio2D` reads a whole ~230 MB
  volume to use a handful of slices. This dominates step time and forces
  `--exclusive` for the calibration job (RAM). A mmap/slice-subsample
  preprocessing pass is the intended fix but not yet implemented.
- **`scratch` backbone does not converge** on 25 k CT scans — always start from a
  pretrained backbone.
- **MedSigLIP patch tokens are nearly collinear** (pairwise cos ≈ 0.94); its
  useful signal is the pooler/summary token. PHI-S balancing handles it, but
  `proj_vlm_patch` is the weakest distillation target.
- **`valid < train` is expected**: Curia's `drop_path_rate=0.4` + head dropout
  are active in train, off in eval; and train loss is an epoch average vs an
  end-of-epoch valid model.
- The calibration `--seed` and the training split `random_state=42` are fixed —
  keep them stable so a resubmission chain stays consistent.
- `download_models.py` still contains a hard-coded HuggingFace token in plaintext
  — revoke it and use `huggingface-cli login` / `$HF_TOKEN` instead.
- `radio/README.md`, `prepare_dataset.py`, `radio/dataset_utils.py` are older /
  partly superseded; this file is the current reference.
