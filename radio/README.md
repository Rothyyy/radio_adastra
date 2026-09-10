# Radio — agglomerative distillation for CT

Adaptation of RADIOv2.5 (https://arxiv.org/pdf/2412.07679) to CT imaging: a
single student backbone is distilled from several frozen **medical foundation
models** acting as teachers.

## Teachers (`model_teacher.py`)

| name  | model        | role                     | summary | patch tokens |
|-------|--------------|--------------------------|---------|--------------|
| seg   | MedSAM ViT-B | dense / segmentation     | –       | 32×32 × 768  |
| curia | Curia (DINOv2, CT) | general CT features | CLS 768 | 32×32 × 768  |
| vlm   | MedSigLIP-448 | vision–language semantics | 1152   | 32×32 × 1152 |

All teachers are run frozen, `no_grad`, bf16. Each does its own CT windowing /
resize / normalization internally so the student only ever sees raw Hounsfield
slices. Every teacher yields a 32×32 token grid so the spatial targets line up
1:1 with a 512 px / 16 px student.

## Student (`model_radio.py`)

`Radio(backbone=...)`:

* `backbone="curia"` (default) — initialise the trainable student from the Curia
  CT foundation model. Strong prior, fast convergence on small datasets.
* `backbone="scratch"` — the original small timm ViT (640-d, depth 14).

Five lightweight MLP heads project the student embedding onto each teacher's
summary / patch feature space. There is **no classification head** — this repo
only does representation pretraining; a downstream head is trained separately
later on `model.backbone` (see `Radio.encode`).

## Loss (`train_radio.py::RadioLoss`)

* summary features: `1 - cos` + smooth-L1 on L2-normalized vectors
* patch features: MSE on L2-normalized tokens
* teacher balancing: each teacher's raw loss is divided by an EMA of itself, so
  no teacher dominates (lightweight stand-in for the paper's PHI-S balancing).

## Run

```bash
python -m radio.train_radio_script \
    --num_exp 1 --backbone curia \
    --lr 1e-4 --w_decay 1e-3 --dropout 0.2 \
    --batch_size 2 --num_slices 10 --accum_steps 2 --num_epoch 80 \
    --save_path train_2d_models
```

Checkpoints (`model_last.pth`, `model_best_valid.pth`), `train_log.csv` and the
loss plot are written to
`/lus/work/CT3/cad17796/SHARED/radio/save_model/<exp>/`. Resume with
`--resume <path>/model_last.pth`.

Set `RADIO_MODEL_ROOT` to override the teacher-weights directory.

## Data

`dataset_radio.py` expects a CSV with a `nifti_path` column pointing at `.npy`
volumes (`[H, W, D]` in HU). No `label` column is needed — pretraining is
label-free. `make_collate_radio2D(n)` samples `n` non-air axial slices per
volume per step.
