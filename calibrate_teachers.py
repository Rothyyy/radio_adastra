"""One pass over the frozen teachers to fit the PHI-S standardization transform.

For each teacher's spatial/patch features X (dim D) we estimate the mean mu and
covariance Sigma over a sample of the data, then build

    X' = R (X - mu) / s          with   R = R_rot @ U^T   (orthogonal, D x D)
                                         Sigma = U diag(L) U^T
                                         s = sqrt(mean(L))

R_rot is a fixed random orthogonal matrix (PHI-random; a Hadamard matrix would
need D a power of 2, which 768 / 1152 are not). The PCA rotation U^T decorrelates
and R_rot spreads variance evenly, so every channel of X' ends up ~unit variance.
R is orthogonal and s a scalar, so MSE in X'-space is proportional to MSE in raw
feature space - just balanced across channels and teachers.

    python calibrate_teachers.py --csv dataset_radio_clean.csv \
        --out teacher_phis.pt --n_batches 200 --num_slices 10 --seed 0

Output: teacher_phis.pt = {teacher: {"mean": [D], "rot": [D,D], "scale": []}}.
Re-run only if the teachers or the data distribution change.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from radio.dataset_radio import DatasetRadio2D, make_collate_radio2D
from radio.model_teacher import Teachers
from radio.train_radio import TEACHER_NAMES

device = "cuda" if torch.cuda.is_available() else "cpu"


class RunningMoments:
    """Streaming mean/covariance, float64 on the feature's own device.

    Accumulates about a fixed shift (the first batch mean) so ``sum(x x^T)`` never
    suffers catastrophic cancellation, then folds it back in ``mean_cov``.
    """

    def __init__(self, dim, device):
        self.n = 0
        self.shift = None
        self.s1 = torch.zeros(dim, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def update(self, x):                       # x: [M, D]
        x = x.double()
        if self.shift is None:
            self.shift = x.mean(0)
        x = x - self.shift
        self.n += x.shape[0]
        self.s1 += x.sum(0)
        self.s2 += x.t() @ x

    def mean_cov(self):
        m = self.s1 / self.n
        mu = (self.shift + m).cpu()
        cov = (self.s2 / self.n - torch.outer(m, m)).cpu()
        return mu, 0.5 * (cov + cov.t())       # symmetrize


def phis_transform(mu, cov, dim, generator):
    """Everything computed in float64, returned as float32 (the dtype the loss
    applies it in). cov must be float64."""
    cov = cov.double()
    evals, evecs = torch.linalg.eigh(cov)                 # cov = evecs diag(evals) evecs^T
    evals = evals.clamp_min(1e-8)
    r_rand, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.float64, generator=generator))
    rot = r_rand @ evecs.t()                              # [D, D] orthogonal, float64
    scale = evals.mean().sqrt()                           # global RMS, float64

    # sanity: per-channel variance of the standardized features (want ~1.0)
    std_var = (rot @ cov @ rot.t()).diagonal() / scale.pow(2)
    ortho_err = (rot @ rot.t() - torch.eye(dim, dtype=torch.float64)).abs().max()

    return (mu.float(), rot.float(), scale.float().reshape(()),
            {"var_mean": float(std_var.mean()), "var_min": float(std_var.min()),
             "var_max": float(std_var.max()), "ortho_err": float(ortho_err)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_radio_clean.csv")
    ap.add_argument("--out", default="teacher_phis.pt")
    ap.add_argument("--n_batches", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_slices", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = pd.read_csv(args.csv).sample(frac=1.0, random_state=args.seed)
    loader = DataLoader(
        DatasetRadio2D(df), batch_size=args.batch_size,
        collate_fn=make_collate_radio2D(args.num_slices),
        num_workers=args.num_workers, drop_last=True,
        prefetch_factor=1 if args.num_workers > 0 else None,   # cap in-flight full volumes
    )

    teachers = Teachers().to(device).eval()
    moments = {}

    seen = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device, dtype=torch.float32)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=torch.bfloat16):
                outs = teachers(x)                         # [seg_cls, seg_p, cur_cls, cur_p, ...]
            for i, name in enumerate(TEACHER_NAMES):
                patch = outs[2 * i + 1]
                if patch is None:
                    continue
                feat = patch.float().flatten(0, 1)         # [N*P, D]
                if name not in moments:
                    moments[name] = RunningMoments(feat.shape[-1], feat.device)
                moments[name].update(feat)
            seen += 1
            if seen % 20 == 0:
                tok = next(iter(moments.values())).n
                print(f"  {seen}/{args.n_batches} batches  ({tok:,} tokens/teacher)", flush=True)
            if seen >= args.n_batches:
                break

    gen = torch.Generator().manual_seed(args.seed)
    out, checks = {}, {}
    for name, m in moments.items():
        mu, cov = m.mean_cov()
        d = mu.shape[0]
        mean, rot, scale, chk = phis_transform(mu, cov, d, gen)
        out[name] = {"mean": mean, "rot": rot, "scale": scale}   # all float32
        checks[name] = (m.n, float(scale), chk)

    # save first, so a reporting bug can never throw away the calibration pass
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = f"{args.out}.tmp{os.getpid()}"
    torch.save(out, tmp)
    os.replace(tmp, args.out)          # atomic
    print("saved", args.out, flush=True)

    for name, (n, scale, c) in checks.items():
        print(f"{name}: D={out[name]['mean'].numel()}  samples={n:,}  scale={scale:.3f}  "
              f"standardized var {c['var_mean']:.3f} "
              f"(min {c['var_min']:.3f}, max {c['var_max']:.3f})  "
              f"ortho_err={c['ortho_err']:.1e}")


if __name__ == "__main__":
    main()
