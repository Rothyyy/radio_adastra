"""Validate a PHI-S calibration file (models/teacher_phis.pt).

Structural checks (no GPU): per teacher, dtype / shape of mean, rot, scale, and
that rot is orthogonal.

Data check (needs the teachers + ideally a GPU): run a fresh sample through the
teachers, apply the transform, and confirm the standardized spatial features
have ~zero mean and ~unit per-channel std. Uses a different --seed than the
calibration so it's genuinely held out.

    python test_calibration.py                       # structure + 20-batch data check
    python test_calibration.py --n_batches 0         # structure only
"""

import argparse

import torch


def check_structure(st):
    ok = True
    for name, d in st.items():
        mean, rot, scale = d["mean"], d["rot"], d["scale"]
        D = mean.numel()
        ortho = (rot.double() @ rot.double().t()
                 - torch.eye(D, dtype=torch.float64)).abs().max().item()
        good = (mean.dtype == torch.float32 and rot.dtype == torch.float32
                and rot.shape == (D, D) and float(scale) > 0 and ortho < 1e-3)
        ok &= good
        print(f"  {name:6s} D={D:4d}  dtypes {mean.dtype}/{rot.dtype}/{scale.dtype}  "
              f"rot{tuple(rot.shape)}  scale={float(scale):.3f}  "
              f"ortho_err={ortho:.1e}   [{'OK' if good else 'BAD'}]")
    return ok


def check_data(st, args):
    import pandas as pd
    from torch.utils.data import DataLoader

    from radio.dataset_radio import DatasetRadio2D, make_collate_radio2D
    from radio.model_teacher import Teachers
    from radio.train_radio import RadioLoss, TEACHER_NAMES

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loss = RadioLoss(phis_path=args.phis).to(dev).eval()
    teachers = Teachers().to(dev).eval()

    df = pd.read_csv(args.csv).sample(frac=1.0, random_state=args.seed)
    loader = DataLoader(
        DatasetRadio2D(df), batch_size=4,
        collate_fn=make_collate_radio2D(args.num_slices),
        num_workers=4, prefetch_factor=1, drop_last=True,
    )

    acc = {n: [] for n in TEACHER_NAMES}
    with torch.no_grad():
        for k, x in enumerate(loader):
            if k >= args.n_batches:
                break
            x = x.to(dev, dtype=torch.float32)
            with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                                dtype=torch.bfloat16):
                outs = teachers(x)
            for i, name in enumerate(TEACHER_NAMES):
                patch = outs[2 * i + 1]
                if patch is None:
                    continue
                std = loss._phis(name, patch.float()).flatten(0, 1)   # [M, D]
                acc[name].append((std.mean().item(), std.std(0).mean().item()))

    ok = True
    for name, vals in acc.items():
        if not vals:
            continue
        m = sum(v[0] for v in vals) / len(vals)
        s = sum(v[1] for v in vals) / len(vals)
        good = abs(s - 1.0) < 0.15 and abs(m) < 0.1
        ok &= good
        print(f"  held-out {name:6s} mean={m:+.3f}  per-channel std={s:.3f}   "
              f"[{'OK' if good else 'CHECK'}]")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phis", default="models/teacher_phis.pt")
    ap.add_argument("--csv", default="dataset_radio_clean.csv")
    ap.add_argument("--n_batches", type=int, default=20, help="0 = skip the data check")
    ap.add_argument("--num_slices", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123, help="use != calibration seed")
    args = ap.parse_args()

    st = torch.load(args.phis, map_location="cpu")
    print(f"{args.phis}: teachers = {list(st)}")
    ok = check_structure(st)
    if args.n_batches > 0:
        ok &= check_data(st, args)

    print("\nPASS" if ok else "\nFAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
