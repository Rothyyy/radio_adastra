import random
import warnings
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class DatasetRadio2D(torch.utils.data.Dataset):
    """Yields whole CT volumes as ``[D, 512, 512]`` float tensors (Hounsfield
    units). Slice sampling happens in the collate fn. Labels are not used —
    this dataset is for self-supervised / distillation pretraining only.

    A volume that fails to load (e.g. a corrupt / pickled ``.npy``) is skipped:
    a different random index is returned instead, so one bad file can't kill a
    multi-hour run. Use ``scan_dataset.py`` to filter the CSV up front.
    """

    def __init__(self, data_csv: Union[str, pd.DataFrame], path_col: str = "nifti_path",
                 transform=None, max_retries: int = 10):
        if isinstance(data_csv, str):
            self.data = pd.read_csv(data_csv)
        else:
            self.data = data_csv.reset_index(drop=True)
        self.path_col = path_col
        self.transform = transform
        self.max_retries = max_retries

    def _load(self, i):
        path = self.data.iloc[i][self.path_col]
        scan = np.load(path)                                   # [H, W, D] in HU
        scan = torch.from_numpy(np.ascontiguousarray(scan)).permute(2, 0, 1).float()
        if scan.shape[1] != 512 or scan.shape[2] != 512:
            scan = resize_input(scan.unsqueeze(1), (512, 512), mode="bilinear").squeeze(1)
        return scan

    def __getitem__(self, i):
        for attempt in range(self.max_retries):
            try:
                return self._load(i)
            except Exception as e:  # corrupt file, missing path, unpicklable array...
                warnings.warn(f"skipping unreadable volume "
                              f"{self.data.iloc[i][self.path_col]}: {type(e).__name__}: {e}")
                i = random.randrange(len(self.data))
        raise RuntimeError(f"could not load a valid volume after {self.max_retries} retries")

    def __len__(self):
        return len(self.data)


def make_collate_radio2D(num_slices: int = 10, hu_threshold: float = -900.0,
                         check_air: bool = True):
    """Return a collate fn that samples ``num_slices`` axial slices per volume
    (without replacement) and stacks them into one ``[B * num_slices, H, W]``
    batch.

    ``check_air=True`` filters near-empty air slices at load time. Set it to
    ``False`` when the dataset was already cleaned by ``preprocess_dataset.py``
    (every slice is informative and this per-step scan is pure overhead).
    """

    def _collate(batch):
        out = []
        for scan in batch:
            if check_air:
                frac_tissue = (scan > hu_threshold).float().mean(dim=(1, 2))
                valid = torch.nonzero(frac_tissue > 0.05, as_tuple=True)[0]
                if valid.numel() < num_slices:
                    valid = torch.arange(scan.shape[0])
            else:
                valid = torch.arange(scan.shape[0])
            idx = valid[torch.randperm(valid.numel())[:num_slices]].sort().values
            out.append(scan[idx])
        return torch.vstack(out)

    return _collate


# Back-compat default (10 slices / volume).
collate_radio2D = make_collate_radio2D(10)


def resize_input(x, shape_size: tuple, mode: str = "trilinear"):
    """Resize ``x`` to ``shape_size`` with ``F.interpolate``."""
    return F.interpolate(x, size=shape_size, mode=mode).squeeze(0)
