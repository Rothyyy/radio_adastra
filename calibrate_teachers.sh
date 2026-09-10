#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=cad17796
#SBATCH --constraint=MI250
#SBATCH --job-name=CalibTeachers
#SBATCH --output=calibrate_teachers.log
#SBATCH --time=02:00:00
#SBATCH --exclusive                  # need the whole node's RAM: each dataloader
#SBATCH --mem=0                      # worker holds full ~1 GB CT volumes in flight
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64

set -euo pipefail

source /lus/work/CT3/cad17796/rsochet/.rs_env/bin/activate
cd /lus/work/CT3/cad17796/rsochet/

export OMP_NUM_THREADS=8
export MIOPEN_USER_DB_PATH="/tmp/miopen-${USER}-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

# One pass over the frozen teachers -> teacher_phis.pt (PHI-S standardization for
# the spatial distillation loss).
#
# Sizing: we fit a per-teacher covariance (D = 768 / 1152). Aim for >= ~1000 * D
# patch tokens AND wide scan coverage (slices within a scan are highly
# correlated, so many scans matter more than many tokens).
#   batch_size 8 * num_slices 4 = 32 slices from 8 scans per batch
#   1500 batches -> 12000 scans (~half the dataset), 48k slices, ~49M tokens
# The log prints per-teacher standardized variance (target ~1.0); raise
# --n_batches if min/max stray far from 1.
#
# RAM: peak ~= num_workers * batch_size * ~1 GB (a full CT volume per in-flight
# sample). 6 * 8 ~= 48 GB, fine on an exclusive node.
python calibrate_teachers.py \
    --csv dataset_radio_clean.csv \
    --out models/teacher_phis.pt \
    --n_batches 1500 \
    --batch_size 8 --num_slices 4 \
    --num_workers 6 \
    --seed 0
