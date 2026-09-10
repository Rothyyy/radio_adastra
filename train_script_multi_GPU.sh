#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=cad17796
#SBATCH --constraint=MI250
#SBATCH --job-name=PretrainRadio
#SBATCH --output=train_radio_pretrain_new_loss.log
#SBATCH --open-mode=append          # chain jobs append to one log instead of clobbering
#SBATCH --time=24:00:00
#SBATCH --exclusive
# One MI250 module = 2 GCDs; an Adastra MI250 node exposes 8 GCDs = 8 "GPUs".
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64

set -euo pipefail

SCRIPT=/lus/work/CT3/cad17796/rsochet/train_script_multi_GPU.sh
CLEAN_CSV=dataset_radio_clean.csv

# --- experiment config ---------------------------------------------------------
# BACKBONE: scratch | curia | vit_b16 | vit_s16 | timm:<name>
#   vit_b16 / vit_s16 = ImageNet ViT-B/16 or ViT-S/16 (augreg in21k->in1k, 384).
#   For those, first fetch weights on the LOGIN node:
#       python download_backbone.py
BACKBONE=vit_b16
LR=1e-4                             # backbone LR (fine-tuning a pretrained model)
HEAD_LR_MULT=5                      # projection heads train from scratch -> higher LR
NUM_EPOCH=80
# -----------------------------------------------------------------------------

RUN_NAME=pretrain_$(echo "${BACKBONE}" | tr ':/' '__')
RUN_DIR=/lus/work/CT3/cad17796/rsochet/save_model/${RUN_NAME}
MAX_RESUBMITS=30                     # safety cap: stop the chain after this many jobs

source /lus/work/CT3/cad17796/rsochet/.rs_env/bin/activate
cd /lus/work/CT3/cad17796/rsochet/
mkdir -p "${RUN_DIR}"

# ---------------------------------------------------------------------------
# Resubmission chain
#   * training writes ${RUN_DIR}/COMPLETED when all epochs are done  -> stop
#   * otherwise queue a successor that starts when THIS job ends (any reason)
#     and resumes from ${RUN_DIR}/model_last.pth
# ---------------------------------------------------------------------------
if [ -f "${RUN_DIR}/COMPLETED" ]; then
    echo "$(date '+%F %T')  training COMPLETED - not resubmitting"
    exit 0
fi

CC="${RUN_DIR}/.chain_count"
n=$(cat "${CC}" 2>/dev/null || echo 0)
[[ "${n}" =~ ^[0-9]+$ ]] || n=0
if [ "${n}" -ge "${MAX_RESUBMITS}" ]; then
    echo "$(date '+%F %T')  reached MAX_RESUBMITS=${MAX_RESUBMITS} - stopping chain"
    exit 1
fi
echo $((n + 1)) > "${CC}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" "${SCRIPT}" || true)
    echo "$(date '+%F %T')  chain ${n} -> queued successor job ${NEXT:-<failed>}"
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
GPUS_PER_NODE=8
NNODES=${SLURM_NNODES:-1}
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=29517
export MASTER_ADDR MASTER_PORT

export OMP_NUM_THREADS=8
export MIOPEN_USER_DB_PATH="/tmp/miopen-${USER}-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
# export NCCL_DEBUG=WARN

# ---------------------------------------------------------------------------
# Step 0 - PHI-S calibration: one pass over the frozen teachers to fit the
# spatial-feature standardization. Skipped once the file exists (normally you
# run calibrate_teachers.sh separately; this is the fallback).
# ---------------------------------------------------------------------------
PHIS=models/teacher_phis.pt
if [ ! -f "${PHIS}" ]; then
    srun --ntasks=1 --cpu-bind=none \
        python calibrate_teachers.py --csv "${CLEAN_CSV}" --out "${PHIS}" \
            --n_batches 800 --batch_size 4 --num_slices 4 --num_workers 6
fi

LAUNCHER="python -m torch.distributed.run \
    --nnodes=${NNODES} \
    --nproc-per-node=${GPUS_PER_NODE} \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

TRAIN_ARGS="\
    --csv ${CLEAN_CSV} \
    --run_dir ${RUN_DIR} \
    --backbone ${BACKBONE} \
    --phis_path ${PHIS} \
    --lr ${LR} --head_lr_mult ${HEAD_LR_MULT} --w_decay 1e-3 --dropout 0.2 \
    --batch_size 2 --num_slices 10 --accum_steps 1 \
    --num_epoch ${NUM_EPOCH} \
    --num_workers 4"

srun --kill-on-bad-exit=1 --cpu-bind=none \
    ${LAUNCHER} -m radio.train_radio_script ${TRAIN_ARGS}
