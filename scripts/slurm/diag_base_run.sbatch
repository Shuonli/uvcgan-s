#!/bin/bash
# Train a base model far enough that the batch size measurement can be
# repeated from the middle of training rather than from initialization.
# Checkpoints are kept so that other runs can branch off them.
#
#   sbatch scripts/slurm/diag_base_run.sbatch
#
# A repeat with another seed needs its own label, e.g.
#
#   DIAG_SEED=1 DIAG_LABEL=base_b32_lr5e-5_s1 \
#       sbatch scripts/slurm/diag_base_run.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=26:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=4
export TMPDIR=/tmp

export DIAG_SUBDIR=base
export DIAG_BATCH=${DIAG_BATCH:-32}
export DIAG_LR=${DIAG_LR:-5e-5}
export DIAG_STEPS=${DIAG_STEPS:-200}
export DIAG_CHECKPOINT=${DIAG_CHECKPOINT:-50}   # every 10k updates
export DIAG_EPOCHS=100000                       # stopped by the time limit
export DIAG_LABEL=${DIAG_LABEL:-base_b${DIAG_BATCH}_lr${DIAG_LR}}

echo "[$(date +%T)] base run $DIAG_LABEL on $(hostname)"
exec "$PYTHON" -u scripts/train/sphenix/diag_batch_size.py
