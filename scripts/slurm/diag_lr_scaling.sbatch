#!/bin/bash
# Batch size and learning rate raised together, to find how far the pair
# can be scaled before training destabilizes. Every run starts from
# initialization, gets the same wall clock and one GPU of the same node, so
# the results are comparable with each other and with diag_batch_size.
#
#   sbatch -w dahlia scripts/slurm/diag_lr_scaling.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:8
#SBATCH --time=05:30:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=4
export TMPDIR=/tmp

RUN_SECONDS=${RUN_SECONDS:-16200}

run() {
    local batch=$1 lr=$2
    local label=lrs_b${batch}_lr${lr}_${SLURM_JOB_ID}

    DIAG_SUBDIR=lrs DIAG_BATCH=$batch DIAG_LR=$lr DIAG_STEPS=200 \
    DIAG_EPOCHS=100000 DIAG_LABEL=$label \
        timeout --kill-after=60 "$RUN_SECONDS" \
        srun --exact -N1 -n1 --gres=gpu:1 --cpus-per-task=11 \
             "$PYTHON" -u scripts/train/sphenix/diag_batch_size.py \
        > "slurm_logs/${label}.log" 2>&1
    echo "[$(date +%T)] done $label"
}

# batch 32 at 1e-4 was the best of the previous sweep; probe above it, and
# ask whether batch 64 with a raised rate can beat it
run  16 1e-4   &
run  32 1.5e-4 &
run  32 2e-4   &
run  32 3e-4   &
run  64 1e-4   &
run  64 1.5e-4 &
run  64 2e-4   &
run 128 1.5e-4 &
wait

echo "[$(date +%T)] all done"
