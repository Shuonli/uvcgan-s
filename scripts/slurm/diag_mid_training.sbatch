#!/bin/bash
# Repeat the batch size measurement from the middle of training instead of
# from initialization: every batch size continues the *same* checkpoint,
# with its weights, optimizers and schedulers, so the comparison is about
# the progress each batch size makes from an identical state.
#
#   BRANCH_EPOCH=350 sbatch -w dahlia scripts/slurm/diag_mid_training.sbatch
#
# BRANCH_EPOCH is an epoch of the base run (c.f. diag_base_run.sbatch);
# with its 200 updates per epoch, epoch 350 is 70k updates into training.
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:3
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
BRANCH_EPOCH=${BRANCH_EPOCH:?set BRANCH_EPOCH to an epoch of the base run}
BASE_LABEL=${BASE_LABEL:-base_b32_lr5e-5}
STEM="model_m(uvcgan-s)_d(resnet)_g(vit-modnet)"
BASE_DIR="$UVCGAN_S_OUTDIR/sphenix/base/${STEM}_${BASE_LABEL}"

# the learning rate has to match the base: resuming restores the optimizer
# state, whose parameter groups carry the rate the base run was using
LR=${LR:-5e-5}

echo "branching epoch $BRANCH_EPOCH of '$BASE_DIR'"

run() {
    local batch=$1
    local label=mid${BRANCH_EPOCH}_b${batch}_${SLURM_JOB_ID}
    local target="$UVCGAN_S_OUTDIR/sphenix/mid/${STEM}_${label}"

    if ! scripts/slurm/branch_from_checkpoint.sh \
            "$BASE_DIR" "$BRANCH_EPOCH" "$target"; then
        echo "[$(date +%T)] $label: branch failed"
        return 1
    fi

    echo "[$(date +%T)] start $label"
    DIAG_SUBDIR=mid DIAG_BATCH=$batch DIAG_LR=$LR DIAG_STEPS=200 \
    DIAG_EPOCHS=100000 DIAG_LABEL=$label \
        timeout --kill-after=60 "$RUN_SECONDS" \
        srun --exact -N1 -n1 --gres=gpu:1 --cpus-per-task=14 \
             "$PYTHON" -u scripts/train/sphenix/diag_batch_size.py \
        > "slurm_logs/${label}.log" 2>&1
    echo "[$(date +%T)] done  $label (rows: $(
        wc -l < "$target/history.csv" 2>/dev/null || echo 0))"
}

run 4   &
run 32  &
run 128 &
wait

echo "[$(date +%T)] all done"
