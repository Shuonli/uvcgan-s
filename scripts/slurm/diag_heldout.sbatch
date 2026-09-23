#!/bin/bash
# Check the recommended settings against held-out truth rather than
# training losses: the configured setting (batch 4 at 5e-5) and the
# recommended ones (batch 32 at 1e-4, with and without a cached gradient
# penalty) are trained from initialization, two seeds each, one GPU per run
# on a single node and for the same wall clock. At the end of every epoch
# both generators are scored on val events against their truth
# (MODEL/val_truth_history.csv, c.f. eval_val_truth.py).
#
#   sbatch -w dahlia scripts/slurm/diag_heldout.sbatch
#   python scripts/slurm/plot_heldout.py --job <jobid>
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:6
#SBATCH --time=07:45:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=4
export TMPDIR=/tmp

RUN_SECONDS=${RUN_SECONDS:-25200}      # 7 h, inside the time limit
EVAL_EVENTS=${EVAL_EVENTS:-5000}       # ~1.3 s per generator per epoch

# run BATCH LR GP_CACHE SEED
run() {
    local batch=$1 lr=$2 gpc=$3 seed=$4
    local label=ho_b${batch}_lr${lr}_gpc${gpc}_s${seed}_${SLURM_JOB_ID}

    echo "[$(date +%T)] start $label"
    DIAG_SUBDIR=heldout DIAG_BATCH=$batch DIAG_LR=$lr DIAG_GP_CACHE=$gpc \
    DIAG_SEED=$seed DIAG_STEPS=200 DIAG_EPOCHS=100000 DIAG_CHECKPOINT=50 \
    DIAG_EVAL_EVENTS=$EVAL_EVENTS DIAG_LABEL=$label \
        timeout --kill-after=60 "$RUN_SECONDS" \
        srun --exact -N1 -n1 --gres=gpu:1 --cpus-per-task=14 \
             "$PYTHON" -u scripts/train/sphenix/diag_batch_size.py \
        > "slurm_logs/${label}.log" 2>&1
    echo "[$(date +%T)] done  $label"
}

for seed in 0 1; do
    run 4  5e-5 0 "$seed" &
    run 32 1e-4 0 "$seed" &
    run 32 1e-4 4 "$seed" &
done
wait

echo "[$(date +%T)] all done"
