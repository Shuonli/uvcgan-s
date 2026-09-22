#!/bin/bash
# Train the sPHENIX configuration at several batch sizes and learning rates
# at once, one GPU each on a single node so that the GPUs are identical,
# and let every run have the same wall clock. The resulting history.csv
# files show the quality metrics against both the number of updates and
# the number of samples seen.
#
#   sbatch -w dahlia scripts/slurm/diag_batch_size.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:6
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

RUN_SECONDS=${RUN_SECONDS:-16200}      # 4.5 h, inside the 5.5 h limit
DIAG_STEPS=${DIAG_STEPS:-200}

# run BATCH LR
run() {
    local batch=$1 lr=$2
    local label=diag_b${batch}_lr${lr}_${SLURM_JOB_ID}

    echo "[$(date +%T)] start $label"
    DIAG_BATCH=$batch DIAG_LR=$lr DIAG_STEPS=$DIAG_STEPS DIAG_LABEL=$label \
        timeout --kill-after=60 "$RUN_SECONDS" \
        srun --exact -N1 -n1 --gres=gpu:1 --cpus-per-task=14 \
             "$PYTHON" -u scripts/train/sphenix/diag_batch_size.py \
        > "slurm_logs/${label}.log" 2>&1
    echo "[$(date +%T)] done  $label (rows: $(
        wc -l < "$UVCGAN_S_OUTDIR/sphenix/diag/"*"_${label}/history.csv" 2>/dev/null || echo 0))"
}

# batch 4 at its own rate; the larger batches also at raised rates, so that
# a slow large batch cannot be blamed on a rate tuned for a small one
run 4   5e-5 &
run 32  5e-5 &
run 32  1e-4 &
run 128 5e-5 &
run 128 1e-4 &
run 128 2e-4 &
wait

echo "[$(date +%T)] all done"
