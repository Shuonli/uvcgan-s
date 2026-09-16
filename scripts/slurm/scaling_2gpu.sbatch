#!/bin/bash
# How much does a second GPU buy? Two questions need separate measurements:
#
#   weak scaling    same batch per process, so the effective batch grows
#                   with the number of processes  (g1_b4  vs g2_b4)
#   strong scaling  same effective batch, so the work per process shrinks
#                   and the training run is unchanged (g1_b4 vs g2_b2)
#
#   sbatch -w venus scripts/slurm/scaling_2gpu.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_STEPS=${BENCH_STEPS:-100} BENCH_EPOCHS=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/tmp
# NOTE: some nodes resolve their own hostname to an address they
#       cannot reach; these jobs are single node, so use loopback
export MASTER_ADDR=127.0.0.1

# run NGPU BATCH NAME REPEAT
run() {
    local n=$1 batch=$2 name=$3 i=$4
    local label=scale_${name}_${i}_${SLURM_JOB_ID}

    BENCH_BATCH=$batch BENCH_LABEL=$label \
        MASTER_PORT=$(( 25000 + (SLURM_JOB_ID + RANDOM) % 10000 )) \
        timeout --kill-after=30 900 \
        srun --exact -N1 -n $n --ntasks-per-node=$n --gres=gpu:$n \
             --cpus-per-task=$SLURM_CPUS_PER_TASK --kill-on-bad-exit=1 -l \
             "$PYTHON" scripts/train/sphenix/bench_uvcgan-s.py \
        > "slurm_logs/${label}.log" 2>&1
    local status=$?

    local dir
    dir=$(ls -d "$UVCGAN_S_OUTDIR/sphenix/bench/"*"_${label}" 2>/dev/null | head -1)
    if [ -n "$dir" ] && [ -f "$dir/history.csv" ]; then
        echo "[$(date +%T)] $name #$i OK (exit $status)"
    else
        echo "[$(date +%T)] $name #$i FAILED (exit $status)"
    fi
}

for i in 1 2 3; do
    run 1  4 g1_b4  $i     # reference
    run 2  2 g2_b2  $i     # strong: effective batch stays 4
    run 2  4 g2_b4  $i     # weak:   effective batch becomes 8
    run 1 32 g1_b32 $i     # reference at the useful batch
    run 2 16 g2_b16 $i     # strong: effective batch stays 32
    run 2 32 g2_b32 $i     # weak:   effective batch becomes 64
done

echo "[$(date +%T)] all done"
