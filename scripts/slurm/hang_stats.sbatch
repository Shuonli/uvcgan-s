#!/bin/bash
# The 8 GPU / batch 4 deadlock is intermittent, so single runs prove
# nothing. Repeat the failing configuration and each candidate fix several
# times and count how often each deadlocks. 20 steps is enough: both
# observed hangs happened around step 4.
#
#   sbatch -w dahlia scripts/slurm/hang_stats.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_BATCH=4 BENCH_STEPS=20 BENCH_EPOCHS=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/tmp
export MASTER_ADDR=$(hostname)
export UVCGAN_S_DDP_TIMEOUT_MIN=2

REPEATS=${REPEATS:-6}

attempt() {
    local name=$1 i=$2; shift 2
    local label=stat_${name}_${i}_${SLURM_JOB_ID}

    env "$@" BENCH_LABEL=$label \
        MASTER_PORT=$(( 24000 + SLURM_JOB_ID % 10000 )) \
        srun --exact -N1 -n8 --ntasks-per-node=8 --gres=gpu:8 \
             --cpus-per-task=$SLURM_CPUS_PER_TASK -l \
             "$PYTHON" scripts/train/sphenix/bench_uvcgan-s.py \
        > "slurm_logs/${label}.log" 2>&1
    local status=$?

    # the run only writes history.csv once an epoch finishes
    local dir
    dir=$(ls -d "$UVCGAN_S_OUTDIR/sphenix/bench/"*"_${label}" 2>/dev/null | head -1)
    if [ -n "$dir" ] && [ -f "$dir/history.csv" ]; then
        echo "[$(date +%T)] $name #$i OK (exit $status)"
    else
        echo "[$(date +%T)] $name #$i HUNG (exit $status)"
    fi
}

for i in $(seq 1 $REPEATS); do
    attempt baseline    $i
    attempt one_bucket  $i UVCGAN_S_DDP_BUCKET_MB=512
    attempt static      $i UVCGAN_S_DDP_STATIC_GRAPH=1
    attempt bf16        $i UVCGAN_S_DDP_COMPRESS=bf16
done

echo "[$(date +%T)] summary"
grep -oE "(baseline|one_bucket|static|bf16) #[0-9]+ (OK|HUNG)" "slurm_logs/hang_stats_${SLURM_JOB_ID}.out" \
    | awk '{n[$1"_"$3]++} END {for (k in n) print "  ", k, n[k]}' | sort
echo "[$(date +%T)] all done"
