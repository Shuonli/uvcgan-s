#!/bin/bash
# Throughput and memory of a single GPU against the batch size.
#
# The step of this model is dominated by kernel launches rather than by
# arithmetic, so a larger batch costs much less time than it adds work.
# This sweep shows where that stops being true.
#
#   sbatch scripts/slurm/batch_sweep.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_STEPS=${BENCH_STEPS:-50} BENCH_EPOCHS=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/tmp

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for batch in 1 2 4 8 16 32 64 128 256; do
    for i in 1 2; do
        label=sweep_b${batch}_${i}_${SLURM_JOB_ID}

        # sample the GPU memory while the run is going
        ( sleep 45
          nvidia-smi --query-gpu=memory.used --format=csv,noheader \
              > "slurm_logs/${label}.mem" 2>/dev/null ) &
        mon=$!

        BENCH_BATCH=$batch BENCH_LABEL=$label \
            timeout --kill-after=30 900 \
            "$PYTHON" scripts/train/sphenix/bench_uvcgan-s.py \
            > "slurm_logs/${label}.log" 2>&1
        status=$?
        wait $mon 2>/dev/null

        dir=$(ls -d "$UVCGAN_S_OUTDIR/sphenix/bench/"*"_${label}" 2>/dev/null | head -1)
        if [ -n "$dir" ] && [ -f "$dir/history.csv" ]; then
            echo "[$(date +%T)] batch $batch #$i OK ($(cat "slurm_logs/${label}.mem" 2>/dev/null))"
        else
            # out of memory is the expected way for this loop to end
            oom=$(grep -c "CUDA out of memory" "slurm_logs/${label}.log" 2>/dev/null)
            echo "[$(date +%T)] batch $batch #$i FAILED (exit $status, oom=$oom)"
            [ "$oom" != "0" ] && break 2
        fi
    done
done

echo "[$(date +%T)] all done"
