#!/bin/bash
# Re-run the configuration that deadlocked (8 GPUs, batch 4, uncompressed
# gradients) several times to tell a reproducible deadlock from a flake.
# A short NCCL timeout turns a hang into a fast failure.
#
#   sbatch -w dahlia scripts/slurm/repro_hang.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_BATCH=4 BENCH_STEPS=50 BENCH_EPOCHS=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/tmp
export MASTER_ADDR=$(hostname)
export UVCGAN_S_DDP_TIMEOUT_MIN=${UVCGAN_S_DDP_TIMEOUT_MIN:-4}

for i in 1 2 3; do
    label=repro_g8_b4_${i}_${SLURM_JOB_ID}
    echo "[$(date +%T)] attempt $i"

    BENCH_LABEL=$label MASTER_PORT=$(( 21000 + SLURM_JOB_ID % 10000 + i )) \
        srun --exact -N1 -n8 --ntasks-per-node=8 --gres=gpu:8 \
             --cpus-per-task=$SLURM_CPUS_PER_TASK --kill-on-bad-exit=1 -l \
             "$PYTHON" scripts/train/sphenix/bench_uvcgan-s.py \
        > "slurm_logs/${label}.log" 2>&1
    echo "[$(date +%T)] attempt $i exit=$?"
done
echo "[$(date +%T)] all done"
