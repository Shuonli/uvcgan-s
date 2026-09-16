#!/bin/bash
# Diagnose the 8 GPU / batch 4 collective deadlock with DDP's detailed
# consistency checks and the NCCL flight recorder, which dumps the
# collectives each rank issued when the watchdog fires.
#
#   sbatch -w dahlia scripts/slurm/debug_hang.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_BATCH=4 BENCH_STEPS=20 BENCH_EPOCHS=2
export BENCH_LABEL=debug_g8_b4_${SLURM_JOB_ID}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/tmp
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 22000 + SLURM_JOB_ID % 10000 ))
export UVCGAN_S_DDP_TIMEOUT_MIN=3

# per-rank record of every collective; dumped on the watchdog timeout
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$PWD/slurm_logs/nccl_trace_${SLURM_JOB_ID}
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# cross-rank checks of model/bucket consistency and collective mismatches
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN

srun --exact -N1 -n8 --ntasks-per-node=8 --gres=gpu:8 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK -l \
     "$PYTHON" scripts/train/sphenix/bench_uvcgan-s.py
echo "exit=$?"
