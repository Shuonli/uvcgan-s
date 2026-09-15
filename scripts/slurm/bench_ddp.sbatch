#!/bin/bash
# Benchmark UVCGAN-S training throughput on N GPUs of one node with DDP,
# one SLURM task per GPU:
#
#   sbatch --ntasks-per-node=4 --gres=gpu:4 -J bench_g4 scripts/slurm/bench_ddp.sbatch
#
# Submit from the repository root (c.f. scripts/slurm/submit_bench.sh for
# the whole 1/2/4/8-GPU matrix). BENCH_* variables of bench_uvcgan-s.py are
# forwarded from the submitting shell.
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_LABEL=${BENCH_LABEL:-${SLURM_JOB_NAME}_${SLURM_JOB_ID}}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TMPDIR=/tmp      # keep DataLoader worker scratch off NFS

# torch.distributed rendezvous; RANK / LOCAL_RANK / WORLD_SIZE are taken
# from SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS by uvcgan_s
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: $SLURM_NTASKS task(s), label $BENCH_LABEL"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1

srun -l --kill-on-bad-exit=1 "$PYTHON" scripts/train/sphenix/bench_uvcgan-s.py
