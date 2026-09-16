#!/bin/bash
# Diagnose the 8 GPU / batch 4 deadlock and test candidate fixes, each in
# its own job step with a short NCCL timeout:
#
#   1. baseline    reproduce with the NCCL flight recorder and DDP's
#                  cross-rank consistency checks
#   2. one_bucket  a bucket cap larger than any model, so each module
#                  issues exactly one all-reduce and the bucket rebuild
#                  cannot produce a different bucketing on different ranks
#   3. find_unused find_unused_parameters=1
#   4. bf16        the configuration that was observed to work
#
#   sbatch -w dahlia scripts/slurm/diagnose_hang.sbatch
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
export BENCH_BATCH=4 BENCH_STEPS=40 BENCH_EPOCHS=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/tmp
# NOTE: some nodes resolve their own hostname to an address they
#       cannot reach; these jobs are single node, so use loopback
export MASTER_ADDR=127.0.0.1
export UVCGAN_S_DDP_TIMEOUT_MIN=3

variant() {
    local name=$1; shift
    local label=diag_${name}_${SLURM_JOB_ID}

    echo "[$(date +%T)] start $name: $*"
    env "$@" BENCH_LABEL=$label MASTER_PORT=$(( 23000 + SLURM_JOB_ID % 10000 )) \
        srun --exact -N1 -n8 --ntasks-per-node=8 --gres=gpu:8 \
             --cpus-per-task=$SLURM_CPUS_PER_TASK -l \
             "$PYTHON" scripts/train/sphenix/bench_uvcgan-s.py \
        > "slurm_logs/${label}.log" 2>&1
    local status=$?
    echo "[$(date +%T)] done  $name (exit $status)"
}

variant baseline \
    TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
    TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$PWD/slurm_logs/nccl_trace_${SLURM_JOB_ID} \
    TORCH_DISTRIBUTED_DEBUG=DETAIL
variant one_bucket  UVCGAN_S_DDP_BUCKET_MB=512
variant find_unused UVCGAN_S_DDP_FIND_UNUSED=1
variant bf16        UVCGAN_S_DDP_COMPRESS=bf16

echo "[$(date +%T)] all done"
