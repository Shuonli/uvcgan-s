#!/bin/bash
# Run the whole DDP scaling benchmark inside ONE 8-GPU allocation, so that
# every point shares the same node and no other job interleaves:
#
#   sbatch -w dahlia scripts/slurm/bench_pack.sbatch
#
# Each configuration is a job step on a subset of the allocated GPUs; steps
# that fit together run concurrently. Per-step logs go to slurm_logs/.
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_STEPS=${BENCH_STEPS:-200}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/tmp
export MASTER_ADDR=$(hostname)

PORT_BASE=$(( 20000 + SLURM_JOB_ID % 20000 ))

# run NGPU BATCH LABEL PORT_OFFSET [VAR=VALUE ...]
run() {
    local n=$1 batch=$2 label=$3 offset=$4
    shift 4

    label="${label}_${SLURM_JOB_ID}"
    echo "[$(date +%T)] start $label: $n GPU(s), batch $batch $*"

    env "$@" BENCH_BATCH=$batch BENCH_LABEL=$label \
        MASTER_PORT=$(( PORT_BASE + offset )) \
        srun --exact -N1 -n $n --ntasks-per-node=$n --gres=gpu:$n \
             --cpus-per-task=$SLURM_CPUS_PER_TASK --kill-on-bad-exit=1 -l \
             "$PYTHON" scripts/train/sphenix/bench_uvcgan-s.py \
        > "slurm_logs/${label}.log" 2>&1

    echo "[$(date +%T)] done  $label (exit $?)"
}

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, $BENCH_STEPS steps/epoch"
nvidia-smi -L | head -n 1

# wave 1: 1 + 2 + 1 + 4 GPUs concurrently
run 1  4 bench_g1_b4  1 &
run 2  4 bench_g2_b4  2 &
run 1 32 bench_g1_b32 3 &
run 4 32 bench_g4_b32 4 &
wait

# wave 2: plain vs bf16-compressed gradients, side by side
run 4  4 bench_g4_b4      5 &
run 4  4 bench_g4_b4_bf16 6 UVCGAN_S_DDP_COMPRESS=bf16 &
wait

# waves 3-5: whole node
run 8  4 bench_g8_b4      7
run 8  4 bench_g8_b4_bf16 8 UVCGAN_S_DDP_COMPRESS=bf16
run 8 32 bench_g8_b32     9

echo "[$(date +%T)] all done"
