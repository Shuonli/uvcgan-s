#!/bin/bash
# Submit the DDP scaling benchmark:
#   - 1 / 2 / 4 / 8 GPUs at 4 samples per GPU per step (the training batch)
#   - 1 and 8 GPUs at 32 samples per GPU per step, to separate the gain of
#     more GPUs from the gain of a bigger batch on a launch-bound model
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p slurm_logs

for n in 1 2 4 8; do
    sbatch --ntasks-per-node=$n --gres=gpu:$n -J bench_g${n}_b4 \
        scripts/slurm/bench_ddp.sbatch
done

for n in 1 8; do
    BENCH_BATCH=32 sbatch --ntasks-per-node=$n --gres=gpu:$n -J bench_g${n}_b32 \
        scripts/slurm/bench_ddp.sbatch
done
