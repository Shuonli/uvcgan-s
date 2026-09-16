#!/bin/bash
# Throughput of a fixed set of configurations on whatever allocation this
# job receives, to compare two GPUs on one node with two GPUs on two nodes:
#
#   sbatch -N1 --ntasks-per-node=2 --gres=gpu:2 -w curvelet       -J topo_1node  scripts/slurm/bench_topology.sbatch
#   sbatch -N2 --ntasks-per-node=1 --gres=gpu:1 -w curvelet,venus -J topo_2nodes scripts/slurm/bench_topology.sbatch
#
# Across nodes the rendezvous needs an address that every node can reach.
# Hostnames are not reliable here (a node may resolve its own name to
# 127.0.1.1), so the first node's IP on MASTER_SUBNET is used instead.
#
#SBATCH -p a6k
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_STEPS=${BENCH_STEPS:-30} BENCH_EPOCHS=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/tmp

MASTER_SUBNET=${MASTER_SUBNET:-10.20.}
first=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

if [ "$SLURM_JOB_NUM_NODES" -eq 1 ]; then
    export MASTER_ADDR=127.0.0.1
else
    export MASTER_ADDR=$(
        srun --overlap -N1 -n1 -w "$first" hostname -I \
            | tr ' ' '\n' | grep "^${MASTER_SUBNET//./\\.}" | head -n 1
    )
fi

# no InfiniBand on this cluster; each task binds NCCL to its node's
# interface on MASTER_SUBNET, c.f. with_subnet_iface.sh
export NCCL_IB_DISABLE=1
export MASTER_SUBNET
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET

echo "nodes: $SLURM_JOB_NODELIST  tasks: $SLURM_NTASKS  master: $MASTER_ADDR"

# run NAME [VAR=VALUE ...]
run() {
    local name=$1; shift
    local i
    for i in 1 2; do
        local label=topo_${SLURM_JOB_NAME#topo_}_${name}_${i}_${SLURM_JOB_ID}

        env "$@" BENCH_LABEL=$label \
            MASTER_PORT=$(( 26000 + (SLURM_JOB_ID + RANDOM) % 10000 )) \
            timeout --kill-after=30 ${ATTEMPT_TIMEOUT:-1500} \
            srun --kill-on-bad-exit=1 -l \
                 scripts/slurm/with_subnet_iface.sh \
                 "$PYTHON" -u scripts/train/sphenix/bench_uvcgan-s.py \
            > "slurm_logs/${label}.log" 2>&1
        local status=$?

        local dir
        dir=$(ls -d "$UVCGAN_S_OUTDIR/sphenix/bench/"*"_${label}" 2>/dev/null | head -n 1)
        if [ -n "$dir" ] && [ -f "$dir/history.csv" ]; then
            echo "[$(date +%T)] $name #$i OK (exit $status)"
        else
            echo "[$(date +%T)] $name #$i FAILED (exit $status)"
        fi
    done
}

run b4        BENCH_BATCH=4
run b32       BENCH_BATCH=32
run b32_bf16  BENCH_BATCH=32 UVCGAN_S_DDP_COMPRESS=bf16

echo "[$(date +%T)] all done"
