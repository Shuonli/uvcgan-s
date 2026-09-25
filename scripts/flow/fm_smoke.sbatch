#!/bin/bash
# Checks before the flow-matching pilot, on one A6000:
#   1. smoke_checks.py: shapes, finite gradients, couplings, solver step
#      counts, checkpoint round trip, step and coupling cost
#   2. every method trained 40 steps, resumed to 80, scored on 1k val events
#      (the full code path: loader, checkpoint, resume, evaluator)
#   3. coupling_diag.py: does the minibatch plan pair mixtures with their
#      own signals (scored with held-out truth)
#
#   sbatch -w dahlia scripts/flow/fm_smoke.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}
export PYTHONPATH="$HOME/pyext/flow:$PWD:$PWD/scripts/flow${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=4
export TMPDIR=/tmp

SMOKE=$UVCGAN_S_OUTDIR/sphenix/flow/smoke_$SLURM_JOB_ID

echo "[$(date +%T)] checks"
"$PYTHON" -u scripts/flow/smoke_checks.py || exit 1

for method in otcfm sbcfm condcfm regress; do
    echo "[$(date +%T)] $method: train 40 steps, resume to 80, score"
    for steps in 40 80; do
        "$PYTHON" -u scripts/flow/fm_train.py --method $method \
            --label smoke_$method --outdir "$SMOKE/$method" \
            --max-steps $steps --minutes 60 --ckpt-minutes 60 \
            --log-steps 10 --inline-events 500 --inline-nfe 4 || exit 1
    done
    "$PYTHON" -u scripts/flow/fm_eval.py "$SMOKE/$method" --n-events 1000 \
        --nfe 2,4 --solver euler,midpoint --decode direct,mixture \
        --samples 1,2 || exit 1
done

echo "[$(date +%T)] coupling diagnostic"
"$PYTHON" -u scripts/flow/coupling_diag.py --batches 256,1024,2048 --repeats 4

echo "[$(date +%T)] all done"
