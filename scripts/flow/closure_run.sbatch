#!/bin/bash
# One closure-test run on one A6000: train the jet -> jet flow
# (fm_train.py --method jetflow, resuming if the run exists) for MINUTES of
# training time, then score its checkpoints on the validation pairs and
# select one (closure_eval.py --select). c.f. FLOW_NOTES.md, "Closure test".
#
#   LABEL=closure_l2_s0 COST=l2 SEED=0 MINUTES=120 \
#       sbatch -w dahlia scripts/flow/closure_run.sbatch [fm_train.py arguments]
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh

: "${LABEL:?}" "${COST:?}" "${MINUTES:?}"
RUN=$UVCGAN_S_OUTDIR/sphenix/flow/$LABEL

echo "[$(date +%T)] $LABEL on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
"$PYTHON" -u scripts/flow/fm_train.py --method jetflow --label "$LABEL" \
    --cost "$COST" --cost-lambda "${LAMBDA:-1.0}" --minutes "$MINUTES" \
    --seed "${SEED:-0}" --ckpt-minutes 10 "$@" || exit 1

echo "[$(date +%T)] selecting on the validation pairs"
"$PYTHON" -u scripts/flow/closure_eval.py "$RUN" --select
echo "[$(date +%T)] done"
