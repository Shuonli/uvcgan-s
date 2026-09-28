#!/bin/bash
# One alpha-DSBM closure stage on one A6000 (dsbm.py: pretrain, refine or
# continue), then the validation selection of its checkpoints
# (dsbm_eval.py --select: one SDE sample per jet, fixed seed). c.f.
# FLOW_NOTES.md, "alpha-DSBM closure test".
#
#   LABEL=dsbm_pre_e1_s0 STAGE=pretrain MINUTES=30 EPS=1.0 SEED=0 \
#       sbatch -w dahlia scripts/flow/dsbm_run.sbatch
#   LABEL=dsbm_ref_e1_s0 STAGE=refine INIT=dsbm_pre_e1_s0 MINUTES=90 EPS=1.0 \
#       SEED=0 sbatch -w dahlia --dependency=afterok:JOB scripts/flow/dsbm_run.sbatch
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

: "${LABEL:?}" "${STAGE:?}" "${MINUTES:?}" "${EPS:?}"

echo "[$(date +%T)] $LABEL ($STAGE) on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
"$PYTHON" -u scripts/flow/dsbm.py --label "$LABEL" --stage "$STAGE" \
    --minutes "$MINUTES" --eps "$EPS" --seed "${SEED:-0}" \
    ${INIT:+--init "$INIT"} "$@" || exit 1

echo "[$(date +%T)] selecting on the validation pairs"
"$PYTHON" -u scripts/flow/dsbm_eval.py "$LABEL" --select --train-n 2000 \
    --steps "${STEPS:-30}"
echo "[$(date +%T)] done"
