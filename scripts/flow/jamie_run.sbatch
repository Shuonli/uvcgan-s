#!/bin/bash
# One run of the toy study (FLOW_NOTES.md, "Jamie's toy exercises with OT flow
# matching") on one A6000: fm_train.py with the toy method arguments in "$@",
# into OUTDIR/sphenix/flow/jamie/runs/LABEL, then its validation curves (every
# checkpoint, 1000 validation inputs, 32 NFE midpoint; descriptive).
#
#   LABEL=E3-D_s0 MINUTES=90 sbatch -w dahlia -J E3-D_s0 scripts/flow/jamie_run.sbatch \
#       --method toyflow --toy-kind clean --toy-pairing unpaired \
#       --toy-source e3_vac --toy-target e3_med
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh
: "${LABEL:?}" "${MINUTES:?}"
RUN=$UVCGAN_S_OUTDIR/sphenix/flow/jamie/runs/$LABEL
echo "[$(date +%T)] $LABEL on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
"$PYTHON" -u scripts/flow/fm_train.py --label "$LABEL" --outdir "$RUN" --backbone uvcgan \
    --minutes "$MINUTES" --seed "${SEED:-0}" --ckpt-minutes "${CKPT_MINUTES:-10}" \
    --batch 256 ${MAX_STEPS:+--max-steps "$MAX_STEPS"} "$@" || exit 1
echo "[$(date +%T)] validation curves"
"$PYTHON" -u scripts/flow/jamie_eval.py "$RUN" --curves --nfe 32 --n-check 1000
echo "[$(date +%T)] done"
