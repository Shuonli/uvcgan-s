#!/bin/bash
# One model of the PYTHIA -> JEWEL translation pilot on one A6000 (FLOW_NOTES.md,
# "PYTHIA -> JEWEL translation pilot"), then its validation curves
# (translation_eval.py --curves). Runs live in OUTDIR/sphenix/flow/translation/runs.
#
#   KIND=otcfm LABEL=tr_otcfm_s0 MINUTES=120 SEED=0 \
#       sbatch -w dahlia scripts/flow/translation_run.sbatch
#   KIND=dsbm LABEL=tr_dsbm_e025_s0 PRE_MINUTES=30 MINUTES=90 EPS=0.25 SEED=0 \
#       sbatch -w dahlia scripts/flow/translation_run.sbatch
#   KIND=condjet COUPLING=entropic OT_REG=7 LABEL=tr_condfm_soft_s0 MINUTES=120 \
#       SEED=0 sbatch -w dahlia scripts/flow/translation_run.sbatch
#
# KIND=otcfm: fm_train.py --method jetflow, the squared L2 cost of the
# standardised states, exact minibatch OT (pool = batch 256), UVCGAN-S backbone,
# MINUTES of training time. KIND=dsbm: dsbm.py pretraining (PRE_MINUTES, run
# ${LABEL}_pre) and online refinement from its last checkpoint (MINUTES,
# rollouts included), the same GPU one after the other. KIND=condjet: the
# stochastic pilot's noise-to-target conditional FM (fm_train.py --method
# condjet, pairs from the exact or entropic plan, COUPLING and OT_REG), run in
# OUTDIR/sphenix/flow/translation/condfm/runs, curves at midpoint 128.
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh

: "${KIND:?}" "${LABEL:?}" "${MINUTES:?}"
TR=$UVCGAN_S_OUTDIR/sphenix/flow/translation
DATA=(--source-domain tr_pythia --target-domain tr_jewel --norm-path "$TR/norm.json")
CURVE_ARGS=()

echo "[$(date +%T)] $LABEL ($KIND) on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
case $KIND in
otcfm)
    "$PYTHON" -u scripts/flow/fm_train.py --method jetflow --label "$LABEL" \
        --outdir "$TR/runs/$LABEL" --cost l2 --backbone uvcgan \
        --minutes "$MINUTES" --seed "${SEED:-0}" --ckpt-minutes 10 \
        "${DATA[@]}" "$@" || exit 1
    RUNS=("$LABEL")
    ;;
dsbm)
    : "${PRE_MINUTES:?}" "${EPS:?}"
    COMMON=(--eps "$EPS" --seed "${SEED:-0}" --outdir "$TR/runs" --monitor translation "${DATA[@]}")
    "$PYTHON" -u scripts/flow/dsbm.py --label "${LABEL}_pre" --stage pretrain \
        --minutes "$PRE_MINUTES" "${COMMON[@]}" "$@" || exit 1
    echo "[$(date +%T)] refinement"
    "$PYTHON" -u scripts/flow/dsbm.py --label "$LABEL" --stage refine \
        --init "${LABEL}_pre" --minutes "$MINUTES" "${COMMON[@]}" "$@" || exit 1
    RUNS=("${LABEL}_pre" "$LABEL")
    ;;
condjet)
    : "${COUPLING:?}"
    "$PYTHON" -u scripts/flow/fm_train.py --method condjet --label "$LABEL" \
        --outdir "$TR/condfm/runs/$LABEL" --cost l2 --backbone uvcgan \
        --coupling "$COUPLING" ${OT_REG:+--ot-reg "$OT_REG"} \
        --minutes "$MINUTES" --seed "${SEED:-0}" --ckpt-minutes 10 \
        "${DATA[@]}" "$@" || exit 1
    RUNS=("$TR/condfm/runs/$LABEL")
    CURVE_ARGS=(--ode-setting 128:midpoint)
    ;;
esac

echo "[$(date +%T)] validation curves"
"$PYTHON" -u scripts/flow/translation_eval.py --curves "${RUNS[@]}" ${CURVE_ARGS[@]+"${CURVE_ARGS[@]}"}
echo "[$(date +%T)] done"
