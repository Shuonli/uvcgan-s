#!/bin/bash
# A continuation of the toy study (FLOW_NOTES.md, "Jamie's toy exercises with OT
# flow matching") on one A6000: the extra terms' weights and rollout accuracy
# on training data at the base checkpoint (jamie_weights.py; frozen in
# docs/flow/jamie_otfm/weights/LABEL.json), then MAX_STEPS updates from the
# base's resume.pt (fm_train.py --init), then the validation curves. The toy
# arguments of the continuation follow in "$@" (without --toy-extra/--toy-lambda).
#
#   LABEL=E3-D_energy BASE=E3-D_s0 TERMS=energy sbatch -w dahlia scripts/flow/jamie_cont.sbatch \
#       --method toyflow --toy-kind clean --toy-pairing unpaired --toy-source e3_vac --toy-target e3_med
#
# ROLL_AT_BASE=1: the solved-endpoint rollout at the base's frozen test NFE.
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
: "${LABEL:?}" "${BASE:?}"
J=$UVCGAN_S_OUTDIR/sphenix/flow/jamie/runs
W=docs/flow/jamie_otfm/weights
STEPS=${MAX_STEPS:-4000}
echo "[$(date +%T)] $LABEL from $BASE on $(hostname)"
EXTRA=()
if [ -n "${TERMS:-}" ]; then
    if [ ! -e "$W/$LABEL.json" ]; then
        "$PYTHON" -u scripts/flow/jamie_weights.py "$J/$BASE" --label "$LABEL" --terms "$TERMS" \
            -- "$@" || exit 1
    fi
    LAM=$("$PYTHON" -c "import json; d = json.load(open('$W/$LABEL.json')); print(','.join(str(d['lambda'][t]) for t in d['terms']))")
    NFE=$("$PYTHON" -c "import json; print(json.load(open('$W/$LABEL.json')).get('roll_nfe', 16))")
    if [ -n "${ROLL_AT_BASE:-}" ]; then
        # the rollout at the base's frozen test solve, not the audit's cheaper K
        # (a coarse rollout's constraint need not hold at the accurate solve)
        NFE=$("$PYTHON" -c "import json; print(json.load(open('docs/flow/jamie_otfm/solver/${BASE}_frozen.json'))['nfe'])")
    fi
    EXTRA=(--toy-extra "$TERMS" --toy-lambda "$LAM" --roll-nfe "$NFE")
    echo "[$(date +%T)] terms $TERMS, lambda $LAM, rollout NFE $NFE"
fi
"$PYTHON" -u scripts/flow/fm_train.py --label "$LABEL" --outdir "$J/$LABEL" --backbone uvcgan \
    --init "$J/$BASE" --minutes 600 --max-steps "$STEPS" --seed "${SEED:-0}" \
    --ckpt-steps "${CKPT_STEPS:-1000}" --batch 256 "$@" ${EXTRA[@]+"${EXTRA[@]}"} || exit 1
echo "[$(date +%T)] validation curves"
"$PYTHON" -u scripts/flow/jamie_eval.py "$J/$LABEL" --curves --nfe 32 --n-check 1000
echo "[$(date +%T)] done"
