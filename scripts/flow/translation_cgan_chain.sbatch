#!/bin/bash
# Waits for a training-time milestone of the CycleGAN run (its budget.json), then
# submits the evaluation chain: the test outputs of that checkpoint and the
# validation curves (GPU), then the report, tables, figures and deck of every
# model in MODELS (CPU, after both). FLOW_NOTES.md, "CycleGAN baseline".
#
#   MILESTONE=2 LABEL=CycleGAN-2h MODELS="OT-FM=OT-CFM__midpoint128,CycleGAN 2 h=CycleGAN-2h__gen" \
#       sbatch -p a6k -w saturn -c 1 scripts/flow/translation_cgan_chain.sbatch
#
#SBATCH -J tr_cgan_chain
#SBATCH --time=30:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${MILESTONE:?}" "${LABEL:?}" "${MODELS:?}"
RUN='model_m(uvcgan-v2)_d(resnet)_g(vit-modnet)_tr_cgan_s0'
R="outdir/sphenix/flow/translation/runs/$RUN"
echo "[$(date +%T)] waiting for milestone ${MILESTONE} h of $RUN"
until [ -f "$R/budget.json" ] && grep -qE "\"milestone_h\": ${MILESTONE}(\.0)?,?$" "$R/budget.json"; do
    sleep 60
done
# the trainer saves the same epoch again right after the milestone save
sleep 150
j1=$(MODE=generate RUNS="$RUN" LABELS="$LABEL" MILESTONE="$MILESTONE" \
    sbatch -w dahlia --parsable -J "tr_gen_$LABEL" scripts/flow/translation_post.sbatch)
j2=$(MODE=curves RUNS="$RUN" sbatch -w dahlia --parsable -J tr_curves_cg \
    scripts/flow/translation_post.sbatch)
j3=$(MODELS="$MODELS" sbatch -p a6k -w saturn -c 32 --mem=64G --parsable -J "tr_deck_$LABEL" \
    --dependency=afterok:$j1:$j2 scripts/flow/translation_cgan_deck.sh)
echo "[$(date +%T)] milestone ${MILESTONE} h reached: generate $j1, curves $j2, deck $j3"
