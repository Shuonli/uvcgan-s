#!/bin/bash
# A translation deck's CPU stage (FLOW_NOTES.md, "PYTHIA -> JEWEL translation
# pilot", "CycleGAN baseline"): the report of every model in MODELS
# (translation_report.py, docs/flow/translation/TAG), the deck tables
# (slides/tables/TAG_*.tex), the deck figures (FIGOUT) and the deck (DECKTEX).
# CycleGAN outputs and curves come first from translation_post.sbatch.
#
#   # OT-FM against CycleGAN (the defaults)
#   MODELS="OT-FM=OT-CFM__midpoint128,CycleGAN 2 h=CycleGAN-2h__gen" \
#       sbatch -p a6k -w saturn -c 32 --mem=64G scripts/flow/translation_cgan_deck.sh
#   # OT-FM alone
#   MODELS="OT-FM=OT-CFM__midpoint128" TAG=otfm CURVES="OT-FM=tr_otcfm_s0" \
#       FIGOUT=docs/flow/translation/deck_otfm DECKTEX=translation_otfm.tex \
#       sbatch -p a6k -w saturn -c 32 --mem=64G scripts/flow/translation_cgan_deck.sh
#
#SBATCH -J tr_deck
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh
export OMP_NUM_THREADS=1

: "${MODELS:?}"
RUN='model_m(uvcgan-v2)_d(resnet)_g(vit-modnet)_tr_cgan_s0'
TAG=${TAG:-cgan}
CURVES=${CURVES:-"OT-FM=tr_otcfm_s0,CycleGAN=$RUN"}
FIGOUT=${FIGOUT:-docs/flow/translation/deck}
DECKTEX=${DECKTEX:-translation_deck.tex}
echo "[$(date +%T)] on $(hostname): $TAG: $MODELS"
MODELS="$MODELS" MULTI= CURVES="$CURVES" OUT="docs/flow/translation/$TAG" DECK="$TAG" \
    bash scripts/flow/translation_report.sh || exit 1
"$PYTHON" -u scripts/flow/translation_deck_figs.py --report "$TAG" --models "$MODELS" \
    --out "$FIGOUT" || exit 1
cd docs/flow/translation/slides && ~/pyext/tectonic_env/bin/tectonic "$DECKTEX" || exit 1
echo "[$(date +%T)] done"
