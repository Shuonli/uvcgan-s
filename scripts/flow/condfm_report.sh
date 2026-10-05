#!/bin/bash
# CPU stage of the stochastic pilot (FLOW_NOTES.md, "Stochastic conditional FM
# pilot"): the population report of the new runs next to the existing outputs
# (translation_report.py; existing outputs linked, not copied, into the pilot's
# own output directory), the repeated-sample analyses and the reading
# (condfm_report.py), the deck-style figures (translation_deck_figs.py) and the
# appendix. The frozen solves come from docs/flow/translation/condfm/
# solver_frozen_{hard,soft}.json (condfm_post.sbatch); CycleGAN at every
# training-time milestone whose outputs exist.
#
#   sbatch -p a6k -w saturn -c 32 --mem=64G scripts/flow/condfm_report.sh
#
#SBATCH -J cf_report
#SBATCH --time=01:30:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh
export OMP_NUM_THREADS=1

TR=$UVCGAN_S_OUTDIR/sphenix/flow/translation
CF=$TR/condfm
OUT=docs/flow/translation/condfm
export TRANSLATION_OUTPUTS=$CF/outputs
mkdir -p "$CF/outputs"
for f in OT-CFM__midpoint128 alpha-DSBM__sde30 alpha-DSBM__sde30_multi8 \
         CycleGAN-2h__gen CycleGAN-8h__gen CycleGAN-24h__gen; do
    for x in npy json; do
        [ -e "$TR/outputs/$f.$x" ] && ln -sfn "$TR/outputs/$f.$x" "$CF/outputs/$f.$x"
    done
done
nfe() { "$PYTHON" -c "import json; print(json.load(open('$OUT/solver_frozen_$1.json'))['nfe'])"; }
NH=$(nfe hard)
NS=$(nfe soft)
check() { [ "$1" == 128 ] && echo ode128 || echo ode256; }
CG=""
for h in 2 8 24; do
    [ -e "$CF/outputs/CycleGAN-${h}h__gen.npy" ] && CG="$CG,CycleGAN $h h=CycleGAN-${h}h__gen"
done
COND="CondFM hard OT=CondFM-hard__midpoint$NH,CondFM soft OT=CondFM-soft__midpoint$NS"
MODELS="OT-FM=OT-CFM__midpoint128,$COND,alpha-DSBM=alpha-DSBM__sde30$CG"
CURVES="OT-FM=tr_otcfm_s0,CondFM hard OT=$CF/runs/tr_condfm_hard_s0,CondFM soft OT=$CF/runs/tr_condfm_soft_s0"
echo "[$(date +%T)] on $(hostname): $MODELS"
"$PYTHON" -u scripts/flow/translation_report.py --models "$MODELS" --curves "$CURVES" \
    --out "$OUT" --boot 200 --procs 30 || exit 1
"$PYTHON" -u scripts/flow/condfm_report.py --models "$COND" \
    --reference "alpha-DSBM=alpha-DSBM__sde30" --det "OT-FM=OT-CFM__midpoint128" \
    --solver "CondFM hard OT=solver_check_hard_$(check $NH).csv,CondFM soft OT=solver_check_soft_$(check $NS).csv" \
    --report "$OUT" --boot 200 --procs 30 || exit 1
"$PYTHON" -u scripts/flow/translation_deck_figs.py --report condfm \
    --models "OT-FM=OT-CFM__midpoint128,$COND" --out "$OUT/deck" || exit 1
if [ -e "$OUT/slides/condfm_appendix.tex" ]; then
    (cd "$OUT/slides" && ~/pyext/tectonic_env/bin/tectonic condfm_appendix.tex) || exit 1
fi
echo "[$(date +%T)] done"
