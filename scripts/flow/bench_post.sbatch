#!/bin/bash
# Benchmark products of one trained run (FLOW_NOTES.md, "Consolidated
# benchmark"), after fm_run.sbatch has scored its checkpoints on val: cone
# scores of the val-selected checkpoint (EMA) on val and JEWEL with 4 Euler
# steps and the accurate solve (midpoint, 32 NFE), both readouts of a joint
# run (JOINT=1); with CONVERGENCE=1 also midpoint 64 NFE on val (is 32
# resolved?); then its benchmark images (bench_images.py) unless IMAGES=0.
# STEP (default best: the val-selected checkpoint; or last, or a step) and
# SETS (default "val jewel") select the checkpoint and the event sets.
#
#   LABEL=bench_joint_s0 JOINT=1 CONVERGENCE=1 \
#       sbatch -w dahlia --dependency=afterok:JOB scripts/flow/bench_post.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh

: "${LABEL:?}"
RUN=$UVCGAN_S_OUTDIR/sphenix/flow/$LABEL
DECODE=mixture
[ -n "${JOINT:-}" ] && DECODE=mixture,direct

echo "[$(date +%T)] benchmark products of $LABEL on $(hostname)"
STEP=${STEP:-best}
SETS=${SETS:-val jewel}
for truth in $SETS; do
    for setting in "4 euler" "32 midpoint"; do
        set -- $setting
        "$PYTHON" -u scripts/flow/fm_eval.py "$RUN" --truth $truth --steps $STEP \
            --nets ema --nfe "$1" --solver "$2" --decode $DECODE
    done
done
if [ -n "${CONVERGENCE:-}" ]; then
    "$PYTHON" -u scripts/flow/fm_eval.py "$RUN" --truth val --steps $STEP \
        --nets ema --nfe 64 --solver midpoint --decode $DECODE
fi
if [ "${IMAGES:-1}" != 0 ]; then
    "$PYTHON" -u scripts/flow/bench_images.py --runs "$LABEL" --step $STEP \
        --sets "$(echo $SETS | tr ' ' ',')"
fi
echo "[$(date +%T)] done"
