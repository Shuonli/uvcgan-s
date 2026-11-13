#!/bin/bash
# CycleGAN (UVCGAN2) on the toy's girth-rule translation, one A6000
# (jamie_cgan.py; FLOW_NOTES.md, "CycleGAN on the girth-rule toy"): HOURS of
# training time with checkpoints at the MILESTONES, then every milestone's
# test outputs, the Exercise 3g report with the flow models, and its figures.
#
#   LABEL=E3g-CG_s0 HOURS=8 MILESTONES=1.5,4 sbatch -w dahlia scripts/flow/jamie_cgan.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -J E3g-CG
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh
: "${LABEL:?}" "${HOURS:?}"
MS=${MILESTONES:-1.5,4}
echo "[$(date +%T)] $LABEL on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
"$PYTHON" -u scripts/flow/jamie_cgan.py --label "$LABEL" --hours "$HOURS" --milestones "$MS" \
    --seed "${SEED:-0}" || exit 1
echo "[$(date +%T)] test outputs"
"$PYTHON" -u scripts/flow/jamie_cgan.py --generate --label "$LABEL" || exit 1
M="D=E3g-D_s0,C hard=E3g-C_s0"
for h in ${MS//,/ } $HOURS; do M="$M,CycleGAN $h h=${LABEL}_${h}h"; done
echo "[$(date +%T)] report: $M"
export OMP_NUM_THREADS=1
"$PYTHON" -u scripts/flow/jamie_report.py --exercise 3g --models "$M" --procs 10 || exit 1
"$PYTHON" -u scripts/flow/jamie_figs.py --exercise 3g --models "$M" || exit 1
echo "[$(date +%T)] done"
