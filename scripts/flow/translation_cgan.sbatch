#!/bin/bash
# CycleGAN (UVCGAN2) baseline of the PYTHIA -> JEWEL translation pilot on one
# A6000 (translation_cgan.py; FLOW_NOTES.md, "CycleGAN baseline"): HOURS of
# training time, checkpoints at the MILESTONES (hours) and every 10 epochs.
#
#   LABEL=tr_cgan_s0 HOURS=24 MILESTONES=2,8 sbatch -w dahlia scripts/flow/translation_cgan.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=28:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh

: "${LABEL:?}" "${HOURS:?}"
echo "[$(date +%T)] $LABEL on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
"$PYTHON" -u scripts/flow/translation_cgan.py --label "$LABEL" --hours "$HOURS" \
    --milestones "${MILESTONES:-2,8}" --seed "${SEED:-0}" --checkpoint 10 "$@" || exit 1
echo "[$(date +%T)] done"
