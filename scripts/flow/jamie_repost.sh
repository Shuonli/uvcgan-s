#!/bin/bash
# Continuations whose single check at the base's NFE failed (FLOW_NOTES.md,
# "Jamie's toy exercises with OT flow matching"): the step-doubling rule again
# from twice that NFE (jamie_post.sbatch with START), then their test outputs
# at the new frozen NFE. On validation only; one GPU, sequentially.
#
#   sbatch -p a6k -w dahlia --gres=gpu:1 -c 6 --mem=64G --time=04:00:00 \
#       scripts/flow/jamie_repost.sh LABEL=START ...
#
#SBATCH -J jm_repost
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
for spec in "$@"; do
    echo "[$(date +%T)] ${spec%%=*} from ${spec#*=} NFE"
    LABEL=${spec%%=*} START=${spec#*=} bash scripts/flow/jamie_post.sbatch || exit 1
done
echo "[$(date +%T)] done"
