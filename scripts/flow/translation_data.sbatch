#!/bin/bash
# Build the PYTHIA -> JEWEL translation data set (translation_data.py): the
# JEWEL ROOT images read once in parallel, one pass over every parent of both
# domains on one GPU. c.f. FLOW_NOTES.md, "PYTHIA -> JEWEL translation pilot".
#
#   sbatch -w saturn scripts/flow/translation_data.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh
export OMP_NUM_THREADS=1

echo "[$(date +%T)] on $(hostname)"
"$PYTHON" -u scripts/flow/translation_data.py --procs 30 "$@" || exit 1
echo "[$(date +%T)] done"
