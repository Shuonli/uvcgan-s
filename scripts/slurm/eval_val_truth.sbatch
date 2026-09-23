#!/bin/bash
# Score the checkpoints of sPHENIX models against the truth of the val
# events, c.f. eval_val_truth.py. Checkpoints scored before are skipped.
#
#   sbatch scripts/slurm/eval_val_truth.sbatch MODEL_DIR [MODEL_DIR ...]
#   EVAL_ARGS="--epochs 350 --nets ema" sbatch ... MODEL_DIR
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=8
export TMPDIR=/tmp

echo "[$(date +%T)] eval on $(hostname): $*"
# shellcheck disable=SC2086
exec "$PYTHON" -u scripts/slurm/eval_val_truth.py ${EVAL_ARGS:-} "$@"
