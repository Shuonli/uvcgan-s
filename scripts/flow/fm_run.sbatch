#!/bin/bash
# One flow-matching run on one A6000: train until MINUTES of training time
# (resuming if the run exists), then score on the 20k val events (midpoint
# 16 NFE; the regression at its one evaluation): the EMA network at every
# checkpoint, the raw one at the last. The scores are not training time.
#
#   METHOD=otcfm LABEL=pilot_otcfm_s0 MINUTES=25 \
#       sbatch -w dahlia scripts/flow/fm_run.sbatch [fm_train.py arguments]
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

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}
export PYTHONPATH="$HOME/pyext/flow:$PWD:$PWD/scripts/flow${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=4
export TMPDIR=/tmp

: "${METHOD:?}" "${LABEL:?}" "${MINUTES:?}"
RUN=$UVCGAN_S_OUTDIR/sphenix/flow/$LABEL

echo "[$(date +%T)] $LABEL on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
"$PYTHON" -u scripts/flow/fm_train.py --method "$METHOD" --label "$LABEL" \
    --minutes "$MINUTES" --seed "${SEED:-0}" "$@" || exit 1

echo "[$(date +%T)] scoring the checkpoints on val"
"$PYTHON" -u scripts/flow/fm_eval.py "$RUN" --truth val --nets ema
"$PYTHON" -u scripts/flow/fm_eval.py "$RUN" --truth val --nets raw --steps last
echo "[$(date +%T)] done"
