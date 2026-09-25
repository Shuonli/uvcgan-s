#!/bin/bash
# Final scores of a trained flow run (after fm_run.sbatch has scored its
# checkpoints on val): its val-selected checkpoint on val and JEWEL (EMA
# network); posterior samplers (SAMPLER=1) also with 1 and 16 samples, and
# at the last checkpoint with 16; then the run's per-jet fidelity
# (jet_fidelity.py) under the label JF.
#
#   LABEL=step2_postflow_jets_s0 SAMPLER=1 JF=postflow-jets-s0 \
#       sbatch -w dahlia --dependency=afterok:JOB scripts/flow/fm_final.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh

: "${LABEL:?}"
RUN=$UVCGAN_S_OUTDIR/sphenix/flow/$LABEL

echo "[$(date +%T)] final scores of $LABEL on $(hostname)"
if [ -n "${SAMPLER:-}" ]; then
    "$PYTHON" -u scripts/flow/fm_eval.py "$RUN" --truth val --steps best \
        --nets ema --nfe 8 --samples 1,16
    "$PYTHON" -u scripts/flow/fm_eval.py "$RUN" --truth val --steps last \
        --nets ema --nfe 8 --samples 16
    "$PYTHON" -u scripts/flow/fm_eval.py "$RUN" --truth jewel --steps best \
        --nets ema --nfe 8 --samples 1,4,16
    if [ -n "${JF:-}" ]; then
        "$PYTHON" -u scripts/flow/jet_fidelity.py --models "$JF=sampler:$LABEL" \
            --samples 16
    fi
else
    "$PYTHON" -u scripts/flow/fm_eval.py "$RUN" --truth jewel --steps best \
        --nets ema
    if [ -n "${JF:-}" ]; then
        "$PYTHON" -u scripts/flow/jet_fidelity.py \
            --models "$JF=single:$LABEL:direct"
    fi
fi
echo "[$(date +%T)] done"
