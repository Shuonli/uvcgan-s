#!/bin/bash
# After the PYTHIA -> JEWEL translation runs (FLOW_NOTES.md, "PYTHIA -> JEWEL
# translation pilot"), on one A6000 (translation_eval.py):
#   MODE=check     the solver check of the final checkpoints on validation
#                  (docs/flow/translation/solver_check.csv)
#   MODE=generate  the final checkpoints' outputs on every PYTHIA test input at
#                  the frozen settings (ODE, STEPS), 8 alpha-DSBM samples of the
#                  first 1000 test inputs, latency
#
#   MODE=check sbatch -w dahlia scripts/flow/translation_post.sbatch
#   (a finer ODE check: MODE=check ODE=64:midpoint RUNS=tr_otcfm_s0 LABELS=OT-CFM
#    CHECK_FILE=solver_check_ode64.csv ...)
#   MODE=generate ODE=128:midpoint STEPS=30 sbatch -w dahlia scripts/flow/translation_post.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh

: "${MODE:?}"
RUNS=(${RUNS:-tr_otcfm_s0 tr_dsbm_e025_s0})
LABELS=${LABELS:-OT-CFM,alpha-DSBM}
echo "[$(date +%T)] $MODE on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
case $MODE in
check)
    "$PYTHON" -u scripts/flow/translation_eval.py --solver-check "${RUNS[@]}" \
        --labels "$LABELS" --n-check 2000 --boot 200 --out docs/flow/translation \
        --ode-setting "${ODE:-32:midpoint}" --check-file "${CHECK_FILE:-solver_check.csv}" || exit 1 ;;
generate)
    "$PYTHON" -u scripts/flow/translation_eval.py --generate "${RUNS[@]}" \
        --labels "$LABELS" --ode-setting "${ODE:-32:midpoint}" --steps "${STEPS:-30}" \
        --n-multi 1000 --multi 8 || exit 1 ;;
esac
echo "[$(date +%T)] done"
