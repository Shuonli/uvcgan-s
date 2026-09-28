#!/bin/bash
# Held-out test report of the alpha-DSBM closure test (dsbm_eval.py; c.f.
# FLOW_NOTES.md, "alpha-DSBM closure test"), after every run was selected on
# the validation pairs: the closure report (one output per jet, fixed seed),
# the N vs 2N step check, K samples per jet, and the figures (FIG_RUNS).
#
#   RUNS="dsbm_ref_e1_s0 ..." LABELS="alpha-DSBM,..." OUT=docs/flow/dsbm/closure_test \
#       FIG_RUNS="..." FIG_LABELS="..." [MODES="report steps multi figure"] \
#       sbatch -w dahlia scripts/flow/dsbm_report.sbatch
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

: "${RUNS:?}" "${LABELS:?}" "${OUT:?}" "${FIG_RUNS:?}" "${FIG_LABELS:?}"

evaluate() {
    echo "[$(date +%T)] dsbm_eval.py $*"
    # shellcheck disable=SC2086
    "$PYTHON" -u scripts/flow/dsbm_eval.py "$@" || exit 1
}

echo "[$(date +%T)] on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"
MODES=${MODES:-report steps multi figure}
for mode in $MODES; do
    case $mode in
    # shellcheck disable=SC2086
    report) evaluate $RUNS --labels "$LABELS" --out "$OUT" ;;
    # shellcheck disable=SC2086
    steps)  evaluate $RUNS --labels "$LABELS" --out "$OUT" --steps-check ;;
    # shellcheck disable=SC2086
    multi)  evaluate $RUNS --labels "$LABELS" --out "$OUT" --multi "${MULTI:-8}" ;;
    # shellcheck disable=SC2086
    figure) evaluate $FIG_RUNS --labels "$FIG_LABELS" --figure "${OUT}_jets.png" ;;
    esac
done
echo "[$(date +%T)] done"
