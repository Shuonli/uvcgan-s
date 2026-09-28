#!/bin/bash
# CPU side of the paired noisy-interpolant pilot (FLOW_NOTES.md, "Paired
# noisy-interpolant pilot"), once bench_post.sbatch (STEP=last SETS=val) has
# written both runs' images at their final update: the consolidated
# benchmark's jets, tables and figures, tower diagnostics and fixed event
# displays, PYTHIA val only (JEWEL stays frozen), into docs/flow/bench/noisy/.
#
#   sbatch -p a6k -w saturn -c 64 --mem=160G --time=03:00:00 \
#       -o slurm_logs/%x_%j.out -J noisy_bench scripts/flow/noisy_bench.sh
#
# STAGES (default all): any of jets report diag.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. ./scripts/flow/env.sh
export OMP_NUM_THREADS=1
STAGES=${STAGES:-jets report diag}
PROCS=${PROCS:-64}
OUT=docs/flow/bench/noisy
C=bench_paired_eta0_s0
N=bench_paired_eta0p1_s0

stems=(truth uvcgan__sig "${C}__euler4__sig" "${C}__midpoint32__sig"
       "${N}__euler4__sig" "${N}__midpoint32__sig")
MODELS="UVCGAN-S=uvcgan__sig,control (eta 0)=${C}__euler4__sig,noisy (eta 0.1)=${N}__euler4__sig,control (eta 0) [mid32]=${C}__midpoint32__sig,noisy (eta 0.1) [mid32]=${N}__midpoint32__sig"

for stage in $STAGES; do
    echo "[$(date +%T)] $stage"
    case $stage in
    jets)
        "$PYTHON" -u scripts/flow/bench_jets.py --set val --procs "$PROCS" \
            --models "$(IFS=,; echo "${stems[*]}")" 2>&1 | grep -v '^#' ;;
    report)
        "$PYTHON" -u scripts/flow/bench_report.py --sets val --models "$MODELS" \
            --procs 32 --out $OUT 2>&1 | grep -v '^#' ;;
    diag)
        "$PYTHON" -u scripts/flow/bench_diag.py --sets val \
            --display "UVCGAN-S=uvcgan__sig,control (eta 0) [mid32]=${C}__midpoint32__sig,noisy (eta 0.1) [mid32]=${N}__midpoint32__sig,control (eta 0)=${C}__euler4__sig,noisy (eta 0.1)=${N}__euler4__sig" \
            --towers "$MODELS" --figure-labels "UVCGAN-S,control (eta 0) [mid32],noisy (eta 0.1) [mid32],control (eta 0),noisy (eta 0.1)" \
            --out $OUT ;;
    esac
done
echo "[$(date +%T)] done"
