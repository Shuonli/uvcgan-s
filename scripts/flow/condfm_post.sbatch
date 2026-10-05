#!/bin/bash
# After a stochastic-pilot run (FLOW_NOTES.md, "Stochastic conditional FM
# pilot"), on one A6000: the solver check on validation (midpoint 128 against
# 256 NFE from the same noise, two noises at 256; translation_eval.py
# --solver-check), the pre-registered decision (128 if adequate, else 256
# against 512 once; the finer solve if that fails too, as a reported
# limitation), then the test outputs at the frozen solve: one sample per test
# input, 8 samples of the first 1000, the condition swap and a repeat.
#
#   RUN=tr_condfm_hard_s0 LABEL=CondFM-hard ARM=hard sbatch -w dahlia scripts/flow/condfm_post.sbatch
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

: "${RUN:?}" "${LABEL:?}" "${ARM:?}"
CF=$UVCGAN_S_OUTDIR/sphenix/flow/translation/condfm
OUT=docs/flow/translation/condfm
export TRANSLATION_OUTPUTS=$CF/outputs
echo "[$(date +%T)] $RUN on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"

check() {     # NFE -> solver_check_${ARM}_ode${NFE}.csv; prints adequate (True/False)
    "$PYTHON" -u scripts/flow/translation_eval.py --solver-check "$CF/runs/$RUN" \
        --labels "$LABEL" --n-check 2000 --boot 200 --out "$OUT" \
        --ode-setting "$1:midpoint" --check-file "solver_check_${ARM}_ode$1.csv" >&2 || exit 1
    "$PYTHON" - "$OUT/solver_check_${ARM}_ode$1.csv" <<'EOF'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1])
print(bool(d.loc[d['setting'].astype(str).str.contains('same noise'), 'adequate'].iloc[0]))
EOF
}

NFE=128
OK=$(check 128)
echo "[$(date +%T)] 128 vs 256 NFE adequate: $OK"
case $OK in True|False) ;; *) echo "solver check failed"; exit 1 ;; esac
if [ "$OK" != "True" ]; then
    OK=$(check 256)
    echo "[$(date +%T)] 256 vs 512 NFE adequate: $OK"
    case $OK in True|False) ;; *) echo "solver check failed"; exit 1 ;; esac
    NFE=$([ "$OK" == "True" ] && echo 256 || echo 512)
fi
"$PYTHON" - "$OUT/solver_frozen_${ARM}.json" "$NFE" "$OK" <<'EOF'
import json, sys
json.dump({ 'nfe' : int(sys.argv[2]), 'solver' : 'midpoint',
            'resolved' : sys.argv[3] == 'True' }, open(sys.argv[1], 'w'), indent = 4)
EOF
echo "[$(date +%T)] frozen: midpoint $NFE; test outputs"
"$PYTHON" -u scripts/flow/translation_eval.py --generate "$CF/runs/$RUN" --labels "$LABEL" \
    --ode-setting "$NFE:midpoint" --n-multi 1000 --multi 8 || exit 1
echo "[$(date +%T)] done"
