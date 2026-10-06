#!/bin/bash
# After a toy run (FLOW_NOTES.md, "Jamie's toy exercises with OT flow matching"),
# on one A6000: the step-doubling solver check on validation (jamie_eval.py
# --check: N against 2N from the same input and, for C, the same noise),
# N = START, 2 START, ... up to 256 until adequate (the finest checked if none
# is, a reported limitation); the frozen N in docs/flow/jamie_otfm/solver/
# LABEL_frozen.json; then the test outputs at N (jamie_eval.py --generate).
# A continuation passes FIXED=N (its base's frozen solve), checked once.
#
#   LABEL=E3-D_s0 sbatch -w dahlia scripts/flow/jamie_post.sbatch
#   LABEL=E3-D_ctl FIXED=64 sbatch -w dahlia scripts/flow/jamie_post.sbatch
#   LABEL=E3-D_ctl BASE=E3-D_s0 sbatch -w dahlia scripts/flow/jamie_post.sbatch
#
# SETS=t2_pair limits the test outputs to those sets (the 2 x 2 cells).
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh
: "${LABEL:?}"
RUN=$UVCGAN_S_OUTDIR/sphenix/flow/jamie/runs/$LABEL
OUT=docs/flow/jamie_otfm/solver
mkdir -p "$OUT"
E=scripts/flow/jamie_eval.py
echo "[$(date +%T)] $LABEL on $(hostname)"

adequate() {
    "$PYTHON" -u "$E" "$RUN" --check --nfe "$1" --n-check 1000 --out "$OUT" --label "$LABEL" >&2 || exit 1
    "$PYTHON" - "$OUT/$LABEL.csv" "$1" <<'EOF'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1])
d = d[d['nfe'] == int(sys.argv[2])]
print(bool(d['adequate'].iloc[-1]))
EOF
}

if [ -n "${BASE:-}" ]; then
    # a continuation: its base's frozen solve
    FIXED=$("$PYTHON" -c "import json; print(json.load(open('$OUT/${BASE}_frozen.json'))['nfe'])")
fi
if [ -n "${FIXED:-}" ]; then
    N=$FIXED
    OK=$(adequate "$N")
    echo "[$(date +%T)] fixed $N, $N vs $((2 * N)): $OK"
else
    N=${START:-32}
    while true; do
        OK=$(adequate "$N")
        echo "[$(date +%T)] $N vs $((2 * N)) NFE adequate: $OK"
        case $OK in True|False) ;; *) echo "solver check failed"; exit 1 ;; esac
        [ "$OK" == "True" ] && break
        if [ "$N" -ge 128 ]; then N=$((2 * N)); break; fi
        N=$((2 * N))
    done
fi
"$PYTHON" - "$OUT/${LABEL}_frozen.json" "$N" "$OK" <<'EOF'
import json, sys
json.dump({ 'nfe' : int(sys.argv[2]), 'solver' : 'midpoint',
            'resolved' : sys.argv[3] == 'True' }, open(sys.argv[1], 'w'), indent = 4)
EOF
echo "[$(date +%T)] frozen midpoint $N; test outputs"
"$PYTHON" -u "$E" "$RUN" --generate --nfe "$N" --label "$LABEL" ${SETS:+--sets "$SETS"} || exit 1
echo "[$(date +%T)] done"
