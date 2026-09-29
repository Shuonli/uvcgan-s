#!/bin/bash
# CPU stage of the PYTHIA -> JEWEL translation pilot: every test metric,
# table and figure (translation_report.py) and the appendix tables
# (translation_tables.py), from the generated outputs.
#
#   ODE=midpoint128 STEPS=30 sbatch -p a6k -w saturn -c 32 --mem=64G scripts/flow/translation_report.sh
#
#SBATCH -J tr_report
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh
export OMP_NUM_THREADS=1

ODE=${ODE:-midpoint128}
STEPS=${STEPS:-30}
echo "[$(date +%T)] on $(hostname)"
"$PYTHON" -u scripts/flow/translation_report.py \
    --models "OT-CFM=OT-CFM__$ODE,OT-CFM (4 Euler)=OT-CFM__euler4,alpha-DSBM=alpha-DSBM__sde$STEPS" \
    --multi "alpha-DSBM=alpha-DSBM__sde${STEPS}_multi8" \
    --curves "OT-CFM=tr_otcfm_s0,alpha-DSBM=tr_dsbm_e025_s0_pre+tr_dsbm_e025_s0" \
    --out docs/flow/translation --boot 200 --procs 30 || exit 1
"$PYTHON" -u scripts/flow/translation_tables.py --dir docs/flow/translation || exit 1
echo "[$(date +%T)] done"
