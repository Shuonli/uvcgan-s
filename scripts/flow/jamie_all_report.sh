#!/bin/bash
# The toy study's CPU stage (FLOW_NOTES.md, "Jamie's toy exercises with OT flow
# matching"): every exercise's tables (jamie_report.py) and figures
# (jamie_figs.py) from the frozen-solver outputs, for the models whose outputs
# exist, then the run/cost ledger (jamie_tables.py). EXERCISES selects (default
# all).
#
#   sbatch -p a6k -w saturn -c 32 --mem=160G scripts/flow/jamie_all_report.sh
#
#SBATCH -J jm_report
#SBATCH --time=04:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. ./scripts/flow/env.sh
export OMP_NUM_THREADS=1
O=$UVCGAN_S_OUTDIR/sphenix/flow/jamie/outputs
have() {   # LABEL=DIR,... -> only those with outputs (labels may hold spaces)
    local out=() s
    local IFS=,
    for s in $1; do [ -d "$O/${s#*=}" ] && out+=("$s"); done
    echo "${out[*]}"
}
EX1=$(have "E1-U=E1-U_s0,E1-P=E1-P_s0,E1-P+ctl=E1-P_ctl,E1-P+abs+bal=E1-P_absbal,E1-P+abs+bal+ring=E1-P_absbalring,E1-P+flat=E1-P_flat")
PD=$(have "E1-P=E1-P_s0,vac prior: syn=PD_vac_syn,vac prior x vac data=PD_vac_vac,vac prior x quench data=PD_vac_q,broad prior: syn=PD_broad_syn,broad prior x vac data=PD_broad_vac,broad prior x quench data=PD_broad_q,vac x quench + UE profile=PD_vac_q_ueprof")
EX3=$(have "D=E3-D_s0,D+ctl=E3-D_ctl,D+energy=E3-D_energy,D seed 1=E3-D_s1,C hard=E3-Ch_s0,C soft=E3-Cs_s0,C paired control=E3-Cp_s0,C hard+ctl=E3-Ch_ctl,C hard+profile=E3-Ch_profile,C hard+div=E3-Ch_div")
EX3G=$(have "D=E3g-D_s0,C hard=E3g-C_s0")
EX4=$(have "D=E4-D_s0,D+ctl=E4-D_ctl,D+global=E4-D_global,D+far=E4-D_far,D nearfar=E4-Dnf_s0,D nearfar+far=E4-Dnf_far,D seed 1=E4-D_s1,D seed 1+far=E4-D_s1_far,C=E4-C_s0,C+far=E4-C_far,C nearfar=E4-Cnf_s0,C nearfar+far=E4-Cnf_far")
# the figures show fewer models than the tables
FIG1=$(have "E1-U=E1-U_s0,E1-P=E1-P_s0,E1-P+abs+bal+ring=E1-P_absbalring")
FIG3=$(have "D=E3-D_s0,D+energy=E3-D_energy,C hard=E3-Ch_s0,C hard+profile=E3-Ch_profile,C paired control=E3-Cp_s0")
FIG4=$(have "D=E4-D_s0,D+far=E4-D_far,D nearfar=E4-Dnf_s0,C=E4-C_s0,C nearfar=E4-Cnf_s0")
R="$PYTHON -u scripts/flow/jamie_report.py"
F="$PYTHON -u scripts/flow/jamie_figs.py"
for ex in ${EXERCISES:-1 2 22 3 3g 4}; do
    case $ex in 1) M=$EX1; FM=$FIG1;; 2) M=$EX1; FM=$EX1;; 22) M=$PD; FM=$PD;;
                3) M=$EX3; FM=$FIG3;; 3g) M=$EX3G; FM=$EX3G;; 4) M=$EX4; FM=$FIG4;; esac
    [ -z "$M" ] && { echo "exercise $ex: no outputs yet"; continue; }
    echo "[$(date +%T)] exercise $ex: $M"
    $R --exercise $ex --models "$M" --procs 30 || exit 1
    $F --exercise $ex --models "${FM:-$M}" || exit 1
done
"$PYTHON" -u scripts/flow/jamie_tables.py || exit 1
echo "[$(date +%T)] done"
