#!/bin/bash
# CPU side of the consolidated subtraction benchmark (FLOW_NOTES.md,
# "Consolidated benchmark"), once bench_post.sbatch has written every run's
# images and fm_eval.py --latency has timed the arms: jets, the paper's
# figures and tables, per-tower diagnostics, true-axis cone scores and the
# cost summary, all into docs/flow/bench/.
#
#   sbatch -p a6k -w saturn -c 64 --mem=128G --time=03:00:00 \
#       -o slurm_logs/%x_%j.out -J bench_all scripts/flow/bench_all.sh
#
# STAGES (default all): any of jets report diag cone cost losst.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. ./scripts/flow/env.sh
export OMP_NUM_THREADS=1
STAGES=${STAGES:-jets report diag cone cost losst}
PROCS=${PROCS:-64}
OUT=docs/flow/bench

UNP=(bb_uvcgan_otcfm1_s0 bb_uvcgan_otcfm1_s1 bb_uvcgan_otcfm1_s2)
PAI=(bench_paired_bkg_s0 bench_paired_bkg_s1 bench_paired_bkg_s2)
JNT=(bench_joint_s0 bench_joint_s1 bench_joint_s2)
SETTINGS=(euler4 midpoint32)

# 'a+b+c' of the seeds' image stems: seeds RUN... SETTING READOUT
seeds() {
    local setting=$1 readout=$2; shift 2
    local out=() r
    for r in "$@"; do out+=("${r}__${setting}__${readout}"); done
    (IFS=+; echo "${out[*]}")
}

stems=(truth area uvcgan__sig uvcgan__sig@thr0.5)
for s in "${SETTINGS[@]}"; do
    for r in "${UNP[@]}" "${PAI[@]}"; do stems+=("${r}__${s}__sig"); done
    for r in "${JNT[@]}"; do stems+=("${r}__${s}__sig" "${r}__${s}__sigdirect"); done
done
for r in "${UNP[0]}" "${PAI[0]}"; do stems+=("${r}__euler4__sig@thr0.5"); done
stems+=("${JNT[0]}__euler4__sig@thr0.5" "${JNT[0]}__euler4__sigdirect@thr0.5")

arms() {   # LABEL=SEEDS list of the report, for one setting and suffix
    local s=$1 sfx=$2
    echo "unpaired FM${sfx}=$(seeds "$s" sig "${UNP[@]}"),paired FM${sfx}=$(seeds "$s" sig "${PAI[@]}"),joint FM: M - B${sfx}=$(seeds "$s" sig "${JNT[@]}"),joint FM: direct S${sfx}=$(seeds "$s" sigdirect "${JNT[@]}")"
}
MAIN="UVCGAN-S=uvcgan__sig,Area=area,$(arms euler4 ''),$(arms midpoint32 ' [mid32]')"
CLEAN="UVCGAN-S [thr0.5]=uvcgan__sig@thr0.5,unpaired FM [thr0.5]=${UNP[0]}__euler4__sig@thr0.5,paired FM [thr0.5]=${PAI[0]}__euler4__sig@thr0.5,joint FM: M - B [thr0.5]=${JNT[0]}__euler4__sig@thr0.5,joint FM: direct S [thr0.5]=${JNT[0]}__euler4__sigdirect@thr0.5"
# main figures: every readout at its val-preferred solve (val cone resolution
# at the selected checkpoints: 4 Euler steps, except joint direct S: 32 NFE)
FIG="UVCGAN-S,Area,unpaired FM,paired FM,joint FM: M - B,joint FM: direct S [mid32]"
FIG_MID="UVCGAN-S,unpaired FM [mid32],paired FM [mid32],joint FM: M - B [mid32],joint FM: direct S [mid32]"

for stage in $STAGES; do
    echo "[$(date +%T)] $stage"
    case $stage in
    jets)
        for set in val jewel; do
            "$PYTHON" -u scripts/flow/bench_jets.py --set $set --procs "$PROCS" \
                --models "$(IFS=,; echo "${stems[*]}")" 2>&1 | grep -v '^#'
        done ;;
    report)
        "$PYTHON" -u scripts/flow/bench_report.py --models "$MAIN,$CLEAN" \
            --figure-models "$FIG" --procs 32 --out $OUT 2>&1 | grep -v '^#'
        # the accurate solve in the same figures (tables as above)
        "$PYTHON" -u scripts/flow/bench_report.py --models "$MAIN" \
            --figure-models "$FIG_MID" --procs 32 --out $OUT/mid32 2>&1 | grep -v '^#'
        rm -f $OUT/mid32/bench_jets.csv $OUT/mid32/bench_summary.csv ;;
    diag)
        "$PYTHON" -u scripts/flow/bench_diag.py \
            --display "UVCGAN-S=uvcgan__sig,unpaired FM=${UNP[0]}__euler4__sig,paired FM=${PAI[0]}__euler4__sig,joint FM: M - B=${JNT[0]}__euler4__sig,joint FM: direct S [mid32]=${JNT[0]}__midpoint32__sigdirect" \
            --figure-labels "$FIG" \
            --towers "UVCGAN-S=uvcgan__sig,$(arms euler4 ''),$(arms midpoint32 ' [mid32]')" \
            --consistency "joint FM=$(IFS=+; echo "${JNT[*]/%/__euler4}"),joint FM [mid32]=$(IFS=+; echo "${JNT[*]/%/__midpoint32}")" \
            --out $OUT ;;
    cone)
        specs=("UVCGAN-S=images:uvcgan__sig")
        for s in "${SETTINGS[@]}"; do
            for k in 0 1 2; do
                specs+=("unpaired-${s}-s$k=images:${UNP[$k]}__${s}__sig"
                        "paired-${s}-s$k=images:${PAI[$k]}__${s}__sig"
                        "jointMB-${s}-s$k=images:${JNT[$k]}__${s}__sig"
                        "jointS-${s}-s$k=images:${JNT[$k]}__${s}__sigdirect")
            done
        done
        # the first one alone: it also writes the truth files
        printf '%s\n' "${specs[@]}" | head -n 1 | xargs -I{} "$PYTHON" -u \
            scripts/flow/jet_fidelity.py --device cpu --models {} \
            --dir "$UVCGAN_S_OUTDIR/sphenix/flow/bench/jet_fidelity" 2>&1 \
            | grep -v 'UserWarning\|from_numpy'
        printf '%s\n' "${specs[@]}" | tail -n +2 | xargs -P 25 -I{} "$PYTHON" -u \
            scripts/flow/jet_fidelity.py --device cpu --models {} \
            --dir "$UVCGAN_S_OUTDIR/sphenix/flow/bench/jet_fidelity" 2>&1 \
            | grep -v 'UserWarning\|from_numpy'
        "$PYTHON" -u scripts/flow/jet_fidelity.py --report \
            --dir "$UVCGAN_S_OUTDIR/sphenix/flow/bench/jet_fidelity" \
            --out $OUT/bench_cone
        # the figure: seed 0 of every arm, both solvers
        "$PYTHON" -u scripts/flow/jet_fidelity.py --report \
            --dir "$UVCGAN_S_OUTDIR/sphenix/flow/bench/jet_fidelity" \
            --show "UVCGAN-S,$(for a in unpaired paired jointMB jointS; do printf '%s-euler4-s0,%s-midpoint32-s0,' $a $a; done | sed 's/,$//')" \
            --out $OUT/bench_cone_s0 --figure $OUT/bench_cone_s0.png > /dev/null ;;
    cost)
        B="$UVCGAN_S_OUTDIR/sphenix/base/model_m(uvcgan-s)_d(resnet)_g(vit-modnet)"
        "$PYTHON" -u scripts/flow/bench_cost.py \
            --arms "unpaired FM=$(IFS=+; echo "${UNP[*]}"),paired FM=$(IFS=+; echo "${PAI[*]}"),joint FM: M - B=$(IFS=+; echo "${JNT[*]}")" \
            --baseline "${B}_base_b4_lr5e-5_s0" "${B}_base_b4_lr5e-5_s1" \
                "${B}_base_b4_lr5e-5_s2" "${B}_base_b32_lr5e-5" \
                "${B}_base_b32_lr5e-5_s1" "${B}_base_b32_lr5e-5_s2" \
            --reference "$UVCGAN_S_OUTDIR/sphenix/pretrained/model_m(uvcgan-s)_d(resnet)_g(vit-modnet)_sgn_bkg_sub" \
            --out $OUT ;;
    losst)
        OMP_NUM_THREADS=16 "$PYTHON" -u scripts/flow/bench_loss_t.py --device cpu \
            --runs "${PAI[0]}" "${JNT[0]}" --n-events 1000 --out $OUT/bench_loss_t.csv \
            2>&1 | grep -v 'Warning\|meshgrid\|from_numpy' ;;
    esac
done
echo "[$(date +%T)] done"
