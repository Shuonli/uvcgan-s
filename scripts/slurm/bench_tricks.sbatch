#!/bin/bash
# Step time of the sPHENIX configuration with cheap speed-ups switched on
# one at a time and all together, at batch 4 and 32:
#
#   base   as configured
#   tf32   matrix products in TF32                  BENCH_TF32=1
#   bf16   forward passes under bfloat16 autocast   UVCGAN_S_AMP=bf16
#   lazy   losses read back once per epoch          UVCGAN_S_LAZY_METRICS=1
#   fused  fused Adam                               BENCH_FUSED=1
#   all    the four together
#
# Four streams, one GPU each (two batch sizes x two repeats), run the
# settings one after the other on the same GPU, the second repeat in the
# reverse order.
#
#   sbatch -w dahlia scripts/slurm/bench_tricks.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export BENCH_STEPS=${BENCH_STEPS:-100} BENCH_EPOCHS=${BENCH_EPOCHS:-3}
export OMP_NUM_THREADS=4
export TMPDIR=/tmp

CONFIGS=( base tf32 bf16 lazy fused all )

setting() {
    case $1 in
        base)  echo "" ;;
        tf32)  echo "BENCH_TF32=1" ;;
        bf16)  echo "UVCGAN_S_AMP=bf16" ;;
        lazy)  echo "UVCGAN_S_LAZY_METRICS=1" ;;
        fused) echo "BENCH_FUSED=1" ;;
        all)   echo "BENCH_TF32=1 UVCGAN_S_AMP=bf16 UVCGAN_S_LAZY_METRICS=1 BENCH_FUSED=1" ;;
    esac
}

# stream BATCH REPEAT CONFIG...
stream() {
    local batch=$1 rep=$2
    shift 2

    for config in "$@"; do
        local label=tricks_${config}_b${batch}_${rep}_${SLURM_JOB_ID}

        # shellcheck disable=SC2046
        env $(setting "$config") BENCH_BATCH="$batch" BENCH_LABEL="$label" \
            timeout --kill-after=30 900 \
            srun --exact -N1 -n1 --gres=gpu:1 --cpus-per-task=12 \
                 "$PYTHON" -u scripts/train/sphenix/bench_uvcgan-s.py \
            > "slurm_logs/${label}.log" 2>&1
        echo "[$(date +%T)] $label exit $?"

        # the weights of a benchmark are of no use, keep only the timings
        rm -f "$UVCGAN_S_OUTDIR/sphenix/bench/"*"_${label}/"*.pth
    done
}

reversed=( $(printf '%s\n' "${CONFIGS[@]}" | tac) )

stream 4  1 "${CONFIGS[@]}"  &
stream 4  2 "${reversed[@]}" &
stream 32 1 "${CONFIGS[@]}"  &
stream 32 2 "${reversed[@]}" &
wait

srun --exact -N1 -n1 --gres=gpu:1 nvidia-smi \
    --query-gpu=name --format=csv,noheader | head -1

"$PYTHON" - "$SLURM_JOB_ID" <<'EOF'
import glob, os, re, sys
import pandas as pd

job  = sys.argv[1]
root = os.path.join(os.environ['UVCGAN_S_OUTDIR'], 'sphenix', 'bench')
rows = []

for d in sorted(glob.glob(os.path.join(root, f'model_*_tricks_*_{job}'))):
    m = re.search(r'tricks_(\w+?)_b(\d+)_(\d)_' + job + '$', d)
    path = os.path.join(d, 'history.csv')
    h = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(h) < 2:
        print(f'no timing for {os.path.basename(d)}')
        continue
    h = h[h.epoch > 1]                           # the first epoch warms up
    steps = int(os.environ['BENCH_STEPS'])
    rows.append({
        'config' : m.group(1), 'batch' : int(m.group(2)),
        'rep' : int(m.group(3)),
        'ms_per_step' : 1000 * h.epoch_time.mean() / steps,
    })

t = pd.DataFrame(rows).pivot_table(
    index = 'config', columns = [ 'batch', 'rep' ], values = 'ms_per_step'
)
print(t.round(1).to_string())

mean = t.T.groupby(level = 'batch').mean().T
print('\nmean ms/step and speed-up over base:')
for b in mean.columns:
    base = mean.loc['base', b]
    for c in mean.index:
        print(f'  batch {b:3d} {c:6s} {mean.loc[c, b]:7.1f} ms'
              f'  {base / mean.loc[c, b]:5.2f}x')
EOF

echo "[$(date +%T)] all done"
