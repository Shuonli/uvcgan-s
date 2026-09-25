#!/bin/bash
# One-off: flatten the training h5 files into float16 .npy (make_cache.py).
#
#   sbatch scripts/flow/make_cache.sbatch
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}
export PYTHONPATH="$HOME/pyext/flow:$PWD:$PWD/scripts/flow${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=1

echo "[$(date +%T)] start on $(hostname)"
"$PYTHON" -u scripts/flow/make_cache.py --procs 24
echo "[$(date +%T)] done"
