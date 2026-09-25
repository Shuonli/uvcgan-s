#!/bin/bash
# fm_eval.py on one A6000, e.g. JEWEL scores, solver sweeps, latency:
#
#   sbatch -w saturn scripts/flow/fm_eval.sbatch RUN_DIR... --truth jewel --steps best
#
#SBATCH -p a6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -o slurm_logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}
export PYTHONPATH="$HOME/pyext/flow:$PWD:$PWD/scripts/flow${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=4

echo "[$(date +%T)] fm_eval.py $* on $(hostname)"
"$PYTHON" -u scripts/flow/fm_eval.py "$@"
echo "[$(date +%T)] done"
