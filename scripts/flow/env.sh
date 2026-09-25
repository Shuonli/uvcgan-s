# Environment of the flow-matching jobs; `source scripts/flow/env.sh` from
# the repository root, e.g. in `sbatch --wrap`.
PYTHON=${PYTHON:-/home/shuhang/miniconda3/envs/fm4npp/bin/python}
export PYTHONPATH="$HOME/pyext/flow:$PWD:$PWD/scripts/flow${PYTHONPATH:+:$PYTHONPATH}"
export UVCGAN_S_DATA=${UVCGAN_S_DATA:-$PWD/data}
export UVCGAN_S_OUTDIR=${UVCGAN_S_OUTDIR:-$PWD/outdir}
export OMP_NUM_THREADS=4
export TMPDIR=/tmp
