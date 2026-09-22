#!/bin/bash
# Copy one checkpoint of a finished or running model into a fresh model
# directory, so that a new run resumes from it.
#
#   branch_from_checkpoint.sh BASE_DIR EPOCH TARGET_DIR
#
# The target keeps no config.json, so the new run writes its own and its
# configuration may differ from the base -- which is the point: the same
# state is continued at several batch sizes. Weights, optimizers and
# schedulers all come from the checkpoint, so the continuation is exact
# apart from what the new configuration changes.
set -euo pipefail

base=$1
epoch=$(printf '%04d' "$2")
target=$3

if [ -f "$target/config.json" ]; then
    echo "branch: '$target' already holds a model" >&2
    exit 1
fi

mkdir -p "$target/checkpoints"
n=0
for f in "$base/checkpoints/${epoch}_"*.pth; do
    [ -e "$f" ] || { echo "branch: no checkpoint $epoch in '$base'" >&2; exit 1; }
    cp -- "$f" "$target/checkpoints/"
    n=$((n + 1))
done

echo "branch: copied $n files of epoch $epoch into '$target'"
