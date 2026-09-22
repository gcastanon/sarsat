#!/usr/bin/env bash
# Stage the latest checkpoint of <src-run> so that <dst-run> can start from it:
#   ./resume_from.sh <src-run> <dst-run> <ippo|mappo>
# then launch with: logger.checkpointing.load_model=True
#                   logger.checkpointing.load_args.checkpoint_uid=00_start
set -euo pipefail
runs="${RUNS:-$HOME/marl/runs}"
src="$runs/$1/checkpoints/rec_$3"
latest=$(ls -d "$src"/2* | sort | tail -1)
dst="$runs/$2/checkpoints/rec_$3/00_start"
mkdir -p "$(dirname "$dst")"
rm -rf "$dst"
cp -r "$latest" "$dst"
mkdir -p "$HOME/marl/snapshots"
cp -r "$latest" "$HOME/marl/snapshots/$1_$(basename "$latest")_$(ls "$latest" | grep -E '^[0-9]+$' | tail -1)"
echo "staged $latest ($(ls "$latest" | tr '\n' ' ')) -> $dst"
