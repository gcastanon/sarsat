#!/usr/bin/env bash
# Copy checkpoints, logs, configs and evaluation JSONs of runs into the repo's runs/marl/
# (git-ignored), so they are reachable from Windows:  ./collect_runs.sh run-a run-b ...
here="$(cd "$(dirname "$0")" && pwd)"
dest="$here/../runs/marl"
runs="${RUNS:-$HOME/marl/runs}"
for r in "$@"; do
  mkdir -p "$dest/$r"
  cp -r "$runs/$r/checkpoints" "$runs/$r/outputs" "$dest/$r/" 2>/dev/null
  cp "$runs/$r/train.log" "$runs/$r"/*.json "$dest/$r/" 2>/dev/null
done
du -sh "$dest"
