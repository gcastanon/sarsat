#!/usr/bin/env bash
# Snapshot a run's checkpoints and stop it:  ./stop_run.sh <run-name> <snapshot-name>
runs="${RUNS:-$HOME/marl/runs}"
mkdir -p "$HOME/marl/snapshots"
cp -r "$runs/$1/checkpoints" "$HOME/marl/snapshots/$2"
for pid in $(pgrep -f train_sarsat.py); do
  if [ "$(readlink "/proc/$pid/cwd")" = "$runs/$1" ]; then
    echo "stopping $pid ($1)"
    kill "$pid"
  fi
done
