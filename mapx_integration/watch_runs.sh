#!/usr/bin/env bash
# Print one line whenever a run logs a new evaluation, crashes or finishes.
#   ./watch_runs.sh run-a run-b ...
cd "${RUNS:-$HOME/marl/runs}"
declare -A seen
pattern='^EVALUATOR|Traceback|Out of memory|completed in'
while true; do
  for r in "$@"; do
    [ -f "$r/train.log" ] || continue
    clean=$(sed -E 's/\x1b\[[0-9;]*m//g' "$r/train.log" | grep -a -E "$pattern")
    n=$(printf '%s\n' "$clean" | grep -c .)
    if [ "$n" != "${seen[$r]:-0}" ]; then
      seen[$r]=$n
      updates=$(grep -a -c TRAINER "$r/train.log")
      last=$(printf '%s\n' "$clean" | tail -1 |
        grep -o -E 'Fraction imaged mean: [0-9.]+|Traceback|Out of memory|completed in.*' | paste -sd' ')
      train=$(sed -E 's/\x1b\[[0-9;]*m//g' "$r/train.log" | grep -a '^ACTOR' | tail -1 |
        grep -o -E 'Fraction imaged mean: [0-9.]+|Steps per second: [0-9.]+' | paste -sd' ')
      echo "$r eval#$n updates=$updates EVAL[$last] TRAIN[$train]"
    fi
  done
  sleep 30
done
