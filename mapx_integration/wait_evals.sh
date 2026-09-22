#!/usr/bin/env bash
# Block until <run> has logged <count> evaluations, any listed run crashed, or a run
# finished; then print the evaluation history of every listed run.
#   ./wait_evals.sh <count> <run> [other runs...]
cd "${RUNS:-$HOME/marl/runs}"
count="$1"; shift
while true; do
  n=$(grep -a -c EVALUATOR "$1/train.log" 2>/dev/null)
  [ "${n:-0}" -ge "$count" ] && break
  for r in "$@"; do
    grep -a -q -E 'Traceback|completed in' "$r/train.log" 2>/dev/null && break 2
  done
  sleep 60
done
for r in "$@"; do
  echo "== $r $(sed -E 's/\x1b\[[0-9;]*m//g' "$r/train.log" | grep -a '^EVALUATOR' |
    grep -o -E 'Fraction imaged mean: [0-9.]+' | sed 's/.*: //' | paste -sd' ')"
  echo "   train: $(sed -E 's/\x1b\[[0-9;]*m//g' "$r/train.log" | grep -a '^ACTOR' | tail -1 |
    grep -o -E 'Fraction imaged mean: [0-9.]+|Steps per second: [0-9.]+' | paste -sd' ')"
  grep -a -E 'Traceback|Out of memory|completed in' "$r/train.log" | cut -c1-200 | tail -2
done
date +%H:%M
