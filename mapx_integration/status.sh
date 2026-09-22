#!/usr/bin/env bash
# Evaluation / training history of runs:  ./status.sh run-a run-b ...
cd "${RUNS:-$HOME/marl/runs}"
for r in "$@"; do
  echo "== $r"
  sed -E 's/\x1b\[[0-9;]*m//g' "$r/train.log" | grep -a -E '^(ACTOR|EVALUATOR|TRAINER)' |
    grep -o -E '^(ACTOR|EVALUATOR|TRAINER)|Fraction imaged mean: [0-9.]+|Steps per second: [0-9.]+|Entropy: [-0-9.e]+|Value loss: [-0-9.e]+|Actor loss: [-0-9.e]+' |
    paste -sd' ' | sed 's/ ACTOR/\nACTOR/g' |
    sed -E 's/Fraction imaged mean/frac/g; s/Steps per second/sps/g' | tail -"${TAIL:-100}"
done
