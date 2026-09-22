#!/usr/bin/env bash
# Extra seeds and solo_plan for the 200-satellite finalists (see scenario_sweep.py).
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
for s in 1 2; do
  python scripts/scenario_sweep.py --seed "$s" --configs ev200_tight ev200_t90 ev200_t90_tight \
    --policies greedy_beam solo_plan coop_plan --out runs/sweep/sweep_windows.jsonl 2>&1 |
    grep -E 'return|Error|Traceback' | sed "s/^/seed$s /" | cut -c1-120
done
