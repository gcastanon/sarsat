#!/usr/bin/env bash
# One-off research tool: extra seeds (1, 2) and the solo_plan policy for the three
# 200-satellite scenario finalists (ev200_tight, ev200_t90, ev200_t90_tight, defined in
# scenario_sweep.py's CONFIGS table), confirming the scenario choice made in DECISIONS.md
# issue 24 held across seeds. Reads nothing; appends JSON lines to
# runs/sweep/sweep_windows.jsonl.
#
# Run: ./scripts/sweep_finalists.sh (from the repo root). Minutes per seed on GPU
# (XLA_PYTHON_CLIENT_ALLOCATOR=platform works around the WSL2 allocation cap, issue 23).
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
for s in 1 2; do
  python scripts/scenario_sweep.py --seed "$s" --configs ev200_tight ev200_t90 ev200_t90_tight \
    --policies greedy_beam solo_plan coop_plan --out runs/sweep/sweep_windows.jsonl 2>&1 |
    grep -E 'return|Error|Traceback' | sed "s/^/seed$s /" | cut -c1-120
done
