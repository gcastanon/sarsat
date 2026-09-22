"""Summarise reference-policy baseline runs into a markdown table.

Reads ``runs/baselines/baselines.jsonl`` (the original hotspots100 greedy_beam / solo_plan
/ coop_plan rows, no ``scenario`` field -- treated as ``hotspots100``),
``runs/baselines/hotspots100_extra.jsonl`` (random / greedy / coop_dedup for the same
scenario and seeds) and ``runs/baselines/events200.jsonl`` (all six policies), and writes
``runs/baselines/summary_all.md`` with, per scenario and policy, the mean/std/min/max
``return`` over the 16 seeds.

Run: ``python scripts/summarize_baselines.py``
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("runs/baselines")
SOURCES = [
    (BASE / "baselines.jsonl", "hotspots100"),
    (BASE / "hotspots100_extra.jsonl", "hotspots100"),
    (BASE / "events200.jsonl", None),  # scenario field already present
]
POLICY_ORDER = ("random", "greedy", "greedy_beam", "solo_plan", "coop_dedup", "coop_plan")
SCENARIO_ORDER = ("hotspots100", "events200")


def load() -> dict:
    rows = defaultdict(lambda: defaultdict(list))  # rows[scenario][policy] -> [return, ...]
    for path, default_scenario in SOURCES:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                scenario = row.get("scenario", default_scenario)
                rows[scenario][row["policy"]].append(row["return"])
    return rows


def main() -> None:
    rows = load()
    lines = ["# Baseline reference-policy returns\n"]
    for scenario in SCENARIO_ORDER:
        if scenario not in rows:
            continue
        lines.append(f"## {scenario}\n")
        lines.append("| policy | n | mean | std | min | max |")
        lines.append("|---|---|---|---|---|---|")
        for policy in POLICY_ORDER:
            values = rows[scenario].get(policy)
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            lines.append(
                f"| {policy} | {arr.size} | {arr.mean():.4f} | {arr.std():.4f} | "
                f"{arr.min():.4f} | {arr.max():.4f} |"
            )
        lines.append("")
    out = BASE / "summary_all.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
