"""Aggregate reference-policy and trained-policy returns into reports/results.json.

Sources:

* ``runs/baselines/*.jsonl`` -- rows ``{scenario?, seed, policy, return, looks?}``. Rows
  with no ``scenario`` key are the original ``sarsat-100sat-hotspots`` baseline run and are
  treated as scenario ``hotspots100``. Several files in this directory are historical
  shards of one another (``part0..3.jsonl`` concatenate to ``baselines.jsonl``), so rows
  are de-duplicated by exact content before aggregating.
* ``<runs-dir>/<run>/eval_seeds.json`` -- one per trained run, holding paired per-seed
  scores: ``{"results": [{"seed", "policy_return", "greedy_beam", "coop_plan",
  "solo_plan"}, ...], "summary": {...}}``. Which runs map to which scenario/label is given
  by the ``RUN_INFO`` table below.

Output: ``reports/results.json``, ``{scenario: [row, ...]}`` where each row is
``{label, kind, mean, std, min, max, n, beats_greedy_beam, per_seed}``.
``beats_greedy_beam`` (rl rows only) counts seeds where ``policy_return`` exceeds that
run's own paired ``greedy_beam`` figure for the same seed.
"""

import argparse
import glob
import json
import os
import statistics as stats
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINES_GLOB = os.path.join(REPO_ROOT, "runs", "baselines", "*.jsonl")
DEFAULT_RESULTS_OUT = os.path.join(REPO_ROOT, "reports", "results.json")

# Reference-policy rows, in the display order used on the slides.
REFERENCE_POLICIES = [
    ("random", "Random"),
    ("greedy", "Greedy"),
    ("greedy_beam", "Greedy-Beam"),
    ("solo_plan", "Solo-Plan"),
    ("coop_dedup", "Coop-Dedup"),
    ("coop_plan", "Coop-Plan"),
]

TEST_SEEDS = (1000, 1015)  # the paired test set every table reports

# run name -> (scenario, label). Order here is the display order for RL rows.
RUN_INFO = {
    # sarsat-100sat-hotspots (100 satellites, non-windowed)
    "mappo_v1": ("hotspots100", "MAPPO, team reward, slot-0 head"),
    "mappo_v3": ("hotspots100", "MAPPO, credit_mix 0.5 (from v1)"),
    "ippo_v4": ("hotspots100", "IPPO, credit_mix 0.5, slot-0 head"),
    "ippo_v5": ("hotspots100", "IPPO, credit_mix 0.5, slot-mixture head"),
    "mappo_v5": ("hotspots100", "MAPPO, credit_mix 0.5, slot-mixture head"),
    "h100_s11": ("hotspots100", "MAPPO, credit_mix 0.5, slot-mixture, seed 11"),
    "h100_s23": ("hotspots100", "MAPPO, credit_mix 0.5, slot-mixture, seed 23"),
    "h100_wide": ("hotspots100", "MAPPO, credit_mix 0.5, 6 slots / 24-step look-ahead"),
    "h100_ft": ("hotspots100", "MAPPO, v5 fine-tuned, credit_mix 0.25"),
    # sarsat-200sat-events (200 satellites, windowed)
    "ev_ippo_a": ("events200", "IPPO, credit_mix 0.5"),
    "ev_mappo_a": ("events200", "MAPPO, credit_mix 0.5"),
    "ev_mappo_c": ("events200", "MAPPO, credit_mix 0.1, look-ahead 48"),
    "ev_mappo_d": ("events200", "MAPPO, credit_mix 0.1"),
    "ev_mappo_e": ("events200", "MAPPO, credit_mix 0.25, gamma 0.998 / lambda 0.98"),
}

# ippo_v3 also appears in DECISIONS.md (slot-0 head, credit_mix 0.5) but is superseded by
# ippo_v4/v5 in the caller's mapping above and is intentionally not duplicated here.


def _round(x):
    return round(x, 6)


def load_reference_rows(baselines_glob):
    """scenario -> policy -> {seed: return}, de-duplicated across overlapping shard files."""
    seen_rows = set()
    data = defaultdict(lambda: defaultdict(dict))
    files = sorted(glob.glob(baselines_glob))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = tuple(sorted(row.items()))
                if key in seen_rows:
                    continue
                seen_rows.add(key)
                scenario = row.get("scenario", "hotspots100")
                policy = row["policy"]
                seed = row["seed"]
                if not TEST_SEEDS[0] <= seed <= TEST_SEEDS[1]:
                    continue  # validation seeds (2000-2015) are reported separately
                data[scenario][policy][seed] = row["return"]
    return data, files


def summarize(values):
    values = list(values)
    n = len(values)
    mean = stats.fmean(values) if n else float("nan")
    std = stats.pstdev(values) if n > 1 else 0.0
    return {
        "mean": _round(mean),
        "std": _round(std),
        "min": _round(min(values)) if n else float("nan"),
        "max": _round(max(values)) if n else float("nan"),
        "n": n,
    }


def build_reference_row(label, seed_to_return):
    seeds = sorted(seed_to_return)
    summary = summarize(seed_to_return[s] for s in seeds)
    return {
        "label": label,
        "kind": "reference",
        **summary,
        "beats_greedy_beam": None,
        "per_seed": [{"seed": s, "return": _round(seed_to_return[s])} for s in seeds],
    }


def build_rl_row(label, results):
    """``results`` is the eval_seeds.json ``results`` list for one run."""
    returns = [r["policy_return"] for r in results]
    summary = summarize(returns)
    beats = sum(1 for r in results if r["policy_return"] > r["greedy_beam"])
    per_seed = [
        {
            "seed": r["seed"],
            "return": _round(r["policy_return"]),
            "greedy_beam": _round(r["greedy_beam"]),
            "beats_greedy_beam": bool(r["policy_return"] > r["greedy_beam"]),
        }
        for r in results
    ]
    return {
        "label": label,
        "kind": "rl",
        **summary,
        "beats_greedy_beam": f"{beats}/{len(results)}",
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-glob", default=DEFAULT_BASELINES_GLOB)
    parser.add_argument(
        "--runs-dir",
        default=os.path.expanduser("~/marl/runs"),
        help="Directory holding one subdirectory per trained run (WSL path by default).",
    )
    parser.add_argument("--out", default=DEFAULT_RESULTS_OUT)
    args = parser.parse_args()

    ref_data, ref_files = load_reference_rows(args.baselines_glob)
    print(f"Read {len(ref_files)} baseline file(s):")
    for f in ref_files:
        print(f"  {f}")

    results = defaultdict(list)
    notes = []

    for scenario, files_by_scenario in ref_data.items():
        for policy_key, label in REFERENCE_POLICIES:
            if policy_key not in files_by_scenario:
                notes.append(f"reference policy '{policy_key}' missing for scenario '{scenario}'")
                continue
            results[scenario].append(build_reference_row(label, files_by_scenario[policy_key]))

    for run, (scenario, label) in RUN_INFO.items():
        path = os.path.join(args.runs_dir, run, "eval_seeds.json")
        if not os.path.exists(path):
            notes.append(f"run '{run}' skipped: no eval_seeds.json at {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row = build_rl_row(label, payload["results"])
        row["run"] = run
        results[scenario].append(row)

    out = dict(results)
    out["_notes"] = notes
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {args.out}")
    for scenario, rows in results.items():
        print(f"\n[{scenario}]")
        for row in rows:
            beats = row["beats_greedy_beam"]
            beats_str = f" beats_greedy_beam={beats}" if beats is not None else ""
            print(
                f"  {row['kind']:>9} {row['label']:<45} "
                f"mean={row['mean']:.3f} std={row['std']:.3f} "
                f"n={row['n']}{beats_str}"
            )
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
