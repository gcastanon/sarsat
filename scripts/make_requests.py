"""Write natural-language tasking requests for a windowed scenario's events, as JSON lines.

Every event (hotspot cluster) of ``--scenario`` for each seed in ``[--seed-start,
--seed-end]`` (reset with ``jax.random.PRNGKey(seed)``, as ``scripts/baseline_seeds.py``
does) becomes one request: the text is written from the event's raw numbers (centre, window,
priority weight), and the record keeps those numbers as the label, the wording's tolerance
and its exact reading (``sarsat/tasking/render.py`` has the conventions; DECISIONS.md
issue 28). The request is issued at step 0, on a random whole minute of 2026 drawn per
seed; every choice is seeded, so a rerun writes the same file.

Needs ``pip install -e ".[tasking]"`` for the GeoNames place names (without it, pass
``--no-places`` and every location is written as coordinates).

Run: ``python scripts/make_requests.py --scenario events500 --seed-start 1000 \
--seed-end 1015 --output data/requests/events500_seeds1000-1015.jsonl``
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import datetime, timedelta, timezone

import jax
import numpy as np
from baseline_seeds import build_env

from sarsat.tasking import extract_events, load_places, render_request

YEAR_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def issue_time(seed: int) -> datetime:
    """A whole minute of 2026, drawn from ``seed`` alone."""
    minute = np.random.default_rng([seed, 0x1550E]).integers(365 * 24 * 60)
    return YEAR_START + timedelta(minutes=int(minute))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="events500")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-end", type=int, default=1015)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-population", type=int, default=5000, choices=[1000, 5000, 15000])
    parser.add_argument("--max-tol-km", type=float, default=25.0)
    parser.add_argument("--no-places", action="store_true")
    args = parser.parse_args()

    env = build_env(args.scenario)
    places = None if args.no_places else load_places(args.min_population)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    forms = collections.Counter()
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for seed in range(args.seed_start, args.seed_end + 1):
            issued = issue_time(seed)
            for event in extract_events(env, jax.random.PRNGKey(seed)):
                rng = np.random.default_rng([seed, event.cluster])
                record = render_request(
                    event, issued, rng, places, env.step_seconds, args.max_tol_km
                )
                record = {
                    "id": f"{args.scenario}-{seed}-{event.cluster:03d}",
                    "scenario": args.scenario,
                    "seed": seed,
                    **record,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                forms["location " + record["meta"]["loc_form"].split("_")[0]] += 1
                forms["time " + record["meta"]["time_form"]] += 1
            print(f"seed {seed}: issued {issued:%Y-%m-%d %H:%MZ}", flush=True)
    print(json.dumps(dict(sorted(forms.items()))))


if __name__ == "__main__":
    main()
