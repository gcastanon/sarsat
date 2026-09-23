"""Write one natural-language request per target of one seed, read them back, and rebuild
the episode from the text alone.

Every active target of ``--scenario`` reset with ``jax.random.PRNGKey(--seed)`` (at
``events500``: 15,000 background targets, open all episode at weight 1, and the 5,000
members of the 100 events, each with its event's window at weight 10) becomes one request
written from its raw numbers by ``sarsat.tasking.render_request`` (DECISIONS.md issue 29).
The requests go to ``--output`` as JSON lines in slot order (gzipped if it ends in ``.gz``).

Then, unless ``--no-check``, each text is read back with ``sarsat.tasking.read.read_request``
(which sees only the text and the issue time), the readings are put into the seed's
episode with ``rebuild_state``, and the script reports how far the rebuilt targets are
from the originals and whether every one is within its record's tolerance. With
``--policies``, it also plays those policies (``scripts/baseline_seeds.py`` names) on the
original and the rebuilt episode and reports both returns.

Run: ``python scripts/seed_requests.py --scenario events500 --seed 1000 \
--output runs/requests/events500_seed1000_targets.jsonl --policies greedy_beam``
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import time

import jax
import numpy as np
from baseline_seeds import ALL_POLICIES, build_env, run
from make_requests import issue_time

from sarsat.tasking import extract_targets, load_places, render_request, score
from sarsat.tasking.read import read_request, rebuild_state
from sarsat.tasking.render import parse_iso


def write_lines(path: str, records: list) -> None:
    """JSON lines in UTF-8; a ``.gz`` path is gzipped with no timestamp in the header, so a
    rerun writes the same bytes."""
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    with open(path, "wb") as f:
        f.write(gzip.compress(data, mtime=0) if path.endswith(".gz") else data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="events500")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-population", type=int, default=5000, choices=[1000, 5000, 15000])
    parser.add_argument("--max-tol-km", type=float, default=25.0)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--policies", nargs="*", default=[], choices=ALL_POLICIES)
    args = parser.parse_args()

    env = build_env(args.scenario)
    key = jax.random.PRNGKey(args.seed)
    places = load_places(args.min_population)
    issued = issue_time(args.seed)

    start = time.perf_counter()
    targets = extract_targets(env, key)
    records = []
    for t in targets:
        rng = np.random.default_rng([args.seed, 1, t.slot])  # 1: not the event stream
        record = render_request(t, issued, rng, places, env.step_seconds, args.max_tol_km)
        records.append(
            {
                "id": f"{args.scenario}-{args.seed}-t{t.slot:05d}",
                "scenario": args.scenario,
                "seed": args.seed,
                **record,
            }
        )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_lines(args.output, records)
    print(f"wrote {len(records)} requests to {args.output} ({time.perf_counter() - start:.0f}s)")
    if args.no_check:
        return

    # Back from the text alone: read every request, rebuild the episode, compare.
    start = time.perf_counter()
    readings = [read_request(r["text"], parse_iso(r["issued_utc"]), places) for r in records]
    scores = [score(p, r) for p, r in zip(readings, records, strict=True)]
    km = np.array([s["km"] for s in scores])
    minutes = np.array([max(s["start_min"], s["stop_min"]) for s in scores])
    ok = np.array([s["ok"] for s in scores])
    kinds = collections.Counter(
        ("event" if r["meta"]["cluster"] >= 0 else "background", s["ok"])
        for r, s in zip(records, scores, strict=True)
    )
    print(f"read back {len(readings)} requests ({time.perf_counter() - start:.0f}s)")
    print(
        f"  location error km: median {np.median(km):.2f}, p90 {np.percentile(km, 90):.2f}, "
        f"max {km.max():.2f}; window error min: max {minutes.max():.0f}; "
        f"within tolerance: {ok.sum()}/{len(ok)} {dict(kinds)}"
    )

    original = jax.jit(env._initial_state)(key)
    rebuilt = rebuild_state(env, key, readings, issued)
    same_windows = np.mean(np.asarray(original.target_window) == np.asarray(rebuilt.target_window))
    same_weights = np.allclose(original.target_priority, rebuilt.target_priority, rtol=1e-5)
    print(f"  rebuilt state: windows equal {same_windows:.4f}, weights equal {same_weights}")

    observe = jax.jit(env._observe)
    for policy in args.policies:
        returns = []
        for state in (original, rebuilt):
            returns.append(run(env, state, observe(state), args.seed, policy)[0])
        print(f"  {policy}: original {returns[0]:.4f}, rebuilt {returns[1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
