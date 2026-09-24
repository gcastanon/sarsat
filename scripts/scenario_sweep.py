"""Measure the cooperation gap of candidate scenarios (reference policies, one seed each).

Like ``scripts/cooperation_gap.py`` but over a table of named configurations, with the
battery exposed, and writing one JSON line per (config, policy) so partial results survive.

Run: ``python scripts/scenario_sweep.py --configs walker20 pairs --out runs/sweep.jsonl``
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import numpy as np

import sarsat.reference as reference
from sarsat import SarSat
from sarsat.reference import ReferenceController, lean_precompute
from sarsat.windows import WindowedSarSat

# The default tables reach tens of GB at 200 satellites x 8000 targets (and more at 500).
reference.precompute = lean_precompute

BASE = dict(max_targets=8000, hotspot_targets=50, hotspot_weight=10.0, time_limit=180)
BIG = dict(max_targets=16000, hotspots=80)  # 4000 hotspot + 12000 background targets
# Events over a persistent background: only the hotspot clusters are windowed.
EV = dict(
    num_satellites=200, planes=20, hotspots=40, window_steps=(10, 30), background_windows=False
)
# 500 satellites over a field scaled by the same 2.5x: 100 event clusters over 15,000
# persistent background targets, 10% duty cycle.
EV500 = dict(
    num_satellites=500,
    planes=25,
    max_targets=20000,
    hotspots=100,
    window_steps=(10, 30),
    background_windows=False,
    recharge_rate=0.01,
)
# The sarsat-500sat-events benchmark itself: 50 planes of 10, 5% duty cycle.
EV500_BENCH = EV500 | dict(planes=50, recharge_rate=0.005)
CONFIGS = {
    # 100-satellite benchmark, for reference.
    "ref100": dict(num_satellites=100, planes=10, hotspots=40),
    # 200 satellites: more planes (same overlap per plane) vs the same planes (denser).
    "walker20": dict(num_satellites=200, planes=20, hotspots=40),
    "walker10": dict(num_satellites=200, planes=10, hotspots=40),
    # More, and heavier, hotspots: the priority contrast lever.
    "hot80": dict(num_satellites=200, planes=20, hotspots=80),
    "hot80w30": dict(num_satellites=200, planes=20, hotspots=80, hotspot_weight=30.0),
    # Tighter battery: 10% sustainable duty cycle instead of 20%.
    "tight": dict(num_satellites=200, planes=20, hotspots=40, recharge_rate=0.01),
    "hot80tight": dict(num_satellites=200, planes=20, hotspots=80, recharge_rate=0.01),
    # Formation pairs (planes of two, 3 s apart) share one view: the dedup lever.
    "pairs": dict(num_satellites=200, planes=100, hotspots=40, train_spacing_s=3.0),
    "pairs_tight": dict(
        num_satellites=200, planes=100, hotspots=40, train_spacing_s=3.0, recharge_rate=0.01
    ),
    # Twice the targets for twice the satellites, so greedy_beam does not saturate.
    "big": BIG | dict(num_satellites=200, planes=20),
    "big_tight": BIG | dict(num_satellites=200, planes=20, recharge_rate=0.01),
    "big_w30": BIG | dict(num_satellites=200, planes=20, hotspot_weight=30.0),
    "big_pairs": BIG | dict(num_satellites=200, planes=100, train_spacing_s=3.0),
    "big_pairs_tight": BIG
    | dict(num_satellites=200, planes=100, train_spacing_s=3.0, recharge_rate=0.01),
    # Half the episode instead of twice the targets.
    "t90": dict(num_satellites=200, planes=20, hotspots=40, time_limit=90),
    "t90_tight": dict(
        num_satellites=200, planes=20, hotspots=40, time_limit=90, recharge_rate=0.01
    ),
    "t90_pairs": dict(
        num_satellites=200, planes=100, hotspots=40, train_spacing_s=3.0, time_limit=90
    ),
    # Time-windowed targets (sarsat.windows): hotspot clusters are events that all of their
    # targets share; the window length is uniform in window_steps.
    "win100_20": dict(num_satellites=100, planes=10, hotspots=40, window_steps=(10, 30)),
    "win100_40": dict(num_satellites=100, planes=10, hotspots=40, window_steps=(20, 60)),
    "win200_20": dict(num_satellites=200, planes=20, hotspots=40, window_steps=(10, 30)),
    "win200_40": dict(num_satellites=200, planes=20, hotspots=40, window_steps=(20, 60)),
    "win200_10": dict(num_satellites=200, planes=20, hotspots=40, window_steps=(5, 15)),
    # Events over a persistent background: only the hotspot clusters are windowed.
    "ev100": dict(
        num_satellites=100, planes=10, hotspots=40, window_steps=(10, 30), background_windows=False
    ),
    "ev100_hot80": dict(
        num_satellites=100, planes=10, hotspots=80, window_steps=(10, 30), background_windows=False
    ),
    "ev200_hot80": dict(
        num_satellites=200, planes=20, hotspots=80, window_steps=(10, 30), background_windows=False
    ),
    "ev200_hot80_w30": dict(
        num_satellites=200,
        planes=20,
        hotspots=80,
        hotspot_weight=30.0,
        window_steps=(10, 30),
        background_windows=False,
    ),
    "ev200_hot80_short": dict(
        num_satellites=200, planes=20, hotspots=80, window_steps=(5, 15), background_windows=False
    ),
    # Keeping 200 satellites from covering the events by brute force.
    "ev200": EV,
    "ev200_tight": EV | dict(recharge_rate=0.01),  # -> sarsat-200sat-events
    "ev200_big": EV | dict(hotspot_targets=100),
    "ev200_p10": EV | dict(planes=10),
    "ev200_t90": EV | dict(time_limit=90),
    "ev200_big_tight": EV | dict(hotspot_targets=100, recharge_rate=0.01),
    "ev200_t90_tight": dict(
        num_satellites=200,
        planes=20,
        hotspots=40,
        window_steps=(10, 30),
        background_windows=False,
        time_limit=90,
        recharge_rate=0.01,
    ),
    "t90_pairs_tight": dict(
        num_satellites=200,
        planes=100,
        hotspots=40,
        train_spacing_s=3.0,
        time_limit=90,
        recharge_rate=0.01,
    ),
    # 500 satellites (DECISIONS.md issue 26). Walker planes only: no trains or formation
    # pairs. The 200-satellite event field first, then the field scaled with the
    # constellation (2.5x the events and background), and the levers on top of that.
    "ev500_same": EV500 | dict(max_targets=8000, hotspots=40),
    "ev500": EV500,
    "ev500_p50": EV500 | dict(planes=50),
    "ev500_rand": EV500 | dict(planes=0),
    "ev500_tight5": EV500 | dict(recharge_rate=0.005),
    "ev500_short": EV500 | dict(window_steps=(5, 15)),
    "ev500_long": EV500 | dict(window_steps=(20, 60)),
    "ev500_w30": EV500 | dict(hotspot_weight=30.0),
    "ev500_hot200": EV500 | dict(hotspots=200),
    "ev500_big": EV500 | dict(hotspot_targets=100),
    "ev500_t90": EV500 | dict(time_limit=90),
    "ev500_t90_tight5": EV500 | dict(time_limit=90, recharge_rate=0.005),
    # Around the 5% duty cycle, which opens the solo_plan gap.
    "ev500_tight7": EV500 | dict(recharge_rate=0.0075),
    "ev500_hot200_tight5": EV500 | dict(hotspots=200, recharge_rate=0.005),
    "ev500_long_tight5": EV500 | dict(window_steps=(20, 60), recharge_rate=0.005),
    "ev500_p50_tight5": EV500 | dict(planes=50, recharge_rate=0.005),
    "ev500_rand_tight5": EV500 | dict(planes=0, recharge_rate=0.005),
    # Every request windowed (DECISIONS.md issue 28): sarsat-500sat-events with the
    # background windowed too, at the events' 10-30 steps or its own longer lengths.
    "ann500": EV500_BENCH | dict(background_windows=True),
    "ann500_bg90": EV500_BENCH | dict(background_windows=True, background_window_steps=(30, 90)),
    "ann500_bg180": EV500_BENCH | dict(background_windows=True, background_window_steps=(60, 180)),
}


def run(env: SarSat, state0, policy: str) -> tuple[float, int]:
    controller = ReferenceController(env, state0, policy)
    state, total, honoured = state0, 0.0, 0
    step_fn = jax.jit(env.step)
    for _ in range(env.time_limit):
        action = controller(state)
        sensing = np.asarray(action[:, -1] > 0.5)
        honoured += int((sensing & np.asarray(env._can_sense(state.battery))).sum())
        state, timestep = step_fn(state, action)
        total += float(timestep.reward[0])
        if bool(timestep.last()):
            break
    return total, honoured


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--policies", nargs="+", default=["greedy_beam", "solo_plan", "coop_plan"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    for name in args.configs:
        kwargs = BASE | CONFIGS[name]
        env = WindowedSarSat(**kwargs)  # == SarSat unless window_steps is set
        for seed in args.seeds:
            state, _ = jax.jit(env.reset)(jax.random.PRNGKey(seed))
            results = {}
            for policy in args.policies:
                start = time.perf_counter()
                ret, looks = run(env, state, policy)
                results[policy] = ret
                print(
                    f"{name:16s} seed {seed} {policy:11s} return {ret:.3f}  looks {looks:6d}  "
                    f"({time.perf_counter() - start:.0f}s)",
                    flush=True,
                )
                if args.out:
                    with open(args.out, "a") as f:
                        row = dict(config=name, seed=seed, policy=policy, ret=ret, looks=looks)
                        f.write(json.dumps(row | {"kwargs": kwargs}) + "\n")
            if "coop_plan" in results and "greedy_beam" in results:
                gaps = " ".join(
                    f"{p}: {results['coop_plan'] / results[p] - 1:+.0%}"
                    for p in ("greedy_beam", "solo_plan")
                    if p in results
                )
                print(f"{name:16s} seed {seed} gap of coop_plan over {gaps}", flush=True)


if __name__ == "__main__":
    main()
