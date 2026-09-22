"""Measure how much cooperation is worth in a SarSat configuration.

Runs four reference policies through the real environment and reports the
priority-weighted fraction of targets imaged:

* ``greedy_beam`` -- independent, beam-aware: each satellite aims where its beam captures
                    the most priority and senses whenever it can. The fair baseline.
* ``solo_plan``  -- independent, far-sighted: also weighs a target by how rarely *it alone*
                    will see it again and rations its looks to its budget. Temporal
                    planning without any knowledge of teammates.
* ``coop_dedup`` -- centralised: no target is taken twice in one step.
* ``coop_plan``  -- centralised: dedup, plus scarcity weighted by *every* satellite's
                    future access, plus the look budget. The cooperative reference.

The gap between ``coop_plan`` and the better independent policy is what a coordinating
multi-agent learner could gain over an independent one. The centralised policies know the
whole constellation's future access, so they are a yardstick, not a bound on MARL.

Run: ``python scripts/cooperation_gap.py --satellites 100 --planes 10 --hotspots 40``
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from sarsat import SarSat
from sarsat.reference import REFERENCE_POLICIES, ReferenceController, precompute  # noqa: F401


def run(env, state0, policy):
    controller = ReferenceController(env, state0, policy)
    state, total, honoured = state0, 0.0, 0
    step_fn = jax.jit(env.step)
    sense_slot = env.action_dim - 1
    for _ in range(env.time_limit):
        action = controller(state)
        sensing = np.asarray(action[:, sense_slot] > 0.5)
        honoured += int((sensing & np.asarray(env._can_sense(state.battery))).sum())
        state, timestep = step_fn(state, action)
        total += float(timestep.reward[0])
        if bool(timestep.last()):
            break
    return total, honoured


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--satellites", type=int, default=100)
    p.add_argument("--background", type=int, default=6000, help="uniform low-value targets")
    p.add_argument("--hotspots", type=int, default=40)
    p.add_argument("--hotspot-targets", type=int, default=50)
    p.add_argument("--hotspot-weight", type=float, default=10.0)
    p.add_argument("--planes", type=int, default=10, help="0 for independent random orbits")
    p.add_argument("--train-spacing-s", type=float, default=0.0)
    p.add_argument("--time-limit", type=int, default=180)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = SarSat(
        num_satellites=args.satellites,
        max_targets=args.background + args.hotspots * args.hotspot_targets,
        time_limit=args.time_limit,
        hotspots=args.hotspots,
        hotspot_targets=args.hotspot_targets,
        hotspot_weight=args.hotspot_weight,
        planes=args.planes,
        train_spacing_s=args.train_spacing_s,
    )
    state, _ = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    start = time.perf_counter()
    results = {}
    for policy in REFERENCE_POLICIES:
        results[policy] = run(env, state, policy)
        print(
            f"{policy:11s} return {results[policy][0]:.3f}  looks {results[policy][1]:5d}",
            flush=True,
        )
    independent = max(results["greedy_beam"][0], results["solo_plan"][0])
    print(
        f"\ncooperation gap: {results['coop_plan'][0] / independent - 1:+.0%} over the best "
        f"independent policy ({time.perf_counter() - start:.0f}s)"
    )


if __name__ == "__main__":
    main()
