"""Headless soak of the live world: rolling episodes of the trained actor at full speed.

Prints, per episode, the return (sum of normalised priority imaged, comparable with the
offline evaluation when no requests are injected), the imaged fraction of the events whose
windows closed, the capture rate by episode phase (first vs last 30 steps, to see whether
the boundary hand-over changes behaviour) and the step time. With ``--requests`` it drops
that many requests per episode at random land points and reports their collection rate.

    PYTHONPATH=. live/.venv/Scripts/python -m live.soak --episodes 3 --requests 10
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from live.policy import DEFAULT_PARAMS, load_policy, make_env
from sarsat.live import LiveWorld
from sarsat.targets import TargetSampler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--requests", type=int, default=0, help="requests per episode")
    ap.add_argument("--deadline", type=int, default=30)
    ap.add_argument("--priority", default="normal")
    args = ap.parse_args()

    env = make_env()
    policy = load_policy(args.params, env)
    world = LiveWorld(env, seed=args.seed)
    sampler = TargetSampler()
    rng = np.random.default_rng(args.seed)
    limit = env.time_limit
    phase = 30

    policy.reset()
    request_steps = set()
    for ep in range(args.episodes):
        if args.requests:
            request_steps = set(rng.choice(np.arange(1, limit - 1), args.requests, replace=False))
        captured_by_phase = [0.0, 0.0]
        times = []
        for _ in range(limit):
            step = int(world.state.step)
            if step in request_steps:
                point = np.asarray(
                    sampler(jax.random.PRNGKey(rng.integers(1 << 30)), jnp.ones((1,), bool))
                )[0]
                r = world.inject_request(*point, args.priority, args.deadline)
                first = r.first_access_global
                wait = None if first is None else first - world.global_step
                print(
                    f"  ep {ep} step {step}: request {r.id} at ({r.lat:.1f}, {r.lon:.1f}) "
                    f"feasible by {r.feasible_count} sats, first in {wait} steps"
                )
            t0 = time.perf_counter()
            action = policy(world.obs)
            result = world.step(action)
            times.append(time.perf_counter() - t0)
            if result.step <= phase:
                captured_by_phase[0] += result.reward
            elif result.step > limit - phase:
                captured_by_phase[1] += result.reward
            for r in result.request_updates:
                if r.status != "pending":
                    took = (r.collected_global or world.global_step) - r.submitted_global
                    print(
                        f"  ep {ep} step {result.step}: request {r.id} {r.status} in {took} steps"
                    )
            if result.new_episode:
                policy.reset()
        events = [e for e in world.events if e.episode == ep]
        done = [r for r in world.requests if r.status != "pending"]
        collected = sum(r.status == "collected" for r in done)
        print(
            f"episode {ep}: return {world.episode_returns[-1]:.4f}, events closed {len(events)} "
            f"imaged {np.mean([e.imaged_fraction for e in events]):.3f}, "
            f"reward first/last {phase} steps "
            f"{captured_by_phase[0]:.4f}/{captured_by_phase[1]:.4f}, "
            f"battery {float(np.mean(result.battery)):.3f}, "
            f"step {np.median(times) * 1e3:.0f} ms (max {np.max(times) * 1e3:.0f}), "
            f"requests collected {collected}/{len(done)}"
        )


if __name__ == "__main__":
    main()
