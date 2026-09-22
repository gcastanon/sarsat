"""Evaluate reference (and naive) policies across a range of seeds and log JSON lines.

For a fixed ``--scenario`` (``hotspots100``: 100 satellites, 8000 max targets, 40 hotspots
of 50 targets each weighted 10x, 10 planes; ``events200``: the same but 200 satellites, 20
planes, and the hotspots are 10-30 step windowed events over a persistent background at a
10% duty cycle -- see ``mapx_integration/mapx/configs/env/scenario/sarsat-*.yaml``), both
with a 180-step time limit, this runs any of ``--policies`` (``random``, ``greedy``, and the
four ``sarsat.reference`` yardsticks ``greedy_beam``, ``solo_plan``, ``coop_dedup``,
``coop_plan``) for every seed in ``[--seed-start, --seed-end]`` (inclusive), resetting with
``jax.random.PRNGKey(seed)``. One JSON line per (seed, policy) is appended to ``--output``
with fields ``scenario``, ``seed``, ``policy``, ``return``, ``looks``, flushed after each
line so progress can be monitored while the run is in flight.

The four ``sarsat.reference`` policies act on the pre-step ``State`` via a
``ReferenceController`` built fresh per episode; ``greedy`` is the naive
``scripts/greedy_baseline.py`` policy acting on the observation; ``random`` draws
``env.action_spec``-shaped uniform actions (continuous slots uniform in [-1, 1], any
categorical "side" slot a fair coin flip) with sense always requested, from a key folded
off ``jax.random.PRNGKey(seed)`` per step -- independent of the env-dynamics reset key.

Uses the same lean (int32/float32) ``reference.precompute`` monkeypatch as
``scripts/scenario_sweep.py``: the default float64 ``(T, N, M)`` tables reach tens of GB at
200 satellites x 8000 targets, which would OOM a 15 GB host.

Run: ``python scripts/baseline_seeds.py --scenario hotspots100 --seed-start 1000 \
--seed-end 1015 --policies greedy_beam solo_plan coop_plan --output runs/baselines/part0.jsonl``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import sarsat.reference as reference
from sarsat.evaluate import greedy_policy
from sarsat.reference import REFERENCE_POLICIES, ReferenceController
from sarsat.windows import WindowedSarSat

ALL_POLICIES = ("random", "greedy") + REFERENCE_POLICIES

# task_config kwargs from mapx_integration/mapx/configs/env/scenario/sarsat-*.yaml.
SCENARIOS = {
    "hotspots100": dict(
        num_satellites=100,
        max_targets=8000,
        hotspots=40,
        hotspot_targets=50,
        hotspot_weight=10.0,
        planes=10,
    ),
    "events200": dict(
        num_satellites=200,
        max_targets=8000,
        hotspots=40,
        hotspot_targets=50,
        hotspot_weight=10.0,
        planes=20,
        window_steps=(10, 30),
        background_windows=False,
        recharge_rate=0.01,
    ),
}


def lean_precompute(env, state):
    """``reference.precompute`` with int32 / float32 tables: the ``(T, N, M)`` arrays reach
    tens of GB in the default dtypes at 200 satellites x 8,000 targets."""
    steps = jnp.arange(1, env.time_limit + 1)
    geometry = jax.jit(jax.vmap(lambda s: env._target_geometry(state, s)[1]))
    access = np.concatenate(
        [np.asarray(geometry(steps[i : i + 20])) for i in range(0, len(steps), 20)]
    )
    prio = np.asarray(state.target_priority, np.float32)
    team = np.cumsum(access.sum(1, dtype=np.int32)[::-1], 0, dtype=np.int32)[::-1]
    own = np.cumsum(access[::-1], 0, dtype=np.int32)[::-1]
    best_team = np.einsum(
        "tnm,tm->tnm", access, prio[None] / np.maximum(team, 1).astype(np.float32)
    ).max(2)
    best_own = np.where(access, prio[None, None] / np.maximum(own, 1), np.float32(0)).max(2)
    return dict(access=access, team=team, own=own, best_team=best_team, best_own=best_own)


reference.precompute = lean_precompute


def build_env(scenario: str) -> WindowedSarSat:
    """``WindowedSarSat`` for ``scenario`` (== plain ``SarSat`` for ``hotspots100``, since it
    has no ``window_steps`` and ``WindowedSarSat`` with ``window_steps=(0, 0)`` reproduces
    ``SarSat`` entirely)."""
    return WindowedSarSat(**SCENARIOS[scenario], time_limit=180)


def random_action(spec, key: jax.Array) -> jax.Array:
    """``spec``-shaped uniform action: continuous slots in [-1, 1], any categorical "side"
    slot a fair coin flip, sense always 1."""
    n, d = spec.shape
    k_point, k_side = jax.random.split(key)
    pointing = jax.random.uniform(k_point, (n, 2), minval=-1.0, maxval=1.0)
    categorical = d - 2  # side (if side_switch) and sense
    side = jax.random.bernoulli(k_side, 0.5, (n, categorical - 1)).astype(jnp.float32)
    sense = jnp.ones((n, 1), jnp.float32)
    return jnp.concatenate([pointing, side, sense], axis=-1)


def run(env, state0, obs0, seed: int, policy: str) -> tuple[float, int]:
    """Roll out ``policy`` from ``state0``/``obs0`` for one episode; returns (return, looks)."""
    step_fn = jax.jit(env.step)
    if policy in REFERENCE_POLICIES:
        controller = ReferenceController(env, state0, policy)
        act = lambda state, obs: controller(state)  # noqa: E731
    elif policy == "greedy":
        act = lambda state, obs: greedy_policy(obs)  # noqa: E731
    elif policy == "random":
        base_key, spec = jax.random.PRNGKey(seed), env.action_spec
        act = lambda state, obs: random_action(spec, jax.random.fold_in(base_key, state.step))  # noqa: E731
    else:
        raise ValueError(f"policy must be one of {ALL_POLICIES}")

    state, obs, total, looks = state0, obs0, 0.0, 0
    for _ in range(env.time_limit):
        action = act(state, obs)
        sensing = np.asarray(action[:, -1] > 0.5)
        looks += int((sensing & np.asarray(env._can_sense(state.battery))).sum())
        state, timestep = step_fn(state, action)
        obs = timestep.observation
        total += float(timestep.reward[0])
        if bool(timestep.last()):
            break
    return total, looks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="hotspots100")
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--seed-end", type=int, required=True, help="inclusive")
    p.add_argument(
        "--policies", nargs="+", choices=ALL_POLICIES, default=["greedy_beam", "solo_plan", "coop_plan"]
    )
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    env = build_env(args.scenario)
    reset_fn = jax.jit(env.reset)

    with open(args.output, "a") as f:
        for seed in range(args.seed_start, args.seed_end + 1):
            state0, timestep0 = reset_fn(jax.random.PRNGKey(seed))
            for policy in args.policies:
                start = time.perf_counter()
                ret, looks = run(env, state0, timestep0.observation, seed, policy)
                elapsed = time.perf_counter() - start
                f.write(
                    json.dumps(
                        {
                            "scenario": args.scenario,
                            "seed": seed,
                            "policy": policy,
                            "return": ret,
                            "looks": looks,
                        }
                    )
                    + "\n"
                )
                f.flush()
                print(
                    f"{args.scenario:12s} seed={seed} policy={policy:11s} "
                    f"return={ret:.3f} looks={looks:6d} ({elapsed:.1f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
