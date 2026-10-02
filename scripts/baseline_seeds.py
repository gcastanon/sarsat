"""Evaluate reference (and naive) policies across a range of seeds and log JSON lines.

For a fixed ``--scenario`` (``hotspots100``: 100 satellites, 8000 max targets, 40 hotspots
of 50 targets each weighted 10x, 10 planes; ``events200``: the same but 200 satellites, 20
planes, and the hotspots are 10-30 step windowed events over a persistent background at a
10% duty cycle; ``events500``: 500 satellites, 50 planes, 100 events over 15,000 background
targets at a 5% duty cycle; ``announced500``: the same with every request windowed and
announced at random at least 30 minutes ahead, which only the observation sees;
``modes100``: the ``hotspots100`` field with 30 km clusters, four SAR collection modes and
per-target image-quality requirements, DECISIONS.md issue 32; ``modes200`` / ``modes500``:
the ``events200`` / ``events500`` fields with the same modes and requirements, issue 35 -- see
``mapx_integration/mapx/configs/env/scenario/sarsat-*.yaml``),
all with a 180-step time limit, this runs any of ``--policies`` (``random``, ``greedy``, and the
four ``sarsat.reference`` yardsticks ``greedy_beam``, ``solo_plan``, ``coop_dedup``,
``coop_plan``) for every seed in ``[--seed-start, --seed-end]`` (inclusive), resetting with
``jax.random.PRNGKey(seed)``. One JSON line per (seed, policy) is appended to ``--output``
with fields ``scenario``, ``seed``, ``policy``, ``return``, ``looks`` (and ``mode_looks``
when the scenario has collection modes), flushed after each
line so progress can be monitored while the run is in flight.

The four ``sarsat.reference`` policies act on the pre-step ``State`` via a
``ReferenceController`` built fresh per episode; ``greedy`` is the naive
``scripts/greedy_baseline.py`` policy acting on the observation (with modes, slot 0 in the
first mode that images it, ``sarsat.evaluate.greedy_policy_for``); ``random`` draws
``env.action_spec``-shaped uniform actions (continuous slots uniform in [-1, 1], any
categorical "side" slot a fair coin flip) with a look always requested (in a uniformly
random mode, when there are several), from a key folded off ``jax.random.PRNGKey(seed)``
per step -- independent of the env-dynamics reset key. ``looks`` counts looks the battery
paid for.

Uses ``sarsat.reference.lean_precompute`` in place of ``reference.precompute``, as
``scripts/scenario_sweep.py`` does: the default float64 ``(T, N, M)`` tables reach tens of
GB at 200 satellites x 8000 targets, which would OOM a 15 GB host.

Run: ``python scripts/baseline_seeds.py --scenario hotspots100 --seed-start 1000 \
--seed-end 1015 --output runs/baselines/part0.jsonl`` (default ``--policies``: the four
reference yardsticks). One episode per (seed, policy); minutes per seed at 100
satellites, longer at 500 -- GPU recommended for anything beyond a handful of seeds.
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
from sarsat.announce import AnnouncedSarSat
from sarsat.evaluate import greedy_policy_for
from sarsat.quality import MODES, QualitySarSat, WindowedQualitySarSat
from sarsat.reference import REFERENCE_POLICIES, ReferenceController, lean_precompute
from sarsat.windows import WindowedSarSat

ALL_POLICIES = ("random", "greedy", *REFERENCE_POLICIES)

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
    "events500": dict(
        num_satellites=500,
        max_targets=20000,
        hotspots=100,
        hotspot_targets=50,
        hotspot_weight=10.0,
        planes=50,
        window_steps=(10, 30),
        background_windows=False,
        recharge_rate=0.005,
    ),
    "announced500": dict(
        num_satellites=500,
        max_targets=20000,
        hotspots=100,
        hotspot_targets=50,
        hotspot_weight=10.0,
        planes=50,
        window_steps=(10, 30),
        background_windows=True,
        background_window_steps=(30, 90),
        recharge_rate=0.005,
        announce_lead_s=1800.0,
    ),
    "modes100": dict(
        num_satellites=100,
        max_targets=8000,
        hotspots=40,
        hotspot_targets=50,
        hotspot_weight=10.0,
        hotspot_radius_km=30.0,
        planes=10,
        hotspot_quality=(1.0, 3.0, 5.0, 7.0),
        background_quality=(1.0, 3.0, 5.0),
        background_quality_weights=(0.4, 0.4, 0.2),
        modes=MODES,
    ),
    "modes200": dict(
        num_satellites=200,
        max_targets=8000,
        hotspots=40,
        hotspot_targets=50,
        hotspot_weight=10.0,
        hotspot_radius_km=30.0,
        planes=20,
        window_steps=(10, 30),
        background_windows=False,
        recharge_rate=0.01,
        hotspot_quality=(1.0, 3.0, 5.0, 7.0),
        background_quality=(1.0, 3.0, 5.0),
        background_quality_weights=(0.4, 0.4, 0.2),
        modes=MODES,
    ),
    "modes500": dict(
        num_satellites=500,
        max_targets=20000,
        hotspots=100,
        hotspot_targets=50,
        hotspot_weight=10.0,
        hotspot_radius_km=30.0,
        planes=50,
        window_steps=(10, 30),
        background_windows=False,
        recharge_rate=0.005,
        hotspot_quality=(1.0, 3.0, 5.0, 7.0),
        background_quality=(1.0, 3.0, 5.0),
        background_quality_weights=(0.4, 0.4, 0.2),
        modes=MODES,
    ),
}


# The default tables reach tens of GB at 200 satellites x 8,000 targets.
reference.precompute = lean_precompute


def build_env(scenario: str) -> WindowedSarSat:
    """``WindowedSarSat`` for ``scenario`` (== plain ``SarSat`` for ``hotspots100``, since it
    has no ``window_steps`` and ``WindowedSarSat`` with ``window_steps=(0, 0)`` reproduces
    ``SarSat`` entirely); ``AnnouncedSarSat`` when the scenario announces its requests;
    ``QualitySarSat`` when its targets carry quality requirements, ``WindowedQualitySarSat``
    when they also have windows (``modes200``, ``modes500``)."""
    kwargs = SCENARIOS[scenario]
    cls = AnnouncedSarSat if "announce_lead_s" in kwargs else WindowedSarSat
    if "hotspot_quality" in kwargs:
        cls = WindowedQualitySarSat if "window_steps" in kwargs else QualitySarSat
    return cls(**kwargs, time_limit=180)


def random_action(spec, key: jax.Array) -> jax.Array:
    """``spec``-shaped uniform action: continuous slots in [-1, 1], any categorical "side"
    slot a fair coin flip, and a look always requested: ``1``, or a uniformly random mode
    when the look slot has several (its maximum is the number of modes)."""
    n, d = spec.shape
    k_point, k_side = jax.random.split(key)
    pointing = jax.random.uniform(k_point, (n, 2), minval=-1.0, maxval=1.0)
    categorical = d - 2  # side (if side_switch) and look
    side = jax.random.bernoulli(k_side, 0.5, (n, categorical - 1)).astype(jnp.float32)
    num_modes = int(spec.maximum[-1])
    if num_modes > 1:  # a key of its own, so the single-mode draws are unchanged
        look = jax.random.randint(jax.random.fold_in(key, 1), (n, 1), 1, num_modes + 1)
    else:
        look = jnp.ones((n, 1))
    return jnp.concatenate([pointing, side, look.astype(jnp.float32)], axis=-1)


def run(
    env, state0, obs0, seed: int, policy: str, mode_looks: np.ndarray | None = None
) -> tuple[float, int]:
    """Roll out ``policy`` from ``state0``/``obs0`` for one episode; returns (return, looks).
    ``mode_looks``, a ``(num_modes,)`` int array if given, is incremented by the looks made
    in each mode."""
    step_fn = jax.jit(env.step)
    if policy in REFERENCE_POLICIES:
        controller = ReferenceController(env, state0, policy)
        act = lambda state, obs: controller(state)  # noqa: E731
    elif policy == "greedy":
        greedy = greedy_policy_for(env)
        act = lambda state, obs: greedy(obs)  # noqa: E731
    elif policy == "random":
        base_key, spec = jax.random.PRNGKey(seed), env.action_spec
        act = lambda state, obs: random_action(spec, jax.random.fold_in(base_key, state.step))  # noqa: E731
    else:
        raise ValueError(f"policy must be one of {ALL_POLICIES}")

    state, obs, total, looks = state0, obs0, 0.0, 0
    for _ in range(env.time_limit):
        action = act(state, obs)
        requested, mode = env.decode_look(action)
        made = np.asarray(requested & env._can_afford(state.battery, mode))
        looks += int(made.sum())
        if mode_looks is not None:
            mode_looks += np.bincount(np.asarray(mode)[made], minlength=env.num_modes)
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
        "--policies",
        nargs="+",
        choices=ALL_POLICIES,
        default=["greedy_beam", "solo_plan", "coop_dedup", "coop_plan"],
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
                mode_looks = np.zeros(env.num_modes, int)
                ret, looks = run(env, state0, timestep0.observation, seed, policy, mode_looks)
                elapsed = time.perf_counter() - start
                record = {
                    "scenario": args.scenario,
                    "seed": seed,
                    "policy": policy,
                    "return": ret,
                    "looks": looks,
                }
                if env._mode_flags:  # looks per collection mode, by name
                    record["mode_looks"] = dict(zip((m.name for m in env.modes), mode_looks.tolist()))
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(
                    f"{args.scenario:12s} seed={seed} policy={policy:11s} "
                    f"return={ret:.3f} looks={looks:6d} ({elapsed:.1f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
