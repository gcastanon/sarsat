"""Diagnose WHERE a trained MAPX/MARL policy loses value relative to the ``coop_plan``
reference on ``CoopSarSat``.

For a handful of seeds this runs three policies through matched episodes -- the trained
checkpoint (greedy/mode actions), and the ``greedy_beam`` / ``coop_plan`` reference
controllers from ``sarsat.reference`` -- and records, step by step, where the return
comes from: hotspot vs. background targets, look quality vs. look count, battery usage
over time, and (for the trained policy only) how closely its executed pointing tracks
the beam-value observation slots it was given.

The scenario for all three policies comes from the run's own hydra config: ``env.kwargs``
merged with ``env.scenario.task_config``, exactly as ``mapx.wrappers.sarsat_coop.
make_sarsat_coop_envs`` merges them to build the training envs. The ``--num-satellites``
etc. flags are fallback *overrides* on top of that merge (applied identically to the
trained and reference scenarios, since matched episodes require the same scenario); they
are not needed for a normal run and default to leaving the hydra config alone.

The trained policy is stepped through the raw ``CoopSarSat``/``WindowedCoopSarSat`` env
exactly as ``eval_checkpoint.py`` does (same actor rebuild, same checkpoint restore, same
per-seed reset that bypasses ``SarSatCoopWrapper.reset``'s extra key split). The reference
policies run on a plain ``sarsat.windows.WindowedSarSat`` (which equals ``SarSat`` unless
the scenario sets ``window_steps``) built with the same scenario kwargs and reset with the
same ``jax.random.PRNGKey(seed)``: since ``CoopSarSat.reset_state`` calls ``SarSat.reset``
with an identical key-splitting sequence, the sampled orbits and targets are bit-identical
to the trained policy's episode, so returns are directly comparable.

For scenarios with event-like (windowed) hotspots, every policy also gets a set of
window-aware metrics: how much of an honoured look's value came from a hotspot target
(always inside its window -- visibility itself is window-gated, see ``sarsat.windows``),
how often a satellite spent a look on something else while an event target was visible to
it, the battery satellites had when an event first came into their reach, and how much
hotspot priority was reachable by some charged satellite during its window but never
imaged.

Must run on CPU -- the GPU is reserved for training:

    JAX_PLATFORMS=cpu python diagnose_policy.py \\
        --run-dir ~/marl/runs/mappo_v1 --system mappo \\
        --seeds 1000,1003,1009 --out ~/marl/runs/mappo_v1/diagnosis.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

# The ``mapx/`` overlay next to this script would shadow the installed package (see the
# identical fix at the top of ``eval_checkpoint.py`` / ``train_sarsat.py``).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

os.environ.setdefault("JAX_PLATFORMS", "cpu")

CHECKPOINT_STEPS = (30, 60, 90, 120, 150, 180)
VALUE_BINS = ((0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 1.0))
DEFAULT_TOL = 0.05

# A small scenario for --smoke-test: fast enough to sanity-check the pipeline in seconds.
# Safe to also apply to the trained policy's env, since none of these keys affect network
# shapes (those depend only on num_slots/horizon/hidden_dim; see make_sarsat_coop_envs).
SMOKE_OVERRIDE: Dict[str, Any] = dict(
    num_satellites=8,
    max_targets=100,
    hotspots=2,
    hotspot_targets=5,
    hotspot_weight=10.0,
    planes=2,
    time_limit=20,
)


def _load_eval_checkpoint():
    """Load ``eval_checkpoint.py`` (a sibling file) by path, reusing its actor-rebuild /
    checkpoint-restore helpers without needing ``_HERE`` back on ``sys.path``."""
    path = os.path.join(_HERE, "eval_checkpoint.py")
    spec = importlib.util.spec_from_file_location("eval_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _merge_scenario_kwargs(cfg) -> Dict[str, Any]:
    """``env.kwargs`` merged with ``env.scenario.task_config``, exactly as
    ``make_sarsat_coop_envs`` merges them to build the training/eval envs."""
    from omegaconf import OmegaConf

    kwargs = OmegaConf.to_container(cfg.env.get("kwargs", {}), resolve=True)
    kwargs |= OmegaConf.to_container(cfg.env.scenario.task_config, resolve=True)
    return kwargs


def _sarsat_only_kwargs(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a merged env-kwargs dict down to what ``sarsat.windows.WindowedSarSat``
    accepts (``SarSat.__init__``'s parameters plus ``window_steps``/``background_windows``),
    dropping ``CoopSarSat``-only keys such as ``num_slots``/``num_candidates``/``horizon``
    that the reference policies' plain env has no use for."""
    import inspect

    from sarsat.env import SarSat

    allowed = set(inspect.signature(SarSat.__init__).parameters) - {"self"}
    allowed |= {"window_steps", "background_windows"}
    return {k: v for k, v in merged.items() if k in allowed}


def lean_precompute(env, state):
    """``sarsat.reference.precompute`` with int32 / float32 tables (see
    ``scripts/scenario_sweep.py``): the ``(T, N, M)`` arrays reach 16 GB in the default
    dtypes at 200 satellites x thousands of targets. Monkeypatched over
    ``sarsat.reference.precompute`` before any ``ReferenceController`` that needs it
    (``solo_plan``/``coop_plan``; ``greedy_beam`` never calls it)."""
    import jax
    import numpy as np

    steps = jax.numpy.arange(1, env.time_limit + 1)
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


def _hotspot_window_geometry(env, state, num_hot: int, batch: int = 20):
    """``(T, N, num_hot)`` bool visibility restricted to the hotspot columns (already
    window-gated for a windowed env, since ``_target_geometry`` ANDs it with the window),
    plus the ``(N, num_hot)`` int32 first step (1-indexed, -1 if never) each hotspot target
    becomes visible to each satellite. Kept to the hotspot columns (``num_hot`` far smaller
    than ``max_targets``) so this stays cheap even at 200 satellites x thousands of targets,
    independent of ``lean_precompute``'s own (full-width) geometry pass."""
    import jax
    import numpy as np

    if num_hot == 0:
        return (
            np.zeros((env.time_limit, env.num_agents, 0), bool),
            np.zeros((env.num_agents, 0), np.int32),
        )
    steps = jax.numpy.arange(1, env.time_limit + 1)
    geometry = jax.jit(jax.vmap(lambda s: env._target_geometry(state, s)[1][:, :num_hot]))
    access_hot = np.concatenate(
        [np.asarray(geometry(steps[i : i + batch])) for i in range(0, len(steps), batch)]
    )
    ever = access_hot.any(axis=0)
    first_step = np.where(ever, np.argmax(access_hot, axis=0) + 1, -1).astype(np.int32)
    return access_hot, first_step


def _new_agg() -> Dict[str, Any]:
    return dict(
        return_total=0.0,
        return_hotspot=0.0,
        return_background=0.0,
        honoured_looks=0,
        empty_looks=0,
        hotspot_looks=0,
        background_looks=0,
        duplicate_waste=0.0,
        battery_sum=0.0,
        battery_n=0,
        low_battery_count=0,
        final_battery_mean=0.0,
        cumulative_at=dict.fromkeys(CHECKPOINT_STEPS, None),
        steps_run=0,
        slot_match=dict(slot0=0, slot1_3=0, no_match=0),
        value_bins=[dict(lo=lo, hi=hi, count=0, sense_count=0) for lo, hi in VALUE_BINS],
        # Window-aware diagnostics (all policies; see sarsat.windows).
        background_while_event_visible=0,
        battery_open_sum=0.0,
        battery_open_n=0,
        event_value_lost=0.0,
    )


def _bucket_value(v: float) -> int:
    for i, (lo, hi) in enumerate(VALUE_BINS):
        if i == len(VALUE_BINS) - 1:
            if lo <= v <= hi:
                return i
        elif lo <= v < hi:
            return i
    return -1


def _finalize(
    agg: Dict[str, Any],
    unit: float,
    hotspot_total: float,
    background_total: float,
    sense_cost: float,
    num_agents: int,
) -> Dict[str, Any]:
    honoured = agg["honoured_looks"]
    battery_n = max(agg["battery_n"], 1)
    out = dict(
        return_total=agg["return_total"],
        return_hotspot=agg["return_hotspot"],
        return_background=agg["return_background"],
        hotspot_fraction_imaged=agg["return_hotspot"] / hotspot_total if hotspot_total else None,
        event_fraction_imaged=agg["return_hotspot"] / hotspot_total if hotspot_total else None,
        background_fraction_imaged=(
            agg["return_background"] / background_total if background_total else None
        ),
        honoured_looks=honoured,
        empty_looks=agg["empty_looks"],
        hotspot_looks=agg["hotspot_looks"],
        background_looks=agg["background_looks"],
        mean_captured_priority_per_honoured_look_units=(
            (agg["return_total"] / honoured / unit) if honoured else 0.0
        ),
        duplicate_waste=agg["duplicate_waste"],
        duplicate_waste_units=agg["duplicate_waste"] / unit if unit else 0.0,
        mean_battery=agg["battery_sum"] / battery_n,
        frac_steps_battery_below_sense_cost=agg["low_battery_count"] / battery_n,
        final_mean_battery=agg["final_battery_mean"],
        cumulative_return_at=agg["cumulative_at"],
        steps_run=agg["steps_run"],
        sense_cost=sense_cost,
        # -- window-aware diagnostics --
        # (b) per honoured look, whether it hit an event target inside its window: hitting a
        # hotspot target *is* being inside its window, since visibility itself is gated on
        # the window (sarsat.windows._target_geometry), so this is exactly hotspot_looks.
        event_look_fraction_of_honoured=agg["hotspot_looks"] / honoured if honoured else None,
        # (c) looks spent on background (or empty) while >=1 event target was visible to
        # that satellite -- a look that could have gone to the event instead.
        background_while_event_visible_looks=agg["background_while_event_visible"],
        background_while_event_visible_fraction=(
            agg["background_while_event_visible"] / honoured if honoured else None
        ),
        # (d) mean battery over (satellite, step) pairs where a hotspot target first became
        # visible to that satellite -- battery available right when an event opens for it.
        battery_at_window_open_mean=(
            agg["battery_open_sum"] / agg["battery_open_n"] if agg["battery_open_n"] else None
        ),
        battery_at_window_open_n=agg["battery_open_n"],
        # (e) hotspot priority that was visible to some charged (battery >= sense_cost)
        # satellite at some point during its window, but never imaged.
        event_value_lost_to_missed_windows=agg["event_value_lost"],
        event_value_lost_fraction=(
            agg["event_value_lost"] / hotspot_total if hotspot_total else None
        ),
    )
    total_sensing_diag = sum(agg["slot_match"].values())
    if total_sensing_diag:
        out["slot_match"] = agg["slot_match"]
        out["slot_match_fraction"] = {
            k: v / total_sensing_diag for k, v in agg["slot_match"].items()
        }
    bins_out = []
    for b in agg["value_bins"]:
        rate = b["sense_count"] / b["count"] if b["count"] else None
        bins_out.append(
            dict(
                lo=b["lo"],
                hi=b["hi"],
                count=b["count"],
                sense_count=b["sense_count"],
                sense_rate=rate,
            )
        )
    if any(b["count"] for b in bins_out):
        out["sense_rate_by_slot0_value"] = bins_out
    return out


def run_trained(
    run_dir: str,
    system: str,
    seed: int,
    tol: float,
    scenario_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from omegaconf import OmegaConf

    ec = _load_eval_checkpoint()
    from mapx.evaluator import make_rec_eval_act_fn
    from mapx.networks import ScannedRNN
    from mapx.wrappers.sarsat_coop import make_sarsat_coop_envs

    from sarsat.coop import SLOT_DIM
    from sarsat.orbits import in_az_el_box

    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    model_name = f"rec_{system}"
    cfg = OmegaConf.load(ec.find_hydra_config(run_dir))
    OmegaConf.set_struct(cfg, False)
    checkpoint_dir = ec.find_checkpoint_dir(run_dir, model_name)
    actor_params, ckpt_step = ec.restore_actor_params(checkpoint_dir)

    if scenario_override:
        # Fallback overrides on top of the run's own scenario (e.g. --smoke-test, or an
        # explicit --num-satellites etc.). Safe for the trained policy too: none of these
        # keys affect network shapes (those depend only on num_slots/horizon/hidden_dim,
        # not on satellite/target counts), so the real checkpoint params still apply.
        for key, value in scenario_override.items():
            cfg.env.scenario.task_config[key] = value
            if key in cfg.env.kwargs:
                cfg.env.kwargs[key] = value

    add_global_state = system == "mappo"
    _, eval_env = make_sarsat_coop_envs(cfg, add_global_state=add_global_state)
    coop_env = eval_env._env  # raw CoopSarSat/WindowedCoopSarSat; the wrapper's conversion
    # is irrelevant here.
    num_agents = eval_env.num_agents
    num_slots = coop_env.num_slots
    num_hot = coop_env.hotspots * coop_env.hotspot_targets
    sense_cost = coop_env.sense_cost

    actor_network = ec.build_actor(cfg, eval_env)
    eval_act_fn = make_rec_eval_act_fn(actor_network.apply, cfg)

    def make_step():
        def step(state, action):
            pointing = coop_env._decode_pointing(state.orbit, action[:, : coop_env._pointing_dim])
            sensing = (action[:, coop_env._pointing_dim] > 0.5) & coop_env._can_sense(state.battery)
            lower = (pointing - coop_env._beam_half_width)[:, None, :]
            upper = (pointing + coop_env._beam_half_width)[:, None, :]
            in_beam = state.visible & sensing[:, None] & in_az_el_box(state.body, lower, upper)
            imaged = jnp.any(in_beam, axis=0)
            reward = jnp.sum(jnp.where(imaged, state.target_priority, 0.0))
            battery_pre = state.battery
            new_step = state.step + 1
            new_battery = jnp.clip(
                state.battery + coop_env.recharge_rate - coop_env.sense_cost * sensing, 0.0, 1.0
            )
            mid_state = state._replace(
                step=new_step, battery=new_battery, target_active=state.target_active & ~imaged
            )
            all_imaged = ~jnp.any(mid_state.target_active)
            last = all_imaged | (mid_state.step >= coop_env.time_limit)
            new_state, timestep = coop_env.make_timestep(mid_state, reward, all_imaged, last)
            hits = jnp.sum(in_beam, axis=0)
            return new_state, timestep, sensing, in_beam, imaged, reward, battery_pre, hits, last

        return jax.jit(step)

    step_fn = make_step()

    reset_key = jax.random.PRNGKey(seed)
    state, raw_timestep = coop_env.reset(reset_key)
    priority_np = np.asarray(state.target_priority)
    unit = float(priority_np.max())
    hotspot_total = float(priority_np[:num_hot].sum())
    background_total = float(priority_np[num_hot:].sum())

    # Window-aware diagnostics: hotspot-only visibility over the whole episode, computed
    # once from the post-reset state (geometry never depends on the policy).
    access_hot, first_visible_hot = _hotspot_window_geometry(coop_env, state, num_hot)
    reachable_charged = np.zeros(num_hot, bool)

    hidden_state = ScannedRNN.initialize_carry((1, num_agents), int(cfg.network.hidden_state_dim))
    actor_state = {"hidden_state": hidden_state}
    act_key = jax.random.PRNGKey(seed)

    agg = _new_agg()
    ts = raw_timestep
    cumret = 0.0
    final_battery = priority_np * 0.0  # placeholder, overwritten each step
    last_battery_post = np.asarray(state.battery)

    for t in range(1, coop_env.time_limit + 1):
        agents_view = np.asarray(ts.observation.agents_view)
        batched_timestep = jax.tree_util.tree_map(lambda x: x[jnp.newaxis], ts)
        act_key, sub_key = jax.random.split(act_key)
        action, actor_state = eval_act_fn(actor_params, batched_timestep, sub_key, actor_state)
        action = action[0]
        action_np = np.asarray(action)

        state, ts, sensing, in_beam, imaged, reward, battery_pre, hits, last = step_fn(
            state, action
        )

        sensing_np = np.asarray(sensing)
        in_beam_np = np.asarray(in_beam)
        imaged_np = np.asarray(imaged)
        hits_np = np.asarray(hits)
        battery_pre_np = np.asarray(battery_pre)
        last_battery_post = np.asarray(state.battery)
        reward_f = float(reward)
        cumret += reward_f

        if t in agg["cumulative_at"]:
            agg["cumulative_at"][t] = cumret

        agg["battery_sum"] += float(battery_pre_np.sum())
        agg["battery_n"] += battery_pre_np.size
        agg["low_battery_count"] += int((battery_pre_np < sense_cost).sum())

        agg["return_total"] += reward_f
        agg["return_hotspot"] += float(priority_np[:num_hot][imaged_np[:num_hot]].sum())
        agg["return_background"] += float(priority_np[num_hot:][imaged_np[num_hot:]].sum())
        agg["duplicate_waste"] += float(priority_np[hits_np > 1].sum())

        honoured_t = int(sensing_np.sum())
        agg["honoured_looks"] += honoured_t
        any_hit = in_beam_np.any(axis=1)
        hotspot_hit = in_beam_np[:, :num_hot].any(axis=1)
        sensing_idx = np.flatnonzero(sensing_np)
        for n in sensing_idx:
            if not any_hit[n]:
                agg["empty_looks"] += 1
            elif hotspot_hit[n]:
                agg["hotspot_looks"] += 1
            else:
                agg["background_looks"] += 1

        # ---- window-aware diagnostics (c, d, e) ----
        if num_hot:
            step_access_hot = access_hot[t - 1]  # (N, num_hot), already window-gated
            can_sense_pre = battery_pre_np >= sense_cost
            reachable_charged |= (step_access_hot & can_sense_pre[:, None]).any(axis=0)

            open_sat = (first_visible_hot == t).any(axis=1)  # (N,)
            if open_sat.any():
                agg["battery_open_sum"] += float(battery_pre_np[open_sat].sum())
                agg["battery_open_n"] += int(open_sat.sum())

            event_visible_n = step_access_hot.any(axis=1)  # (N,)
            for n in sensing_idx:
                if event_visible_n[n] and not hotspot_hit[n]:
                    agg["background_while_event_visible"] += 1

        # ---- trained-policy-only diagnostics: slot tracking & sense-rate by slot0 value ----
        slot0_valid = agents_view[:, 0] > 0.5
        val = agents_view[:, 4]
        for n in range(num_agents):
            if not slot0_valid[n]:
                continue
            b = _bucket_value(float(val[n]))
            if b < 0:
                continue
            agg["value_bins"][b]["count"] += 1
            if sensing_np[n]:
                agg["value_bins"][b]["sense_count"] += 1

        own_point = action_np[:, :3]
        for n in sensing_idx:
            best_dist, best_k = np.inf, -1
            for k in range(num_slots):
                base = k * SLOT_DIM
                if agents_view[n, base] <= 0.5:
                    continue
                d = float(np.linalg.norm(own_point[n] - agents_view[n, base + 1 : base + 4]))
                if d < best_dist:
                    best_dist, best_k = d, k
            if best_k < 0 or best_dist >= tol:
                agg["slot_match"]["no_match"] += 1
            elif best_k == 0:
                agg["slot_match"]["slot0"] += 1
            else:
                agg["slot_match"]["slot1_3"] += 1

        agg["steps_run"] = t
        if bool(last):
            break

    agg["final_battery_mean"] = float(last_battery_post.mean())
    if num_hot:
        final_active_hot = np.asarray(state.target_active[:num_hot])
        lost_mask = reachable_charged & final_active_hot
        agg["event_value_lost"] = float(priority_np[:num_hot][lost_mask].sum())
    return (
        _finalize(agg, unit, hotspot_total, background_total, sense_cost, num_agents),
        int(ckpt_step),
        checkpoint_dir,
    )


def run_reference(policy: str, seed: int, scenario_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    import sarsat.reference as reference_module

    # REQUIRED at 200 satellites: the default float64 (T, N, M) tables reach ~16 GB for
    # solo_plan/coop_plan's precompute. Must happen before any ReferenceController that
    # needs it (greedy_beam never calls precompute).
    reference_module.precompute = lean_precompute

    from sarsat.reference import ReferenceController
    from sarsat.windows import WindowedSarSat

    env = WindowedSarSat(**scenario_kwargs)  # == SarSat unless scenario sets window_steps
    num_hot = env.hotspots * env.hotspot_targets
    sense_cost = env.sense_cost
    num_agents = env.num_agents

    def make_step():
        def step(state, action):
            step_i = state.step + 1
            pointing, sensing, in_beam = env._apply_action(state, step_i, action)
            imaged = jnp.any(in_beam, axis=0)
            reward = jnp.sum(jnp.where(imaged, state.target_priority, 0.0))
            battery_pre = state.battery
            new_battery = jnp.clip(
                state.battery + env.recharge_rate - env.sense_cost * sensing, 0.0, 1.0
            )
            new_state = state._replace(
                step=step_i, battery=new_battery, target_active=state.target_active & ~imaged
            )
            all_imaged = ~jnp.any(new_state.target_active)
            last = all_imaged | (step_i >= env.time_limit)
            hits = jnp.sum(in_beam, axis=0)
            return new_state, sensing, in_beam, imaged, reward, battery_pre, hits, last

        return jax.jit(step)

    step_fn = make_step()
    state0, _ = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    priority_np = np.asarray(state0.target_priority)
    unit = float(priority_np.max())
    hotspot_total = float(priority_np[:num_hot].sum())
    background_total = float(priority_np[num_hot:].sum())

    # Window-aware diagnostics: hotspot-only visibility over the whole episode, computed
    # once from the post-reset state (geometry never depends on the policy).
    access_hot, first_visible_hot = _hotspot_window_geometry(env, state0, num_hot)
    reachable_charged = np.zeros(num_hot, bool)

    controller = ReferenceController(env, state0, policy)
    agg = _new_agg()
    state = state0
    last_battery_post = np.asarray(state.battery)
    cumret = 0.0

    for t in range(1, env.time_limit + 1):
        action = controller(state)
        state, sensing, in_beam, imaged, reward, battery_pre, hits, last = step_fn(state, action)

        sensing_np = np.asarray(sensing)
        in_beam_np = np.asarray(in_beam)
        imaged_np = np.asarray(imaged)
        hits_np = np.asarray(hits)
        battery_pre_np = np.asarray(battery_pre)
        last_battery_post = np.asarray(state.battery)
        reward_f = float(reward)
        cumret += reward_f

        if t in agg["cumulative_at"]:
            agg["cumulative_at"][t] = cumret

        agg["battery_sum"] += float(battery_pre_np.sum())
        agg["battery_n"] += battery_pre_np.size
        agg["low_battery_count"] += int((battery_pre_np < sense_cost).sum())

        agg["return_total"] += reward_f
        agg["return_hotspot"] += float(priority_np[:num_hot][imaged_np[:num_hot]].sum())
        agg["return_background"] += float(priority_np[num_hot:][imaged_np[num_hot:]].sum())
        agg["duplicate_waste"] += float(priority_np[hits_np > 1].sum())

        any_hit = in_beam_np.any(axis=1)
        hotspot_hit = in_beam_np[:, :num_hot].any(axis=1)
        sensing_idx = np.flatnonzero(sensing_np)
        agg["honoured_looks"] += len(sensing_idx)
        for n in sensing_idx:
            if not any_hit[n]:
                agg["empty_looks"] += 1
            elif hotspot_hit[n]:
                agg["hotspot_looks"] += 1
            else:
                agg["background_looks"] += 1

        # ---- window-aware diagnostics (c, d, e) ----
        if num_hot:
            step_access_hot = access_hot[t - 1]  # (N, num_hot), already window-gated
            can_sense_pre = battery_pre_np >= sense_cost
            reachable_charged |= (step_access_hot & can_sense_pre[:, None]).any(axis=0)

            open_sat = (first_visible_hot == t).any(axis=1)  # (N,)
            if open_sat.any():
                agg["battery_open_sum"] += float(battery_pre_np[open_sat].sum())
                agg["battery_open_n"] += int(open_sat.sum())

            event_visible_n = step_access_hot.any(axis=1)  # (N,)
            for n in sensing_idx:
                if event_visible_n[n] and not hotspot_hit[n]:
                    agg["background_while_event_visible"] += 1

        agg["steps_run"] = t
        if bool(last):
            break

    agg["final_battery_mean"] = float(last_battery_post.mean())
    if num_hot:
        final_active_hot = np.asarray(state.target_active[:num_hot])
        lost_mask = reachable_charged & final_active_hot
        agg["event_value_lost"] = float(priority_np[:num_hot][lost_mask].sum())
    return _finalize(agg, unit, hotspot_total, background_total, sense_cost, num_agents)


def run_seed(args) -> Dict[str, Any]:
    run_dir, system, seed, tol, reference_kwargs, trained_override = args
    t0 = time.perf_counter()
    trained, ckpt_step, checkpoint_dir = run_trained(run_dir, system, seed, tol, trained_override)
    t1 = time.perf_counter()
    print(f"[seed {seed}] trained policy done in {t1 - t0:.0f}s", flush=True)
    greedy_beam = run_reference("greedy_beam", seed, reference_kwargs)
    t2 = time.perf_counter()
    print(f"[seed {seed}] greedy_beam done in {t2 - t1:.0f}s", flush=True)
    coop_plan = run_reference("coop_plan", seed, reference_kwargs)
    t3 = time.perf_counter()
    print(f"[seed {seed}] coop_plan done in {t3 - t2:.0f}s", flush=True)
    return dict(
        seed=seed,
        checkpoint_step=ckpt_step,
        checkpoint_dir=checkpoint_dir,
        trained=trained,
        greedy_beam=greedy_beam,
        coop_plan=coop_plan,
    )


POLICY_NAMES = ("trained", "greedy_beam", "coop_plan")


def _fmt(x: Optional[float], spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def print_table(rows: List[Dict[str, Any]], title: str) -> None:
    header = (
        f"{'seed':>6} {'policy':>12} {'return':>8} {'hot%':>7} {'bg%':>7} "
        f"{'honoured':>9} {'empty':>6} {'hotL':>6} {'bgL':>6} {'val/look':>9} "
        f"{'meanBat':>8} {'lowBat%':>8} {'finBat':>7} {'dupWaste':>9}"
    )
    print(f"\n== {title} ==")
    print(header)
    print("-" * len(header))
    for row in rows:
        seed = row["seed"]
        for pol in POLICY_NAMES:
            d = row[pol]
            print(
                f"{seed:>6} {pol:>12} {_fmt(d['return_total'])} "
                f"{_fmt(100 * d['hotspot_fraction_imaged'] if d['hotspot_fraction_imaged'] is not None else None, '.1f'):>6}% "
                f"{_fmt(100 * d['background_fraction_imaged'] if d['background_fraction_imaged'] is not None else None, '.1f'):>6}% "
                f"{d['honoured_looks']:>9} {d['empty_looks']:>6} {d['hotspot_looks']:>6} "
                f"{d['background_looks']:>6} {_fmt(d['mean_captured_priority_per_honoured_look_units'], '.3f'):>9} "
                f"{_fmt(d['mean_battery'], '.3f'):>8} "
                f"{_fmt(100 * d['frac_steps_battery_below_sense_cost'], '.1f'):>7}% "
                f"{_fmt(d['final_mean_battery'], '.3f'):>7} {_fmt(d['duplicate_waste_units'], '.3f'):>9}"
            )


def print_window_table(rows: List[Dict[str, Any]], title: str) -> None:
    header = (
        f"{'seed':>6} {'policy':>12} {'evtLookF':>9} {'bgWhileEvt':>11} "
        f"{'battOpen':>9} {'evtLostF':>9} {'evtLost':>8}"
    )
    print(f"\n== {title} ==")
    print(header)
    print("-" * len(header))
    for row in rows:
        seed = row["seed"]
        for pol in POLICY_NAMES:
            d = row[pol]
            print(
                f"{seed:>6} {pol:>12} "
                f"{_fmt(100 * d['event_look_fraction_of_honoured'] if d['event_look_fraction_of_honoured'] is not None else None, '.1f'):>8}% "
                f"{d['background_while_event_visible_looks']:>11} "
                f"{_fmt(d['battery_at_window_open_mean'], '.3f'):>9} "
                f"{_fmt(100 * d['event_value_lost_fraction'] if d['event_value_lost_fraction'] is not None else None, '.1f'):>8}% "
                f"{_fmt(d['event_value_lost_to_missed_windows'], '.4f'):>8}"
            )


def average_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average every numeric field across seeds, per policy."""
    numeric_keys = [
        "return_total",
        "return_hotspot",
        "return_background",
        "hotspot_fraction_imaged",
        "event_fraction_imaged",
        "background_fraction_imaged",
        "honoured_looks",
        "empty_looks",
        "hotspot_looks",
        "background_looks",
        "mean_captured_priority_per_honoured_look_units",
        "duplicate_waste",
        "duplicate_waste_units",
        "mean_battery",
        "frac_steps_battery_below_sense_cost",
        "final_mean_battery",
        "event_look_fraction_of_honoured",
        "background_while_event_visible_looks",
        "background_while_event_visible_fraction",
        "battery_at_window_open_mean",
        "event_value_lost_to_missed_windows",
        "event_value_lost_fraction",
    ]
    out = {"seed": "avg"}
    for pol in POLICY_NAMES:
        vals = [r[pol] for r in rows]
        avg = {}
        for k in numeric_keys:
            xs = [v[k] for v in vals if v.get(k) is not None]
            avg[k] = sum(xs) / len(xs) if xs else None
        cum_keys = vals[0]["cumulative_return_at"].keys()
        avg["cumulative_return_at"] = {}
        for ck in cum_keys:
            xs = [
                v["cumulative_return_at"][ck]
                for v in vals
                if v["cumulative_return_at"].get(ck) is not None
            ]
            avg["cumulative_return_at"][ck] = sum(xs) / len(xs) if xs else None
        out[pol] = avg
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="~/marl/runs/mappo_v1")
    parser.add_argument("--system", default="mappo", choices=["mappo", "ippo"])
    parser.add_argument("--seeds", default="1000,1003,1009")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument(
        "--num-satellites",
        type=int,
        default=None,
        help="Fallback override for the run's scenario num_satellites (default: read from "
        "the run's hydra config, env.kwargs merged with env.scenario.task_config).",
    )
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--hotspots", type=int, default=None)
    parser.add_argument("--hotspot-targets", type=int, default=None)
    parser.add_argument("--hotspot-weight", type=float, default=None)
    parser.add_argument("--planes", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Shrink the scenario (also for the trained policy) to sanity-check the "
        "pipeline in seconds instead of minutes. Not for real diagnosis.",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    out_path = (
        os.path.abspath(os.path.expanduser(args.out))
        if args.out
        else os.path.join(run_dir, "diagnosis.json")
    )

    from omegaconf import OmegaConf

    ec = _load_eval_checkpoint()
    hydra_config_path = ec.find_hydra_config(run_dir)
    cfg = OmegaConf.load(hydra_config_path)
    OmegaConf.set_struct(cfg, False)
    merged_kwargs = _merge_scenario_kwargs(cfg)

    cli_override: Dict[str, Any] = {}
    for flag, key in (
        ("num_satellites", "num_satellites"),
        ("max_targets", "max_targets"),
        ("hotspots", "hotspots"),
        ("hotspot_targets", "hotspot_targets"),
        ("hotspot_weight", "hotspot_weight"),
        ("planes", "planes"),
        ("time_limit", "time_limit"),
    ):
        value = getattr(args, flag)
        if value is not None:
            cli_override[key] = value
    if args.smoke_test:
        cli_override = {**SMOKE_OVERRIDE, **cli_override}  # explicit flags still win

    # Applied identically to trained and reference: matched episodes require the same
    # scenario for both (see module docstring), so there is no valid mode in which they'd
    # deliberately differ.
    reference_kwargs = _sarsat_only_kwargs({**merged_kwargs, **cli_override})
    trained_override = cli_override or None

    tasks = [
        (run_dir, args.system, seed, args.tol, reference_kwargs, trained_override)
        for seed in seeds
    ]
    workers = max(1, min(args.workers, len(seeds), 3))
    print(f"Running {len(seeds)} seed(s) across {workers} worker process(es)...", flush=True)

    t0 = time.perf_counter()
    if workers == 1:
        results = [run_seed(t) for t in tasks]
    else:
        ctx = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            results = list(ex.map(run_seed, tasks))
    results.sort(key=lambda r: r["seed"])
    print(f"All seeds done in {time.perf_counter() - t0:.0f}s", flush=True)

    print_table(results, "per-seed")
    avg = average_rows(results)
    print_table([avg], "averaged across seeds")

    print_window_table(results, "window-aware diagnostics, per-seed")
    print_window_table([avg], "window-aware diagnostics, averaged across seeds")

    print("\n== trained-policy pointing diagnostics (averaged) ==")
    for r in results:
        sm = r["trained"].get("slot_match_fraction")
        if sm:
            print(
                f"seed {r['seed']}: slot0={sm.get('slot0', 0):.1%} "
                f"slot1-3={sm.get('slot1_3', 0):.1%} no_match={sm.get('no_match', 0):.1%}"
            )
        bins = r["trained"].get("sense_rate_by_slot0_value")
        if bins:
            parts = [
                f"[{b['lo']:.2f},{b['hi']:.2f}]:{_fmt(b['sense_rate'], '.1%') if b['sense_rate'] is not None else 'n/a'} (n={b['count']})"
                for b in bins
            ]
            print("  sense rate by slot0 value: " + "  ".join(parts))

    payload = dict(
        run_dir=run_dir,
        system=args.system,
        seeds=seeds,
        hydra_config=hydra_config_path,
        scenario_kwargs=reference_kwargs,
        cli_override=cli_override,
        tolerance=args.tol,
        results=results,
        averaged=avg,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote diagnosis to {out_path}")


if __name__ == "__main__":
    main()
