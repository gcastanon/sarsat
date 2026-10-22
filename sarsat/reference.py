"""Reference policies that act from the full environment state.

These are yardsticks for what coordination is worth (DECISIONS.md issue 22), not
learners: the centralised ones see every satellite's future access. All four share one
interface, :class:`ReferenceController`, which maps a pre-step ``State`` to the action
executed at the next step, so ``sarsat.evaluate.run_episode`` can log them like any
observation-based policy.

* ``greedy_beam`` -- independent, beam-aware: aim where the beam captures the most priority
                    and sense whenever the battery allows. The fair independent baseline.
* ``solo_plan``  -- independent, far-sighted: weighs a target by how rarely *it alone* will
                    see it again and rations looks to its budget.
* ``coop_dedup`` -- centralised: no target is taken twice in one step.
* ``coop_plan``  -- centralised: dedup, scarcity weighted by *every* satellite's future
                    access, and the look budget. The cooperative reference.

With collection modes (``modes`` given, DECISIONS.md issue 33) a look is a target *and* a
mode: every policy weighs each mode the battery can pay for with that mode's footprint,
band, squint limit and quality, and takes the best value per unit of battery (value times
the cheapest mode's cost over the mode's). The planners' look budget is counted in looks of
the cheapest mode, their future-opportunity tables count only access some mode can serve,
and a target's future value is scaled by the cost of the cheapest mode good enough for it.
With one mode all of this reduces to the single-mode policies exactly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.env import SarSat
from sarsat.orbits import az_el
from sarsat.types import State

REFERENCE_POLICIES = ("greedy_beam", "solo_plan", "coop_dedup", "coop_plan")


def beam_values(az, el, weight, half_width, exclude=None):
    """Value of aiming at each accessible candidate: summed weight of the targets the beam
    would then contain. All inputs are for one satellite's accessible set."""
    hit = (np.abs(az[:, None] - az[None, :]) <= half_width[0]) & (
        np.abs(el[:, None] - el[None, :]) <= half_width[1]
    )
    return hit @ (weight if exclude is None else np.where(exclude, 0.0, weight)), hit


class ModeOption(NamedTuple):
    """One collection mode as :func:`choose` weighs it at one step."""

    serve: np.ndarray  # (N, M) bool: targets a look in this mode could image
    half_width: np.ndarray  # (2,) beam half-widths [rad]
    scale: float  # the cheapest mode's cost over this one's: value per unit of battery
    afford: np.ndarray | None  # (N,) bool: battery pays for a look, or None for "always"


def access_fn(env, state):
    """``step -> (N, M)`` access some mode can serve: :meth:`SarSat._target_geometry`'s
    visibility, narrowed to each mode's band, squint limit and quality when they matter."""
    if not env._mode_flags:
        return lambda step: env._target_geometry(state, step)[1]

    def access(step):
        body, visible = env._target_geometry(state, step)
        return visible & jnp.any(mode_serve(env, state, body), axis=0)

    return access


def mode_serve(env, state, body) -> jax.Array:
    """``(K, N, M)`` whether a look in each mode reaches and is good enough for each target
    (the envelope's visibility is not included)."""
    n, m = body.shape[:2]
    serve = jnp.ones((env.num_modes, n, m), bool)
    if env._mode_bands_differ:
        radius = state.orbit.radius
        serve = jnp.stack([env._in_mode_band(radius, body, k) for k in range(env.num_modes)])
    required = env._required_quality(state)
    if required is not None:
        serve = serve & (env._mode_quality[:, None, None] >= required[None, None, :])
    return serve


def value_scale(env, state) -> np.ndarray | float:
    """Per target, the cheapest mode's cost over that of the cheapest mode good enough for
    it (0 if none is), or ``1.0`` without modes: what a look at it is worth per unit of
    battery, relative to its priority."""
    if not env._mode_flags:
        return 1.0
    cost = np.asarray(env._mode_cost, np.float64)
    required = env._required_quality(state)
    if required is None:
        return 1.0  # the cheapest mode is good enough for every target
    good = np.asarray(env._mode_quality)[:, None] >= np.asarray(required)[None, :]  # (K, M)
    cheapest = np.where(good, cost[:, None], np.inf).min(axis=0)
    return np.where(np.isfinite(cheapest), cost.min() / cheapest, 0.0)


def precompute(env, state):
    """Future access (T, N, M); per-target opportunity counts from each step on, for the
    whole constellation (T, M) and per satellite (T, N, M); and each satellite's best
    scarcity-weighted value at every step, both flavours (T, N)."""
    steps = jnp.arange(1, env.time_limit + 1)
    geometry = jax.jit(jax.vmap(access_fn(env, state)))
    access = np.concatenate(
        [np.asarray(geometry(steps[i : i + 20])) for i in range(0, len(steps), 20)]
    )
    prio = np.asarray(state.target_priority) * value_scale(env, state)
    team = np.cumsum(access.sum(1)[::-1], 0)[::-1]  # (T, M)
    own = np.cumsum(access[::-1], 0)[::-1]  # (T, N, M)
    best_team = np.einsum("tnm,tm->tnm", access, prio[None] / np.maximum(team, 1.0)).max(2)
    best_own = np.where(access, prio[None, None] / np.maximum(own, 1.0), 0.0).max(2)
    return dict(access=access, team=team, own=own, best_team=best_team, best_own=best_own)


def lean_precompute(env, state, pair_budget: int = 40_000_000):
    """:func:`precompute` in the smallest dtypes that hold the tables (bool access, byte
    opportunity counts, float32 values), built one step at a time.

    The default tables are float64 / int64 and reach tens of GB at 200 satellites x 8000
    targets; these reach 4 GB at 500 satellites x 20,000 targets. The geometry is evaluated
    in chunks of steps sized so that a chunk holds about ``pair_budget`` satellite-target
    pairs. Values are the same as the scripts' earlier int32 / float32 tables to the last
    bit (the counts are exact and the divisions are done in the same precision); against
    the full-precision tables ``coop_plan`` moves by about 2e-4 at 100 satellites.
    """
    n, m, t_max = env.num_agents, env.max_targets, env.time_limit
    steps = jnp.arange(1, t_max + 1)
    chunk = max(1, int(pair_budget // (n * m)))
    geometry = jax.jit(jax.vmap(access_fn(env, state)))
    access = np.concatenate(
        [np.asarray(geometry(steps[i : i + chunk])) for i in range(0, t_max, chunk)]
    )  # (T, N, M) bool
    prio = (np.asarray(state.target_priority) * value_scale(env, state)).astype(np.float32)
    team = np.cumsum(access.sum(1, dtype=np.int32)[::-1], 0, dtype=np.int32)[::-1]  # (T, M)
    count = np.uint8 if t_max < 256 else np.uint16
    own = np.empty(access.shape, count)  # (T, N, M) opportunities left, per satellite
    best_team = np.empty((t_max, n), np.float32)
    best_own = np.empty((t_max, n), np.float64)
    for t in range(t_max - 1, -1, -1):
        own[t] = access[t] if t == t_max - 1 else own[t + 1] + access[t]
        team_weight = prio / np.maximum(team[t], 1).astype(np.float32)
        best_team[t] = (access[t] * team_weight[None]).max(1)
        own_count = np.maximum(own[t], 1).astype(np.int32)  # float64 division, as before
        best_own[t] = np.where(access[t], prio[None] / own_count, 0.0).max(1)
    return dict(access=access, team=team, own=own, best_team=best_team, best_own=best_own)


def choose(policy, t, vis, az, el, prio, hw, battery, pre, env, modes=None):
    """Target index each satellite aims at (-1: no look) under one of the four policies,
    and the index into ``modes`` (a list of :class:`ModeOption`) of the mode it looks in.
    Without ``modes`` there is one: the beam ``hw`` over the targets in ``vis``."""
    n_sat, remaining = vis.shape[0], env.time_limit - t
    choice, mode = np.full(n_sat, -1), np.zeros(n_sat, int)
    if modes is None:
        modes = [ModeOption(vis, hw, 1.0, None)]
    # Sensing budget for the rest of the episode: what the current charge buys outright,
    # plus what the sustained duty cycle (recharge / cost) adds over the steps left.
    looks_left = np.minimum(
        np.floor(battery / (env.sense_cost - env.recharge_rate) + 1e-6)
        + env.recharge_rate / env.sense_cost * remaining,
        remaining + 1,
    )
    planning = policy in ("solo_plan", "coop_plan")
    if policy == "coop_plan":
        weight = prio / np.maximum(pre["team"][t - 1], 1.0)
    else:
        weight = prio

    def threshold(n):
        """Value of satellite ``n``'s ``looks_left``-th best future opportunity: a look
        worth less than this now should wait for a better one later."""
        best = pre["best_team" if policy == "coop_plan" else "best_own"][t - 1 :, n]
        k = int(looks_left[n])
        return np.sort(best)[::-1][k - 1] if 0 < k <= len(best) else 0.0

    # Per satellite, the (mode, candidate targets) it could look at.
    cands = {
        n: [(k, np.flatnonzero(o.serve[n])) for k, o in enumerate(modes)
            if (o.afford is None or o.afford[n]) and o.serve[n].any()]
        for n in range(n_sat)
    }
    cands = {n: c for n, c in cands.items() if c}
    if policy.startswith("coop"):
        # Most valuable opportunity chooses first; beams already claimed count for nothing.
        first = {
            n: max(
                beam_values(az[n, i], el[n, i], weight[i], modes[k].half_width)[0].max()
                * modes[k].scale
                for k, i in c
            )
            for n, c in cands.items()
        }
        order = sorted(cands, key=lambda n: -first[n])
    else:
        order = list(cands)
    taken = np.zeros(vis.shape[1], bool)
    for n in order:
        best_value, best = 0.0, None
        for k, idx in cands[n]:
            if policy == "solo_plan":
                w = prio[idx] / np.maximum(pre["own"][t - 1, n, idx], 1.0)
            else:
                w = weight[idx]
            value, hit = beam_values(
                az[n, idx],
                el[n, idx],
                w,
                modes[k].half_width,
                exclude=taken[idx] if policy.startswith("coop") else None,
            )
            value = value * modes[k].scale  # per unit of battery; exact for one mode
            i = value.argmax()
            if best is None or value[i] > best_value:
                best_value, best = value[i], (k, idx, hit, i)
        if best is None or best_value <= 0:
            continue
        # A near-full battery wastes recharge if it idles; otherwise skip if clearly better
        # opportunities lie ahead (future values are optimistic, hence the 0.5).
        if planning and not (battery[n] >= 0.9 or best_value >= 0.5 * threshold(n)):
            continue
        k, idx, hit, i = best
        choice[n], mode[n] = idx[i], k
        if policy.startswith("coop"):
            taken[idx[hit[i]]] = True
    return choice, mode


class ReferenceController:
    """Callable ``state -> action`` for one of :data:`REFERENCE_POLICIES`.

    Precomputes the constellation's future access for ``state0`` once (the planners need
    it), so a controller is bound to one episode: build a new one per ``reset``.
    """

    def __init__(self, env: SarSat, state0: State, policy: str) -> None:
        if policy not in REFERENCE_POLICIES:
            raise ValueError(f"policy must be one of {REFERENCE_POLICIES}")
        self.env, self.policy = env, policy
        self._geometry = jax.jit(env._target_geometry)
        self._prio = np.asarray(state0.target_priority)
        self._hw = np.asarray(env._beam_half_width)
        self._pre = precompute(env, state0) if policy != "greedy_beam" else None

    def __call__(self, state: State) -> jax.Array:
        """Action for the step about to be taken from ``state``."""
        t = int(state.step) + 1
        body, vis = self._geometry(state, jnp.int32(t))
        body, vis = np.asarray(body), np.asarray(vis)
        az, el = (np.asarray(x) for x in az_el(jnp.asarray(body)))
        modes = None
        if self.env._mode_flags:
            serve = np.asarray(mode_serve(self.env, state, jnp.asarray(body))) & vis[None]
            battery = np.asarray(state.battery)
            cost = np.asarray(self.env._mode_cost)
            modes = [
                ModeOption(
                    serve[k],
                    np.asarray(self.env._mode_beam[k]),
                    float(self.env.sense_cost / cost[k]),
                    battery >= cost[k] - 1e-6,
                )
                for k in range(self.env.num_modes)
            ]
        choice, mode = choose(
            self.policy,
            t,
            vis,
            az,
            el,
            self._prio,
            self._hw,
            np.asarray(state.battery),
            self._pre,
            self.env,
            modes,
        )
        sensing = choice >= 0
        n = self.env.num_agents
        aim_az, aim_el = az_el(jnp.asarray(body[np.arange(n), np.where(sensing, choice, 0)]))
        pointing = self.env._encode_pointing(state.orbit, aim_az[:, None], aim_el[:, None])[:, 0]
        look = np.where(sensing, mode + 1, 0)  # the look slot: 0 none, k + 1 mode k
        return jnp.concatenate([pointing, jnp.asarray(look, jnp.float32)[:, None]], axis=-1)
