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
"""

from __future__ import annotations

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


def precompute(env, state):
    """Future access (T, N, M); per-target opportunity counts from each step on, for the
    whole constellation (T, M) and per satellite (T, N, M); and each satellite's best
    scarcity-weighted value at every step, both flavours (T, N)."""
    steps = jnp.arange(1, env.time_limit + 1)
    geometry = jax.jit(jax.vmap(lambda s: env._target_geometry(state, s)[1]))
    access = np.concatenate(
        [np.asarray(geometry(steps[i : i + 20])) for i in range(0, len(steps), 20)]
    )
    prio = np.asarray(state.target_priority)
    team = np.cumsum(access.sum(1)[::-1], 0)[::-1]  # (T, M)
    own = np.cumsum(access[::-1], 0)[::-1]  # (T, N, M)
    best_team = np.einsum("tnm,tm->tnm", access, prio[None] / np.maximum(team, 1.0)).max(2)
    best_own = np.where(access, prio[None, None] / np.maximum(own, 1.0), 0.0).max(2)
    return dict(access=access, team=team, own=own, best_team=best_team, best_own=best_own)


def choose(policy, t, vis, az, el, prio, hw, battery, pre, env):
    """Target index each satellite aims at (-1: no look) under one of the four policies."""
    n_sat, remaining = vis.shape[0], env.time_limit - t
    choice = np.full(n_sat, -1)
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
        best = pre["best_team" if policy == "coop_plan" else "best_own"][t - 1 :, n]
        k = int(looks_left[n])
        return np.sort(best)[::-1][k - 1] if 0 < k <= len(best) else 0.0

    cands = {n: np.flatnonzero(vis[n]) for n in range(n_sat) if vis[n].any()}
    if policy.startswith("coop"):
        # Most valuable opportunity chooses first; beams already claimed count for nothing.
        first = {
            n: beam_values(az[n, i], el[n, i], weight[i], hw)[0].max() for n, i in cands.items()
        }
        order = sorted(cands, key=lambda n: -first[n])
    else:
        order = list(cands)
    taken = np.zeros(vis.shape[1], bool)
    for n in order:
        idx = cands[n]
        if policy == "solo_plan":
            w = prio[idx] / np.maximum(pre["own"][t - 1, n, idx], 1.0)
        else:
            w = weight[idx]
        value, hit = beam_values(
            az[n, idx], el[n, idx], w, hw, exclude=taken[idx] if policy.startswith("coop") else None
        )
        best = value.argmax()
        if value[best] <= 0:
            continue
        # A near-full battery wastes recharge if it idles; otherwise skip if clearly better
        # opportunities lie ahead (future values are optimistic, hence the 0.5).
        if planning and not (battery[n] >= 0.9 or value[best] >= 0.5 * threshold(n)):
            continue
        choice[n] = idx[best]
        if policy.startswith("coop"):
            taken[idx[hit[best]]] = True
    return choice


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
        choice = choose(
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
        )
        sensing = choice >= 0
        n = self.env.num_agents
        aim_az, aim_el = az_el(jnp.asarray(body[np.arange(n), np.where(sensing, choice, 0)]))
        pointing = self.env._encode_pointing(state.orbit, aim_az[:, None], aim_el[:, None])[:, 0]
        return jnp.concatenate([pointing, jnp.asarray(sensing, jnp.float32)[:, None]], axis=-1)
