"""What a policy will collect from here on, if nothing else changes.

Inside an episode the environment's transition is deterministic (randomness enters only at
``reset``), and a greedy policy is a function of its observation and its own carried state,
so playing the policy forward from the current state to the end of the episode gives an
*exact* forecast of the rest of the episode: which target slot is imaged at which step by
which satellite, in which mode. :func:`forecast` does that for any ``SarSat`` family
environment and any policy ``sarsat.evaluate.run_episode`` accepts; :func:`diff` compares two
forecasts (the world as it is against the world with a provisional change) slot by slot.

The policy is left as it was found. A recurrent actor carries a hidden state across steps,
so running it forward would move that state; a policy that owns state therefore exposes
``snapshot() -> s`` and ``restore(s)`` (duck-typed, as ``reset`` is in
:class:`sarsat.evaluate.StatefulPolicy`), and the forecast restores it on the way out. A
policy with ``reset`` but no ``snapshot`` is refused rather than silently disturbed. A plain
function ``observation -> action`` has nothing to preserve, and a reference policy named by
its string is rebuilt from the current state for the forecast alone.

Compute: one un-jitted policy call plus one jitted environment step per forecast step, so a
forecast costs about what the same number of live steps costs (``LiveWorld`` measures 83 ms
per 500-satellite step on a CPU; here without the drawing geometry). DECISIONS.md issue 40.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.env import SarSat
from sarsat.evaluate import Policy

# A capture as the forecast records it: (in-episode step, satellite, mode index or -1).
Capture = Tuple[int, int, int]


@dataclass
class Forecast:
    """The rest of an episode under one policy, from the state it was taken from.

    ``captures`` lists every (step, target slot, satellite) in which a beam imaged a target
    that was still active; ``mode`` is the mode of each capture (-1 without modes). A slot
    appears once at most, since a target goes inactive when imaged. Steps are in-episode
    steps (``state.step`` after the transition, as ``LiveWorld.StepResult.step``).
    """

    start_step: int  # ``state.step`` the forecast starts from; its first step is start + 1
    steps: np.ndarray  # (H,) int, the in-episode steps forecast
    sensing: np.ndarray  # (H, N) bool
    mode: np.ndarray  # (H, N) int, mode of each look, -1 when not sensing
    captures: np.ndarray  # (n, 3) int: step, target slot, satellite
    capture_mode: np.ndarray  # (n,) int
    reward: np.ndarray  # (H,) float, the environment's team reward per step
    battery: np.ndarray  # (H, N) after each step
    seconds: float = 0.0  # wall time the forecast took
    _first: Optional[Dict[int, Capture]] = field(default=None, repr=False, compare=False)

    @property
    def horizon(self) -> int:
        return len(self.steps)

    @property
    def end_step(self) -> int:
        return int(self.steps[-1]) if len(self.steps) else self.start_step

    @property
    def battery_end(self) -> np.ndarray:
        """``(N,)`` charge at the end of the forecast."""
        return self.battery[-1] if len(self.battery) else np.zeros((0,))

    def first_capture(self) -> Dict[int, Capture]:
        """Target slot -> (step, satellite, mode) of its (first) capture."""
        if self._first is None:
            first: Dict[int, Capture] = {}
            for (step, slot, sat), mode in zip(self.captures, self.capture_mode, strict=True):
                first.setdefault(int(slot), (int(step), int(sat), int(mode)))
            self._first = first
        return self._first

    @property
    def collected(self) -> np.ndarray:
        """The target slots imaged within the forecast, sorted."""
        return np.array(sorted(self.first_capture()), dtype=int)


@dataclass
class ForecastDiff:
    """Slot-level difference between a base forecast and an alternative one.

    ``lost``: collected in the base, not in the alternative. ``gained``: the reverse.
    ``shifted``: collected in both at a different step, by a different satellite or in a
    different mode (slot -> (base capture, alternative capture)). ``same``: collected in
    both identically. The set of collected slots is unchanged iff ``lost`` and ``gained``
    are empty; ``shifted`` is what to flag when it is.
    """

    lost: Dict[int, Capture]
    gained: Dict[int, Capture]
    shifted: Dict[int, Tuple[Capture, Capture]]
    same: Dict[int, Capture]

    @property
    def same_set(self) -> bool:
        return not self.lost and not self.gained


def diff(base: Forecast, alt: Forecast) -> ForecastDiff:
    """Compare what two forecasts collect, slot by slot."""
    a, b = base.first_capture(), alt.first_capture()
    lost = {s: c for s, c in a.items() if s not in b}
    gained = {s: c for s, c in b.items() if s not in a}
    shifted, same = {}, {}
    for s in a.keys() & b.keys():
        if a[s] == b[s]:
            same[s] = a[s]
        else:
            shifted[s] = (a[s], b[s])
    return ForecastDiff(lost=lost, gained=gained, shifted=shifted, same=same)


# ----------------------------------------------------------------------- policies


@contextlib.contextmanager
def preserved(policy: Any) -> Iterator[None]:
    """Run a block that steps ``policy`` and leave the policy as it was.

    A policy that owns state (it has ``reset``) must expose ``snapshot()`` / ``restore()``;
    anything else is stateless and passes through."""
    if hasattr(policy, "snapshot") and hasattr(policy, "restore"):
        saved = policy.snapshot()
        try:
            yield
        finally:
            policy.restore(saved)
    elif hasattr(policy, "reset"):
        raise TypeError(
            "a stateful policy must provide snapshot() and restore() to be forecast without "
            "disturbing it"
        )
    else:
        yield


def act_fn(env: SarSat, state: Any, policy: Policy) -> Callable[[Any, Any], jax.Array]:
    """``(state, observation) -> action`` for ``policy``, as ``run_episode`` builds it: a
    reference policy's name becomes a controller planned from ``state`` (the forecast's
    start, so its access tables see the field as it is now), a stateful policy is called
    on the observation, and a plain function is jitted."""
    if isinstance(policy, str):
        from sarsat.reference import ReferenceController

        controller = ReferenceController(env, state, policy)
        return lambda state, obs: controller(state)
    if hasattr(policy, "reset"):
        return lambda state, obs: policy(obs)
    jitted = jax.jit(lambda obs: policy(obs))
    return lambda state, obs: jitted(obs)


# ----------------------------------------------------------------------- forecasting


def advance_fn(env: SarSat):
    """Jitted ``(state, action) -> (state, observation, reward, sensing, mode, in_beam)``,
    on the geometry cache when the state carries one (the coop family), else recomputed."""

    def advance(state, action):
        if hasattr(state, "body"):
            body, visible = state.body, state.visible
        else:
            body, visible = env._target_geometry(state, state.step + 1)
        _, mode, sensing, in_beam = env.resolve_look(state, action, body, visible)
        state, timestep = env.step(state, action)
        reward = jnp.mean(jnp.asarray(timestep.reward))
        return state, timestep.observation, reward, sensing, mode, in_beam

    return jax.jit(advance)


def forecast(
    env: SarSat,
    state: Any,
    obs: Any,
    policy: Policy,
    horizon: Optional[int] = None,
    advance: Optional[Callable] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    progress_every: int = 10,
) -> Forecast:
    """Play ``policy`` forward from ``state`` / ``obs`` to the end of the episode (or for
    ``horizon`` steps, whichever comes first) and record what it collects.

    ``state`` and ``obs`` are not modified (JAX states are immutable) and the policy is left
    as found (:func:`preserved`). ``advance`` lets a caller reuse one compiled step function
    across forecasts (:func:`advance_fn`); by default one is compiled per call. ``progress``,
    if given, is called every ``progress_every`` steps and at the end with ``(steps done,
    steps to do)``, for a caller that shows how far a long forecast has got."""
    t0 = time.perf_counter()
    advance = advance or advance_fn(env)
    act = act_fn(env, state, policy)
    start = int(state.step)
    limit = int(env.time_limit)
    remaining = max(0, limit - start)
    if horizon is not None:
        remaining = min(remaining, int(horizon))

    steps, sensing, modes, captures, cmodes, rewards, batteries = [], [], [], [], [], [], []
    with preserved(policy):
        for i in range(remaining):
            if progress is not None and i and i % progress_every == 0:
                progress(i, remaining)
            action = act(state, obs)
            state, obs, reward, sense, mode, in_beam = advance(state, action)
            step = int(state.step)
            sense, mode, in_beam = np.asarray(sense), np.asarray(mode), np.asarray(in_beam)
            steps.append(step)
            sensing.append(sense)
            modes.append(np.where(sense, mode, -1))
            rewards.append(float(reward))
            batteries.append(np.asarray(state.battery))
            sats, slots = np.nonzero(in_beam)
            if sats.size:
                captures.append(np.stack([np.full_like(slots, step), slots, sats], axis=1))
                cmodes.append(mode[sats])
    if progress is not None:
        progress(remaining, remaining)
    n = int(env.num_agents)
    return Forecast(
        start_step=start,
        steps=np.asarray(steps, int),
        sensing=np.stack(sensing) if sensing else np.zeros((0, n), bool),
        mode=np.stack(modes) if modes else np.zeros((0, n), int),
        captures=np.concatenate(captures) if captures else np.zeros((0, 3), int),
        capture_mode=np.concatenate(cmodes) if cmodes else np.zeros((0,), int),
        reward=np.asarray(rewards, float),
        battery=np.stack(batteries) if batteries else np.zeros((0, n)),
        seconds=time.perf_counter() - t0,
    )
