"""Time-windowed targets: a target pays only while its window is open.

Two subclasses that leave their parents' behaviour intact:

* :class:`WindowedSarSat` -- :class:`sarsat.env.SarSat` whose state carries a
  ``target_window`` ``(M, 2)`` of first and last step at which each target can be imaged.
  Every hotspot cluster draws one window (an event all of its targets share) and every
  background target its own; lengths are uniform in ``window_steps`` and the window is
  placed uniformly inside the episode. Accessibility (:meth:`_target_geometry`) is gated
  on the window, so the base observation, ``step``, and every reference policy in
  :mod:`sarsat.reference` respect windows without change. The windows are sampled from a
  key folded off the reset key, so orbits and targets are exactly those of ``SarSat`` for
  the same key and ``window_steps=(0, 0)`` reproduces it entirely.
* :class:`WindowedCoopSarSat` -- :class:`sarsat.coop.CoopSarSat` on top of it: the cluster
  look-ahead is gated on the cluster windows and each beam slot gains a ninth feature,
  the fraction of the target's window still open. Everything else, including the
  eight-feature layout of ``CoopSarSat``, is unchanged.

Why windows: with persistent targets a satellite that skips a target can leave it to the
next pass; with windows, *someone* has to be there in time, which is the decision a
centralised planner makes well and an independent policy cannot (DECISIONS.md issue 24).
"""

from typing import NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from sarsat.coop import SLOT_DIM, CoopSarSat
from sarsat.env import SarSat
from sarsat.types import Orbit


class WindowedState(NamedTuple):
    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    target_window: jax.Array  # (M, 2) int32 first and last step at which the target pays


class WindowedCoopState(NamedTuple):
    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    target_window: jax.Array
    cluster_access: jax.Array
    team_future: jax.Array
    body: jax.Array
    visible: jax.Array


class WindowedSarSat(SarSat):
    """``SarSat`` whose targets can only be imaged inside a time window."""

    def __init__(
        self,
        *args,
        window_steps: Tuple[int, int] = (0, 0),
        background_windows: bool = True,
        background_window_steps: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        """``window_steps`` is the ``(min, max)`` window length; with
        ``background_windows=False`` only the hotspot clusters are windowed events and the
        background stays available all episode. ``background_window_steps`` gives the
        background its own window lengths (default: ``window_steps``, drawn exactly as
        before; the cluster windows are the same either way)."""
        super().__init__(*args, **kwargs)
        for steps in (window_steps, background_window_steps or window_steps):
            if not 0 <= steps[0] <= steps[1] <= self.time_limit:
                raise ValueError("window lengths must satisfy 0 <= min <= max <= time_limit")
        self.window_steps = tuple(int(w) for w in window_steps)
        self.background_windows = background_windows
        self.background_window_steps = tuple(
            int(w) for w in (background_window_steps or window_steps)
        )

    def _initial_state(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> WindowedState:
        base = super()._initial_state(key, num_targets)
        # Folded off the reset key so orbits and targets are those of ``SarSat``.
        window = self._sample_windows(jax.random.fold_in(key, 24))
        return WindowedState(*base, target_window=window)

    def _sample_windows(self, key: jax.Array) -> jax.Array:
        """``(M, 2)`` first and last step at which each target slot can be imaged."""
        m = self.max_targets
        lo, hi = self.window_steps
        if hi == 0:
            return jnp.stack([jnp.zeros((m,), jnp.int32), jnp.full((m,), self.time_limit)], -1)
        # One window per target slot, then one per cluster.
        window = self._place_windows(key, m + self.hotspots, lo, hi)  # (units, 2)
        background = window[:m]
        if self.background_window_steps != self.window_steps:
            background = self._place_windows(
                jax.random.fold_in(key, 1), m, *self.background_window_steps
            )
        if not self.hotspots:
            return background
        slot = jnp.arange(m)
        is_hot = slot < self.hotspots * self.hotspot_targets
        if not self.background_windows:
            background = jnp.stack(
                [jnp.zeros((m,), jnp.int32), jnp.full((m,), self.time_limit)], -1
            )
        return jnp.where(is_hot[:, None], window[m:][slot % self.hotspots], background)

    def _place_windows(self, key: jax.Array, count: int, lo: int, hi: int) -> jax.Array:
        """``(count, 2)`` windows of length uniform in ``[lo, hi]``, placed uniformly."""
        k_len, k_open = jax.random.split(key)
        length = jax.random.randint(k_len, (count,), lo, hi + 1)
        first = 1 + jnp.floor(
            jax.random.uniform(k_open, (count,)) * (self.time_limit - length + 1)
        ).astype(jnp.int32)
        return jnp.stack([first, first + length - 1], axis=-1)

    def _target_geometry(self, state, step: jax.Array) -> Tuple[jax.Array, jax.Array]:
        body, visible = super()._target_geometry(state, step)
        window = state.target_window
        return body, visible & (window[:, 0] <= step) & (step <= window[:, 1])


class WindowedCoopSarSat(CoopSarSat, WindowedSarSat):
    """``CoopSarSat`` over windowed targets: gated look-ahead and a window-left feature."""

    state_cls = WindowedCoopState
    slot_dim = SLOT_DIM + 1

    def _schedule_mask(self, base: WindowedState, steps: jax.Array) -> jax.Array:
        window = base.target_window[: self.num_clusters]  # cluster c's window is slot c's
        return (window[:, 0] <= steps[:, None]) & (steps[:, None] <= window[:, 1])

    def _extra_slot_features(self, state, slot_target: jax.Array, t1: jax.Array) -> list:
        scale = max(self.window_steps[1], self.background_window_steps[1]) or self.time_limit
        closes = state.target_window[slot_target, 1]
        return [jnp.clip((closes - t1 + 1) / scale, 0.0, 1.0)[..., None]]
