"""Announced requests: every collection window is revealed at a random time before it opens.

In :mod:`sarsat.windows` every window exists from the reset, and the cooperative
observation's team access counts and critic summary already reflect windows that open much
later. Here each request (an event cluster, or one background target) also carries an
*announcement step*, drawn uniformly from the start of the episode up to ``announce_lead_s``
before its window opens (``0`` when the window opens sooner than that: a request placed
before the episode began). Nothing in the observation depends on a request before it is
announced.

The dynamics and reward are those of :class:`sarsat.windows.WindowedSarSat`: since a window
opens at least ``announce_lead_s`` after its announcement, visibility (gated on the window)
already implies the request is known, so ``step`` and the plain observation need no change,
and neither do the policies that act only on what is visible now (``greedy_beam``,
``coop_dedup``). ``solo_plan`` and ``coop_plan`` precompute every window at the reset: under
announcements they are clairvoyant yardsticks, not fair ones.

* :class:`AnnouncedSarSat` -- ``WindowedSarSat`` whose state also carries
  ``target_announce`` ``(M,)``. Targets of one cluster share its announcement.
* :class:`AnnouncedCoopSarSat` -- ``WindowedCoopSarSat`` whose observation (actor slots,
  look-ahead, and the critic's team summary) masks clusters and targets not yet announced.

Orbits, targets and windows are those of ``WindowedSarSat`` for the same key; the
announcements are sampled from a further key folded off the reset key (DECISIONS.md
issue 28).
"""

import math
from typing import NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from sarsat.types import Orbit
from sarsat.windows import WindowedCoopSarSat, WindowedSarSat


class AnnouncedState(NamedTuple):
    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    target_window: jax.Array
    target_announce: jax.Array  # (M,) int32 step from which the request is known


class AnnouncedCoopState(NamedTuple):
    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    target_window: jax.Array
    target_announce: jax.Array
    cluster_access: jax.Array
    team_future: jax.Array
    body: jax.Array
    visible: jax.Array


class AnnouncedSarSat(WindowedSarSat):
    """``WindowedSarSat`` whose requests are announced at random, ahead of their windows."""

    def __init__(self, *args, announce_lead_s: float = 1800.0, **kwargs) -> None:
        """``announce_lead_s`` is the least notice a request gets before its window opens,
        in seconds (rounded up to whole steps)."""
        super().__init__(*args, **kwargs)
        if announce_lead_s < 0:
            raise ValueError("announce_lead_s must be non-negative")
        self.announce_lead_s = float(announce_lead_s)
        self.announce_lead = math.ceil(self.announce_lead_s / self.step_seconds - 1e-9)

    def _initial_state(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> AnnouncedState:
        base = super()._initial_state(key, num_targets)
        announce = self._sample_announcements(jax.random.fold_in(key, 28), base.target_window)
        return AnnouncedState(*base, target_announce=announce)

    def _sample_announcements(self, key: jax.Array, window: jax.Array) -> jax.Array:
        """``(M,)`` announcement step of each target slot: uniform on
        ``[0, first - announce_lead]``, one draw per request (cluster or background target)."""
        m = self.max_targets
        latest = jnp.maximum(window[:, 0] - self.announce_lead, 0)  # (M,)
        u = jax.random.uniform(key, (m + self.hotspots,))
        if self.hotspots:
            slot = jnp.arange(m)
            is_hot = slot < self.hotspots * self.hotspot_targets
            u = jnp.where(is_hot, u[m:][slot % self.hotspots], u[:m])
        else:
            u = u[:m]
        announce = jnp.floor(u * (latest + 1).astype(jnp.float32)).astype(jnp.int32)
        return jnp.minimum(announce, latest)  # guards u * (latest + 1) rounding up to it

    def known(self, state) -> jax.Array:
        """``(M,)`` whether each target's request has been announced by ``state.step``."""
        return state.target_announce <= state.step


class AnnouncedCoopSarSat(WindowedCoopSarSat, AnnouncedSarSat):
    """``WindowedCoopSarSat`` whose observation only uses announced requests."""

    state_cls = AnnouncedCoopState

    def _known(self, state) -> Tuple[jax.Array, jax.Array]:
        known_m = self.known(state)
        if not self.hotspots:
            return jnp.ones((self.num_clusters,), bool), known_m
        return known_m[: self.num_clusters], known_m  # cluster c's announcement is slot c's
