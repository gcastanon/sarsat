"""Image-quality requirements: a target is only imaged by a look whose mode is good enough.

:class:`QualitySarSat` is :class:`sarsat.env.SarSat` whose state carries a
``target_quality`` ``(M,)``: the image quality each target requires, on the scale of
:attr:`sarsat.modes.SensorMode.quality`. It fills the ``_required_quality`` hook of
DECISIONS.md issue 31, so the dynamics and the plain observation's per-mode flags respect
it without change. Every hotspot cluster draws one level that all of its targets share (an
event needs one kind of image), and every background target draws its own. The levels are
sampled from a key folded off the reset key, so orbits and targets are exactly those of
``SarSat`` for the same key, and the default single level ``0.0`` reproduces it entirely.
:class:`QualityCoopSarSat` is the same on :class:`sarsat.coop.CoopSarSat`, and
:class:`WindowedQualitySarSat` / :class:`WindowedQualityCoopSarSat` combine it with the
time-windowed events of :mod:`sarsat.windows` (DECISIONS.md issue 35).

Why per-cluster levels: background targets are sparse (about one per 30,000 km^2), so a
wide beam rarely catches more than the target it is aimed at. A wide, coarse mode only
pays off on a cluster that accepts its quality, which is the choice a satellite makes
between coverage and detail (DECISIONS.md issue 32).

``MODES`` are four SAR collection modes with rough public figures (issue 32), cheapest
first, so :func:`sarsat.evaluate.greedy_policy_for` (the first mode that images the target)
picks the cheapest one that is good enough. ``quality`` is on a radar-NIIRS-like scale,
where halving the resolution cell adds about one level.
"""

from typing import NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from sarsat.coop import SLOT_DIM, CoopSarSat
from sarsat.env import SarSat
from sarsat.modes import SensorMode
from sarsat.types import Orbit
from sarsat.windows import WindowedCoopSarSat, WindowedSarSat

# Beam half-widths at ~550 km altitude and ~35 degrees incidence: about 0.036 deg per km of
# footprint width cross-track and 0.043 deg per km of footprint length along-track. The strip
# modes look broadside, but collect while the ground slides ~420 km past in one 60 s step; a
# snapshot (issue 7) at their physical +-5-10 degrees would reach a target on few passes, so
# their squint limit is the +-20 degrees that sweep spans (issue 32).
MODES = (
    # 3-5 m resolution, ~28 x 46 km strip, 7-15 s per scene.
    SensorMode(
        name="stripmap",
        sense_cost=0.10,
        quality=3.5,
        duration_s=10.0,
        incidence_deg=(20.0, 45.0),
        max_squint_deg=20.0,
        beam_half_width_deg=(1.0, 2.0),
    ),
    # 0.5-1 m, ~5 km patch, 5-20 s plus a slew to dwell on it.
    SensorMode(
        name="spotlight",
        sense_cost=0.13,
        quality=5.0,
        duration_s=10.0,
        incidence_deg=(15.0, 55.0),
        max_squint_deg=25.0,
        beam_half_width_deg=(0.2, 0.2),
    ),
    # 15-40 m, ~100 km swath; cheapest per km^2, not per look.
    SensorMode(
        name="scansar",
        sense_cost=0.20,
        quality=1.5,
        duration_s=25.0,
        incidence_deg=(20.0, 45.0),
        max_squint_deg=20.0,
        beam_half_width_deg=(3.6, 4.0),
    ),
    # 0.15-0.25 m, ~4 km patch; the dwell sweeps squint, so its centre squint is tighter.
    SensorMode(
        name="staring",
        sense_cost=0.30,
        quality=7.0,
        duration_s=45.0,
        incidence_deg=(20.0, 50.0),
        max_squint_deg=15.0,
        beam_half_width_deg=(0.15, 0.15),
    ),
)


class QualityState(NamedTuple):
    """:class:`sarsat.types.State` plus the image quality each target requires."""

    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    target_quality: jax.Array  # (M,) float32 quality a look must deliver to image the target


class QualitySarSat(SarSat):
    """``SarSat`` whose targets each require a minimum image quality."""

    _quality_state_cls = QualityState  # the base state plus ``target_quality``

    def __init__(
        self,
        *args,
        hotspot_quality: Sequence[float] = (0.0,),
        hotspot_quality_weights: Optional[Sequence[float]] = None,
        background_quality: Sequence[float] = (0.0,),
        background_quality_weights: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> None:
        """Each hotspot cluster draws its required quality from ``hotspot_quality`` and each
        background target from ``background_quality``, with probabilities proportional to
        the matching ``*_weights`` (uniform by default). Pair these with ``modes``: the
        default sensor's quality is ``0.0``, so any positive requirement is unreachable."""
        super().__init__(*args, **kwargs)
        self.hotspot_quality = self._levels(hotspot_quality, hotspot_quality_weights)
        self.background_quality = self._levels(background_quality, background_quality_weights)

    @staticmethod
    def _levels(levels: Sequence[float], weights: Optional[Sequence[float]]):
        levels = tuple(float(q) for q in levels)
        weights = tuple(float(w) for w in (weights or [1.0] * len(levels)))
        if not levels or len(weights) != len(levels):
            raise ValueError("quality levels need one weight each, and at least one level")
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("quality weights must be non-negative with a positive sum")
        return levels, weights

    def _initial_state(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> QualityState:
        base = super()._initial_state(key, num_targets)
        # Folded off the reset key so orbits and targets are those of ``SarSat``.
        quality = self._sample_quality(jax.random.fold_in(key, 32))
        return self._quality_state_cls(*base, target_quality=quality)

    def _sample_quality(self, key: jax.Array) -> jax.Array:
        """``(M,)`` required quality for every target slot."""
        m = self.max_targets
        k_hot, k_background = jax.random.split(key)
        quality = self._draw(k_background, m, *self.background_quality)
        if not self.hotspots:
            return quality
        slot = jnp.arange(m)
        cluster = self._draw(k_hot, self.hotspots, *self.hotspot_quality)
        is_hot = slot < self.hotspots * self.hotspot_targets
        return jnp.where(is_hot, cluster[slot % self.hotspots], quality)

    @staticmethod
    def _draw(key: jax.Array, count: int, levels, weights) -> jax.Array:
        p = jnp.asarray(weights, jnp.float32)
        index = jax.random.choice(key, len(levels), (count,), p=p / jnp.sum(p))
        return jnp.asarray(levels, jnp.float32)[index]

    def _required_quality(self, state) -> Optional[jax.Array]:
        return state.target_quality


class QualityCoopState(NamedTuple):
    """:class:`sarsat.coop.CoopState` plus the image quality each target requires."""

    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    target_quality: jax.Array
    cluster_access: jax.Array
    team_future: jax.Array
    body: jax.Array
    visible: jax.Array


class QualityCoopSarSat(CoopSarSat, QualitySarSat):
    """``CoopSarSat`` over targets with quality requirements: its mode-aware beam slots
    (DECISIONS.md issue 33) only count what a mode is good enough to image."""

    state_cls = QualityCoopState


class WindowedQualityState(NamedTuple):
    """:class:`sarsat.windows.WindowedState` plus the image quality each target requires."""

    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    target_window: jax.Array
    target_quality: jax.Array


class WindowedQualitySarSat(QualitySarSat, WindowedSarSat):
    """Time-windowed targets (``WindowedSarSat``) that also require a minimum image quality.
    Orbits, targets and windows are those of ``WindowedSarSat`` for the same key."""

    _quality_state_cls = WindowedQualityState


class WindowedQualityCoopState(NamedTuple):
    """:class:`sarsat.windows.WindowedCoopState` plus the image quality each target requires."""

    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    target_window: jax.Array
    target_quality: jax.Array
    cluster_access: jax.Array
    team_future: jax.Array
    body: jax.Array
    visible: jax.Array


class WindowedQualityCoopSarSat(WindowedCoopSarSat, WindowedQualitySarSat):
    """``WindowedCoopSarSat`` (window-gated look-ahead, window-left slot feature) over targets
    with quality requirements, with the mode-aware beam slots of issue 33."""

    state_cls = WindowedQualityCoopState
    slot_dim = SLOT_DIM + 1
