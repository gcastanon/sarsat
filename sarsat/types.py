"""State and observation containers for the SarSat environment."""

from typing import NamedTuple

import jax


class Orbit(NamedTuple):
    """Circular-orbit elements, one entry per satellite. Constant during an episode."""

    radius: jax.Array  # (N,) orbit radius from Earth's centre [km]
    mean_motion: jax.Array  # (N,) angular rate along the orbit [rad/s]
    inclination: jax.Array  # (N,) [rad]
    raan: jax.Array  # (N,) right ascension of the ascending node at t=0 [rad]
    phase: jax.Array  # (N,) argument of latitude at t=0 [rad]


class State(NamedTuple):
    """Full environment state. N = satellites, M = ``max_targets`` (padded)."""

    key: jax.Array  # PRNG key; unused by the dynamics, required by auto-reset wrappers
    step: jax.Array  # () int32, number of steps taken
    orbit: Orbit
    battery: jax.Array  # (N,) state of charge in [0, 1]
    target_latlon: jax.Array  # (M, 2) geodetic latitude / longitude [deg]
    target_pos: jax.Array  # (M, 3) Earth-fixed position [km]
    target_priority: jax.Array  # (M,) reward for imaging the target; 0 for padding
    target_active: jax.Array  # (M,) bool, True until imaged; False for padding


class Observation(NamedTuple):
    """Per-agent observation. Field names match ``mapx.types.Observation``."""

    agents_view: jax.Array  # (N, obs_dim) float32 egocentric features
    action_mask: jax.Array  # (N, 3) bool; the sense slot closes when the battery is too low
    step_count: jax.Array  # (N,) int32
