"""The raw numbers of a windowed scenario's events: one per hotspot cluster.

An event of :class:`sarsat.windows.WindowedSarSat` is a cluster of ``hotspot_targets``
targets Gaussian-scattered around a centre, all paying ``hotspot_weight`` inside one
shared window of steps. That is exactly the five numbers a tasking request carries: the
centre's latitude and longitude, the window's first and last step, and the priority.

The centre is not kept in the state (only the scattered targets are), so it is re-drawn
here with the keys ``SarSat._initial_state`` and ``SarSat._sample_target_field`` use; the
mean of the cluster's targets would be off by ``hotspot_radius_km / sqrt(hotspot_targets)``.
"""

from typing import List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.windows import WindowedSarSat


class Event(NamedTuple):
    cluster: int
    lat: float  # centre latitude [deg]
    lon: float  # centre longitude [deg]
    first: int  # first step at which the event's targets pay
    last: int  # last step at which they pay (inclusive)
    priority: float  # weight of an event target relative to a background target


def cluster_centres(env: WindowedSarSat, key: jax.Array) -> jax.Array:
    """``(hotspots, 2)`` latitude / longitude of the cluster centres for reset ``key``."""
    _, _, k_target = jax.random.split(key, 3)  # as in SarSat._initial_state
    k_centre, _ = jax.random.split(k_target)  # as in SarSat._sample_target_field
    return env._sample_targets(k_centre, jnp.ones((env.hotspots,), bool))


def extract_events(env: WindowedSarSat, key: jax.Array) -> List[Event]:
    """The events of the episode ``env.reset(key)`` starts, in cluster order."""
    if not env.hotspots:
        raise ValueError("the scenario has no hotspot clusters, so no events")
    state = jax.jit(env._initial_state)(key)
    centres = np.asarray(cluster_centres(env, key), np.float64)
    window = np.asarray(state.target_window[: env.hotspots])  # cluster c's window is slot c's
    weight = np.asarray(state.target_priority, np.float64)  # normalised to sum to one
    num_hot = env.hotspots * env.hotspot_targets
    # Relative to a background target, so the number is the scenario's own weight.
    background = weight[num_hot] if num_hot < env.max_targets and weight[num_hot] > 0 else None
    events = []
    for c in range(env.hotspots):
        priority = weight[c] / background if background else env.hotspot_weight
        events.append(
            Event(
                cluster=c,
                lat=float(centres[c, 0]),
                lon=float(centres[c, 1]),
                first=int(window[c, 0]),
                last=int(window[c, 1]),
                priority=round(float(priority), 4),
            )
        )
    return events
