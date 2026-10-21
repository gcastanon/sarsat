"""A fixed episode: every reset returns the same satellites, targets, priorities and charge.

:class:`FixedSarSat` and :class:`FixedCoopSarSat` read a scenario file (JSON, as written by
``scripts/adversarial_scenario.py``) and start every episode from it, whatever the reset
key: its satellites' circular orbital elements and initial charge, and its targets'
positions and priorities (normalised to sum to one). Everything else, from the dynamics to
the observation and the reference policies, is the parent's. The file's ``environment``
block must agree with the constructor's ``sense_cost``, ``recharge_rate``, ``time_limit``
and ``step_seconds``, which default to it. With ``hotspots`` set, the first
``hotspots * hotspot_targets`` targets of the file are the hotspot (event) targets, which
the coop observation's cluster look-ahead tracks (DECISIONS.md issue 38).

Why: a hand-built episode that a heuristic gets wrong in a known way is a test a learner
can be trained on directly, and its optimum can be found by enumeration
(``scripts/adversarial_bruteforce.py``).
"""

import json
from pathlib import Path
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.coop import CoopSarSat
from sarsat.env import SarSat
from sarsat.orbits import EARTH_RADIUS_KM
from sarsat.targets import latlon_to_unit_vector
from sarsat.types import Orbit, State

_REPO = Path(__file__).resolve().parents[1]
_SHARED = {"sense_cost": "sense_cost", "recharge_rate": "recharge_rate",
           "time_limit": "time_limit_steps", "step_seconds": "step_seconds"}


def load_scenario(path: Union[str, Path]) -> dict:
    """The scenario document at ``path``; a relative path is taken from the repository root."""
    path = Path(path)
    return json.loads((path if path.is_absolute() else _REPO / path).read_text())


class _Fixed:
    """Mixin: ``_initial_state`` returns the scenario file's episode."""

    def __init__(self, *args, scenario_file: str, **kwargs) -> None:
        doc = load_scenario(scenario_file)
        sats, targets, block = doc["satellites"], doc["targets"], doc["environment"]
        for name, size in (("num_satellites", len(sats)), ("max_targets", len(targets))):
            if kwargs.setdefault(name, size) != size:
                raise ValueError(f"{name}={kwargs[name]} but the scenario file has {size}")
        for arg, key in _SHARED.items():
            if kwargs.setdefault(arg, block[key]) != block[key]:
                raise ValueError(f"{arg}={kwargs[arg]} but the scenario file has {block[key]}")
        super().__init__(*args, **kwargs)
        f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
        self._fixed_orbit = Orbit(
            radius=f32([s["radius_km"] for s in sats]),
            mean_motion=f32([s["mean_motion_rad_s"] for s in sats]),
            inclination=f32(np.radians([s["inclination_deg"] for s in sats])),
            raan=f32(np.radians([s["raan_deg"] for s in sats])),
            phase=f32(np.radians([s["arg_latitude_deg_t0"] for s in sats])),
        )
        self._fixed_battery = f32([s["initial_battery"] for s in sats])
        self._fixed_latlon = f32([[t["lat_deg"], t["lon_deg"]] for t in targets])
        priority = f32([t["priority"] for t in targets])
        self._fixed_priority = priority / priority.sum()

    def _initial_state(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> State:
        del num_targets  # the file fixes the targets
        latlon = jnp.asarray(self._fixed_latlon)
        return State(
            key=key,
            step=jnp.zeros((), jnp.int32),
            orbit=Orbit(*(jnp.asarray(x) for x in self._fixed_orbit)),
            battery=jnp.asarray(self._fixed_battery),
            target_latlon=latlon,
            target_pos=EARTH_RADIUS_KM * latlon_to_unit_vector(latlon),
            target_priority=jnp.asarray(self._fixed_priority),
            target_active=jnp.ones((self.max_targets,), bool),
        )


class FixedSarSat(_Fixed, SarSat):
    """``SarSat`` that always starts from the scenario file's episode."""


class FixedCoopSarSat(_Fixed, CoopSarSat):
    """``CoopSarSat`` that always starts from the scenario file's episode."""
