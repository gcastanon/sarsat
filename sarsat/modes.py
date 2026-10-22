"""Collection modes: the sensor settings a satellite chooses between, one per look.

A :class:`SensorMode` bundles what differs between, say, a spotlight and a strip-map
collection: the battery a look costs, the image quality it delivers, how long it takes,
where it can look (incidence band and squint limit) and its beam footprint.
``SarSat(modes=[...])`` lets each satellite pick one mode per step, or none, through the
action's ``look`` slot (DECISIONS.md issue 31). Without ``modes`` the environment has the
single mode its scalar arguments describe, which is the sensor every earlier issue models.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple, Union


@dataclass(frozen=True)
class SensorMode:
    """One collection mode. The defaults are the default ``SarSat`` sensor.

    Attributes:
        name: label for logs and plots.
        sense_cost: battery fraction one look in this mode consumes.
        quality: image quality the mode delivers, on whatever scale target requirements
            use (higher is better). A look only images a target whose required quality
            this meets (:meth:`sarsat.env.SarSat._required_quality`).
        duration_s: how long one look takes. Recorded but **not simulated**: every look is
            a snapshot at the step boundary (issue 7) whatever its duration.
        incidence_deg: ``(min, max)`` incidence angle at the target the mode can image at.
        max_squint_deg: largest along-track angle off broadside the mode can image at.
        beam_half_width_deg: rectangular beam half-widths ``(cross-track, squint)``.
    """

    name: str = "default"
    sense_cost: float = 0.10
    quality: float = 0.0
    duration_s: float = 0.0
    incidence_deg: Tuple[float, float] = (20.0, 45.0)
    max_squint_deg: float = 25.0
    beam_half_width_deg: Tuple[float, float] = (4.0, 4.0)

    def __post_init__(self) -> None:
        # Accept lists (Hydra / JSON configs) for the pairs, and keep the object hashable.
        object.__setattr__(self, "incidence_deg", tuple(float(x) for x in self.incidence_deg))
        object.__setattr__(
            self, "beam_half_width_deg", tuple(float(x) for x in self.beam_half_width_deg)
        )
        if len(self.incidence_deg) != 2 or len(self.beam_half_width_deg) != 2:
            raise ValueError(f"mode {self.name!r}: incidence and beam need two values each")
        if self.duration_s < 0:
            raise ValueError(f"mode {self.name!r}: duration_s must be non-negative")


def as_mode(mode: Union[SensorMode, Mapping[str, Any]]) -> SensorMode:
    """A :class:`SensorMode` from itself or from a mapping of its fields (a config entry)."""
    return mode if isinstance(mode, SensorMode) else SensorMode(**dict(mode))
