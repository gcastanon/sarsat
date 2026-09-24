"""What a tasking request looks like once it has been understood."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PRIORITIES = ("low", "normal", "high")
MAX_DEADLINE_MINUTES = 30
DEFAULT_DEADLINE_MINUTES = 30


@dataclass
class Parsed:
    """A request reduced to a location, a priority and a deadline.

    ``lat``/``lon`` are ``None`` when no location could be resolved; ``place`` is the name
    that was looked up (or an empty string); ``source`` says who resolved the location:
    ``coordinates`` (given in the text), ``gazetteer`` (name matched offline), ``model``
    (the language model's own guess, approximate) or ``none``.
    """

    text: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    place: str = ""
    priority: str = "normal"
    deadline_minutes: int = DEFAULT_DEADLINE_MINUTES
    source: str = "none"
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)
    extraction: Dict[str, Any] = field(default_factory=dict)  # the model's raw fields

    @property
    def located(self) -> bool:
        return self.lat is not None and self.lon is not None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["located"] = self.located
        return d


def clamp_deadline(minutes: Optional[float]) -> int:
    """Deadlines are 1..30 minutes; missing or out of range becomes the default / the cap."""
    if minutes is None:
        return DEFAULT_DEADLINE_MINUTES
    return int(max(1, min(MAX_DEADLINE_MINUTES, round(float(minutes)))))


def normalise_priority(value: Optional[str]) -> str:
    v = (value or "normal").strip().lower()
    if v in PRIORITIES:
        return v
    if v in {"urgent", "critical", "top", "highest", "max", "maximum", "immediate", "asap"}:
        return "high"
    if v in {"lowest", "min", "minimum", "background", "routine", "whenever"}:
        return "low"
    if v in {"medium", "mid", "standard", "default", "regular"}:
        return "normal"
    return "normal"
