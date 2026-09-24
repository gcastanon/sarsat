"""The rule-based fast path: explicit coordinates, priority words, deadlines.

This runs before any model. When the text carries coordinates it settles the location
outright (no model, no gazetteer); priority and deadline phrases are parsed here in every
case, so the model only has to name the place. The same rules double as the test oracle for
the model path.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from live.nl.schema import Parsed, clamp_deadline

_NUM = r"[-+]?\d+(?:\.\d+)?"
# "48.86, 2.35" / "48.86 2.35" / "lat 48.86 lon 2.35" / "48.86N 2.35E" / "48.86° N, 2.35° E"
_COORD_PAIR = re.compile(
    rf"(?:lat(?:itude)?\s*[:=]?\s*)?({_NUM})\s*°?\s*([NS])?\s*[,;/ ]\s*"
    rf"(?:lon(?:gitude)?\s*[:=]?\s*)?({_NUM})\s*°?\s*([EW])?(?![\d.])",
    re.IGNORECASE,
)
_LAT_LON_WORDS = re.compile(
    rf"lat(?:itude)?\s*[:=]?\s*({_NUM})\s*°?\s*([NS])?.*?lon(?:gitude)?\s*[:=]?\s*({_NUM})\s*°?\s*([EW])?",
    re.IGNORECASE | re.DOTALL,
)

_DEADLINE = [
    (
        re.compile(
            r"\bwithin\s+(?:the\s+next\s+)?(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|h(?:ou)?rs?|h)\b", re.I
        ),
        None,
    ),
    (
        re.compile(
            r"\bin\s+(?:the\s+next\s+)?(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|h(?:ou)?rs?|h)\b", re.I
        ),
        None,
    ),
    (
        re.compile(
            r"\b(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|h(?:ou)?rs?)\s*(?:deadline|window|or less|max(?:imum)?)\b",  # noqa: E501
            re.I,
        ),
        None,
    ),
    (
        re.compile(
            r"\b(?:deadline|by|before|no later than)\s+(?:the\s+next\s+)?(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|h(?:ou)?rs?|h)\b",  # noqa: E501
            re.I,
        ),
        None,
    ),
    (re.compile(r"\bwithin\s+(?:the\s+)?(?:half\s+an?\s+hour|half-hour)\b", re.I), 30),
    (re.compile(r"\b(?:within\s+)?(?:a\s+)?quarter\s+(?:of\s+an\s+)?hour\b", re.I), 15),
    (re.compile(r"\b(?:asap|as soon as possible|immediately|right away|right now|now)\b", re.I), 5),
]
_HIGH = re.compile(
    r"\b(high(?:est)?[\s-]*priority|urgent(?:ly)?|critical|top[\s-]*priority|priority\s*(?:1|one|a)|"
    r"emergency|max(?:imum)?[\s-]*priority|immediate(?:ly)?|asap|as soon as possible)\b",
    re.I,
)
_LOW = re.compile(
    r"\b(low(?:est)?[\s-]*priority|priority\s*(?:3|three|c)|low[\s-]*prio|routine|background|"
    r"when(?:ever)?\s+(?:convenient|possible|you can)|no\s+rush|not\s+urgent|if\s+possible|opportunistic)\b",  # noqa: E501
    re.I,
)
_NORMAL = re.compile(r"\b(normal|medium|standard|regular)[\s-]*priority\b", re.I)


def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """``(lat, lon)`` if the text states them, else ``None``."""
    m = _LAT_LON_WORDS.search(text) or _COORD_PAIR.search(text)
    if not m:
        return None
    lat, ns, lon, ew = m.group(1), m.group(2), m.group(3), m.group(4)
    lat, lon = float(lat), float(lon)
    if ns and ns.upper() == "S":
        lat = -abs(lat)
    if ew and ew.upper() == "W":
        lon = -abs(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def parse_deadline(text: str) -> Optional[int]:
    """Deadline in minutes if the text states one, else ``None``."""
    for pattern, fixed in _DEADLINE:
        m = pattern.search(text)
        if not m:
            continue
        if fixed is not None:
            return fixed
        value, unit = float(m.group(1)), m.group(2).lower()
        return round(value * 60 if unit.startswith("h") else value)
    return None


def parse_priority(text: str) -> Optional[str]:
    """``high``/``low``/``normal`` if the text says so, else ``None``."""
    if _HIGH.search(text):
        return "high"
    if _LOW.search(text):
        return "low"
    if _NORMAL.search(text):
        return "normal"
    return None


def parse_rules(text: str) -> Parsed:
    """Everything the rules can settle; the location only when coordinates are given."""
    p = Parsed(text=text)
    coords = parse_coordinates(text)
    if coords:
        p.lat, p.lon = coords
        p.source, p.confidence = "coordinates", 1.0
    deadline = parse_deadline(text)
    p.deadline_minutes = clamp_deadline(deadline)
    if deadline is not None and deadline > p.deadline_minutes:
        p.notes.append(f"deadline of {deadline} min capped at {p.deadline_minutes}")
    priority = parse_priority(text)
    p.priority = priority or "normal"
    p.extraction = {
        "coordinates": coords,
        "deadline_minutes": deadline,
        "priority": priority,
    }
    return p
