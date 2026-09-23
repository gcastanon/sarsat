"""Read a request written by :mod:`sarsat.tasking.render` back into its five numbers, and
rebuild an episode from the readings.

:func:`read_request` sees only the text and the time it was issued. It follows the
conventions in :mod:`sarsat.tasking.render` and returns what an exact reading gives (the
record's ``stated``), so a rebuilt target sits within the record's ``tol`` of the
original. It covers the generator's own wordings, not free text: it is the reference an
LLM parser is compared with, and the path from a data set back to an episode.

:func:`rebuild_state` puts the readings of every target of a seed into the state
``env.reset(key)`` starts (orbits from the key, targets from the text), in slot order.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.orbits import EARTH_RADIUS_KM
from sarsat.targets import latlon_to_unit_vector
from sarsat.tasking.geo import COMPASS16, KM_PER_UNIT, Places, destination
from sarsat.tasking.render import (
    COMPASS_WORDS,
    PRIORITY_PHRASES,
    TIERS,
    parse_iso,
    steps_from_times,
)
from sarsat.windows import WindowedSarSat, WindowedState

UNITS = {"km": "km", "kilometers": "km", "mi": "mi", "miles": "mi", "nm": "nmi"}
UNITS["nautical miles"] = "nmi"
UNIT_RE = "|".join(sorted(UNITS, key=len, reverse=True))
DIRECTIONS = {c: 22.5 * k for k, c in enumerate(COMPASS16)}
DIRECTIONS.update({COMPASS_WORDS[c]: 22.5 * k for k, c in enumerate(COMPASS16)})
DIR_RE = "|".join(sorted(DIRECTIONS, key=len, reverse=True))  # "north-northwest" before "north"

DMS_RE = re.compile(r"(\d{2,3})(?:°|d)(\d{2}(?:\.\d)?)(?:′|')(?:(\d{2})(?:″|\"))?([NSEW])")  # noqa: RUF001
DECIMAL_RE = re.compile(r"(-?\d+\.\d+)([NSEW])?")

DUR = r"\d+ (?:hours?|h)(?: \d+ min)?|\d+ (?:minutes?|min)"
CLOCK = r"(\d\d):?(\d\d)(?:z| utc)"

PHRASE_TIER = {
    phrase.lower(): tier
    for tier, (prefixes, suffixes) in PRIORITY_PHRASES.items()
    for phrase in (*prefixes, *suffixes)
}
PHRASE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(PHRASE_TIER, key=len, reverse=True)) + r")\b"
)


class ReadError(ValueError):
    """The text has no reading under the generator's conventions."""


# ---------------------------------------------------------------------------------------- #
# Location
# ---------------------------------------------------------------------------------------- #


def _find_place(text: str, places: Places) -> Optional[Tuple[int, int, int]]:
    """``(place, start, end)`` of the longest ``Name, Region`` label in ``text``."""
    best = None
    for region in places.regions:
        needle = ", " + region
        pos = text.find(needle)
        while pos != -1:
            end = pos + len(needle)
            if end == len(text) or not text[end].isalnum():
                words = text[:pos].split(" ")
                for k in range(min(8, len(words)), 0, -1):  # the longest name that exists
                    name = " ".join(words[-k:])
                    i = places.lookup(name, region)
                    if i is not None:
                        if best is None or end - (pos - len(name)) > best[2] - best[1]:
                            best = (i, pos - len(name), end)
                        break
            pos = text.find(needle, pos + 1)
    return best


def read_location(text: str, places: Optional[Places]) -> Tuple[float, float, Optional[str], str]:
    """``(lat, lon, time zone of the named place or None, text without the location)``."""
    found = _find_place(text, places) if places is not None else None
    if found is not None:
        i, s, e = found
        before, after = text[:s], text[e:]
        a_lat, a_lon = float(places.lat[i]), float(places.lon[i])
        zone = str(places.timezone[i])
        m = re.search(rf"(\d+) ({UNIT_RE}) ({DIR_RE}) of $", before)
        if m:
            dist = float(m.group(1)) * KM_PER_UNIT[UNITS[m.group(2)]]
            lat, lon = destination(a_lat, a_lon, DIRECTIONS[m.group(3)], dist)
            return lat, lon, zone, before[: m.start()] + after
        m = re.search(rf"(\d+) ({UNIT_RE}) from $", before)
        m2 = re.match(r" on a bearing of (\d{3}) degrees", after)
        if m and m2:
            dist = float(m.group(1)) * KM_PER_UNIT[UNITS[m.group(2)]]
            lat, lon = destination(a_lat, a_lon, float(m2.group(1)), dist)
            return lat, lon, zone, before[: m.start()] + after[m2.end() :]
        m2 = re.match(rf", bearing (\d{{3}}), (\d+) ({UNIT_RE})\b", after)
        if m2:
            dist = float(m2.group(2)) * KM_PER_UNIT[UNITS[m2.group(3)]]
            lat, lon = destination(a_lat, a_lon, float(m2.group(1)), dist)
            return lat, lon, zone, before + after[m2.end() :]
        return a_lat, a_lon, zone, before + after
    dms = list(DMS_RE.finditer(text))
    if len(dms) >= 2:
        values = []
        for m in dms[:2]:
            value = int(m.group(1)) + float(m.group(2)) / 60 + int(m.group(3) or 0) / 3600
            values.append(-value if m.group(4) in "SW" else value)
        return values[0], values[1], None, text[: dms[0].start()] + text[dms[1].end() :]
    dec = list(DECIMAL_RE.finditer(text))
    if len(dec) >= 2:
        values = [float(m.group(1)) * (-1 if m.group(2) in ("S", "W") else 1) for m in dec[:2]]
        return values[0], values[1], None, text[: dec[0].start()] + text[dec[1].end() :]
    raise ReadError(f"no location in {text!r}")


# ---------------------------------------------------------------------------------------- #
# Window and priority
# ---------------------------------------------------------------------------------------- #


def _minutes(text: str) -> int:
    m = re.fullmatch(r"(\d+) (?:hours?|h)(?: (\d+) min)?", text)
    if m:
        return 60 * int(m.group(1)) + int(m.group(2) or 0)
    return int(re.fullmatch(r"(\d+) (?:minutes?|min)", text).group(1))


def _next(issued: datetime, hour: int, minute: int, zone=timezone.utc) -> datetime:
    """The next ``hour:minute`` in ``zone`` at or after ``issued``, in UTC."""
    local = issued.astimezone(zone)
    t = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if t < local:
        t = (t + timedelta(days=1)).replace(hour=hour, minute=minute)
    return t.astimezone(timezone.utc)


def read_window(text: str, issued: datetime, zone: Optional[str]) -> Tuple[datetime, datetime]:
    """``(start, stop)`` in UTC."""
    t = text.lower()
    after = lambda minutes: issued + timedelta(minutes=minutes)  # noqa: E731
    if m := re.search(r"between (\d+) and (\d+) minutes from now", t):
        return after(int(m.group(1))), after(int(m.group(2)))
    if m := re.search(r"from t\+(\d+) to t\+(\d+) min", t):
        return after(int(m.group(1))), after(int(m.group(2)))
    if m := re.search(rf"starting ({DUR}) from now and ending ({DUR}) from now", t):
        return after(_minutes(m.group(1))), after(_minutes(m.group(2)))
    if m := re.search(rf"starting in ({DUR}), for ({DUR})", t):
        start = _minutes(m.group(1))
        return after(start), after(start + _minutes(m.group(2)))
    if m := re.search(rf"starting at {CLOCK} for ({DUR})", t):
        start = _next(issued, int(m.group(1)), int(m.group(2)))
        return start, start + timedelta(minutes=_minutes(m.group(3)))
    if m := re.search(rf"(?:within|in) the next ({DUR})", t):
        return issued, after(_minutes(m.group(1)))
    if m := re.search(rf"(?:any time before|any time until|by) {CLOCK}", t):
        return issued, _next(issued, int(m.group(1)), int(m.group(2)))
    if m := re.search(r"between (\d{1,2}):(\d\d) (am|pm) and (\d{1,2}):(\d\d) (am|pm) local", t):
        if zone is None:
            raise ReadError(f"local time without a named place in {text!r}")
        z = ZoneInfo(zone)
        (h1, m1, p1), (h2, m2, p2) = m.groups()[:3], m.groups()[3:]
        start = _next(issued, int(h1) % 12 + 12 * (p1 == "pm"), int(m1), z)
        stop = _next(start, int(h2) % 12 + 12 * (p2 == "pm"), int(m2), z)
        return start, stop
    if m := re.search(rf"{CLOCK}(?: and |-| to ){CLOCK}", t):
        start = _next(issued, int(m.group(1)), int(m.group(2)))
        return start, _next(start, int(m.group(3)), int(m.group(4)))
    raise ReadError(f"no window in {text!r}")


def read_priority(text: str) -> float:
    """The weight of the tier named by the longest priority phrase in ``text``."""
    phrases = PHRASE_RE.findall(text.lower())
    if not phrases:
        raise ReadError(f"no priority in {text!r}")
    return TIERS[PHRASE_TIER[max(phrases, key=len)]]


def read_request(text: str, issued: datetime, places: Optional[Places] = None) -> Dict:
    """The five numbers of a request: ``lat, lon, start_utc, stop_utc, priority``."""
    lat, lon, zone, rest = read_location(text, places)
    start, stop = read_window(rest, issued, zone)
    return {
        "lat": lat,
        "lon": lon,
        "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stop_utc": stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "priority": read_priority(rest),
    }


# ---------------------------------------------------------------------------------------- #
# Episode
# ---------------------------------------------------------------------------------------- #


def rebuild_state(
    env: WindowedSarSat, key: jax.Array, readings: List[Dict], issued: datetime
) -> WindowedState:
    """The state ``env.reset(key)`` starts, with its targets replaced by ``readings`` (one
    per target slot, in slot order): positions, windows (in steps from ``issued``) and
    weights normalised as the environment normalises them."""
    if len(readings) > env.max_targets:
        raise ValueError(f"{len(readings)} readings but max_targets={env.max_targets}")
    base = jax.jit(env._initial_state)(key)
    m, n = env.max_targets, len(readings)
    latlon = np.zeros((m, 2), np.float32)
    weight = np.zeros((m,), np.float32)
    window = np.stack([np.zeros((m,), np.int32), np.full((m,), env.time_limit, np.int32)], -1)
    for j, r in enumerate(readings):
        latlon[j] = r["lat"], r["lon"]
        weight[j] = r["priority"]
        first, last = steps_from_times(
            parse_iso(r["start_utc"]), parse_iso(r["stop_utc"]), issued, env.step_seconds
        )
        window[j] = max(first, 0), min(last, env.time_limit)
    latlon = jnp.asarray(latlon)
    return base._replace(
        target_latlon=latlon,
        target_pos=EARTH_RADIUS_KM * latlon_to_unit_vector(latlon),
        target_priority=jnp.asarray(weight / weight.sum()),
        target_active=jnp.arange(m) < n,
        target_window=jnp.asarray(window),
    )
