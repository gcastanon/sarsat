"""Natural-language tasking requests written from an event's raw numbers.

The label of a request is always the raw event (:class:`sarsat.tasking.events.Event`):
centre latitude / longitude, window start / stop, priority. The text is written *from*
those numbers, and every wording carries

* ``stated`` -- what an exact reading of the text gives (the rounded coordinates, the
  point 40 km NW of the named place, the clock times), and
* ``tol`` -- a bound on how far ``stated`` can be from the label: ``km`` for the location,
  ``min`` for the window.

A parser is then scored against the raw label within ``tol``; ``stated`` separates what
the wording lost from what the parser got wrong.

Conventions a parser must share (they make every time wording exact):

* The request is issued at ``issued`` = step 0 of the episode (every window is known at
  reset, as it is to ``coop_plan``), on a whole minute. Step ``s`` is ``issued + s *
  step_seconds``; ``start`` is the window's first step and ``stop`` its last, inclusive.
* A clock time without a date is its next occurrence at or after ``issued``; local times
  are in the time zone of the place the location names.
* "Within the next N minutes" opens at ``issued``, a minute before step 1 (``tol`` 1 min).
* Distances are great-circle on the environment's sphere; a bearing is the initial true
  bearing from the named place; a compass point is the centre of its 22.5 degree sector.
* Priority is written as one of four tiers (``TIERS``); the label keeps the raw weight.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, NamedTuple, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from sarsat.tasking.events import Event
from sarsat.tasking.geo import (
    COMPASS16,
    KM_PER_DEG,
    KM_PER_UNIT,
    Places,
    bearing_deg,
    destination,
    distance_km,
)

TIERS = {"routine": 1.0, "priority": 3.0, "immediate": 10.0, "flash": 30.0}

COMPASS_WORDS = {
    "N": "north", "NNE": "north-northeast", "NE": "northeast", "ENE": "east-northeast",
    "E": "east", "ESE": "east-southeast", "SE": "southeast", "SSE": "south-southeast",
    "S": "south", "SSW": "south-southwest", "SW": "southwest", "WSW": "west-southwest",
    "W": "west", "WNW": "west-northwest", "NW": "northwest", "NNW": "north-northwest",
}  # fmt: skip
UNIT_WORDS = {
    "km": ("km", "kilometers"),
    "mi": ("mi", "miles"),
    "nmi": ("nm", "nautical miles"),
}

# Prefix phrases open a request ("URGENT: ..."), suffix phrases close one ("..., this is urgent").
PRIORITY_PHRASES = {
    "routine": (
        ("Routine", "Low priority", "No rush", "Routine tasking"),
        ("no rush", "routine priority", "whenever it fits", "low priority"),
    ),
    "priority": (
        ("Priority", "Elevated priority", "Moderately urgent", "Priority tasking"),
        ("elevated priority", "fairly important", "priority please", "moderately urgent"),
    ),
    "immediate": (
        ("High priority", "URGENT", "Urgent", "Priority IMMEDIATE", "Time-critical",
         "Top priority", "Rush", "Immediate"),
        ("this is urgent", "high priority", "treat as top priority", "time-critical",
         "please expedite", "urgent", "immediate priority"),
    ),
    "flash": (
        ("FLASH", "Flash priority", "Emergency", "Drop everything"),
        ("flash priority", "this is an emergency", "highest priority available",
         "drop everything else"),
    ),
}  # fmt: skip

VERBS = (
    "image",
    "take a picture of",
    "get a SAR image of",
    "collect on",
    "get a SAR look at",
    "task a pass over",
    "get eyes on",
    "grab an image of",
)

# {P} priority prefix, {S} priority suffix, {V} verb, {L} location, {T} time; "^" capitalises.
TEMPLATES = (
    "{P}: {V} {L} {T}.",
    "{P} - please {V} {L} {T}.",
    "Please {V} {L} {T}, {S}.",
    "Need you to {V} {L} {T}; {S}.",
    "Can we {V} {L} {T}? {^S}.",
    "{^T}, {V} {L}. {P}.",
    "{P} request: {L}, {T}.",
    "{^V} {L} {T} - {S}.",
    "{P!upper} // SAR COLLECT // {L} // {T!upper}",
)


class Wording(NamedTuple):
    text: str
    form: str
    tol: float  # km for a location, minutes for a window
    stated: Tuple  # (lat, lon) or (start, stop): what an exact reading gives


# ---------------------------------------------------------------------------------------- #
# Location
# ---------------------------------------------------------------------------------------- #


def _hemi(value: float, pos: str, neg: str) -> str:
    return pos if value >= 0 else neg


def _rounding_tol(lat: float, half_deg: float) -> float:
    """Bound [km] on the error of rounding lat and lon each by at most ``half_deg``."""
    cos_lat = math.cos(math.radians(max(abs(lat) - half_deg, 0.0)))  # widest the lon step gets
    return 1.01 * KM_PER_DEG * half_deg * math.sqrt(1.0 + cos_lat**2) + 1e-3


def decimal_wording(lat: float, lon: float, digits: int, style: int) -> Wording:
    """Decimal degrees rounded to ``digits``; ``style`` 0-3 picks the notation."""
    la, lo = f"{abs(lat):.{digits}f}", f"{abs(lon):.{digits}f}"
    s_lat = float(la) * (1 if lat >= 0 else -1)
    s_lon = float(lo) * (1 if lon >= 0 else -1)
    if style == 0:
        text = f"{lat:.{digits}f}, {lon:.{digits}f}"
    elif style == 1:
        text = f"{la}{_hemi(lat, 'N', 'S')} {lo}{_hemi(lon, 'E', 'W')}"
    elif style == 2:
        text = f"lat {lat:.{digits}f}, lon {lon:.{digits}f}"
    else:
        width = digits + 4  # three integer digits, the point, the decimals
        text = f"{la}{_hemi(lat, 'N', 'S')} {lo.zfill(width)}{_hemi(lon, 'E', 'W')}"
    tol = _rounding_tol(lat, 0.5 * 10.0**-digits)
    return Wording(text, f"decimal{digits}", tol, (s_lat, s_lon))


def _sexagesimal(value: float, unit_s: float) -> Tuple[int, int, float]:
    """Degrees, minutes and seconds of ``abs(value)`` rounded to ``unit_s`` seconds."""
    total = round(abs(value) * 3600.0 / unit_s) * unit_s
    deg, rem = divmod(total, 3600.0)
    minutes, seconds = divmod(rem, 60.0)
    return int(deg), int(minutes), seconds


def dms_wording(lat: float, lon: float, precision: str, ascii_marks: bool) -> Wording:
    """Degrees-minutes(-seconds): ``precision`` is ``sec``, ``decmin`` or ``min``."""
    unit_s = {"sec": 1.0, "decmin": 6.0, "min": 60.0}[precision]
    deg_m, min_m, sec_m = ("d", "'", '"') if ascii_marks else ("°", "′", "″")  # noqa: RUF001
    parts, stated = [], []
    for value, pos, neg, width in ((lat, "N", "S", 2), (lon, "E", "W", 3)):
        d, m, s = _sexagesimal(value, unit_s)
        if precision == "sec":
            body = f"{d:0{width}d}{deg_m}{m:02d}{min_m}{int(s):02d}{sec_m}"
        elif precision == "decmin":
            body = f"{d:0{width}d}{deg_m}{m + s / 60.0:04.1f}{min_m}"
        else:
            body = f"{d:0{width}d}{deg_m}{m:02d}{min_m}"
        parts.append(body + _hemi(value, pos, neg))
        stated.append((d + m / 60.0 + s / 3600.0) * (1 if value >= 0 else -1))
    tol = _rounding_tol(lat, 0.5 * unit_s / 3600.0)
    return Wording(" ".join(parts), f"dms_{precision}", tol, tuple(stated))


def _distance_step(dist: float) -> float:
    """Rounding step of a spoken distance, in the distance's own unit."""
    return 1.0 if dist < 20 else 5.0 if dist < 200 else 10.0


def relative_options(
    lat: float, lon: float, places: Places, i: int, max_tol_km: float
) -> List[Tuple[str, str]]:
    """``(bearing style, unit)`` pairs whose wording from place ``i`` stays within
    ``max_tol_km``: the bound is half the distance step plus the distance times the bearing's
    half-width (the arc swept at that range), which holds on the sphere too."""
    d_km = float(distance_km(places.lat[i], places.lon[i], lat, lon))
    options = []
    for style, half_deg in (("compass", 11.25), ("bearing", 0.5)):
        for unit, km in KM_PER_UNIT.items():
            half_step_km = 0.5 * _distance_step(d_km / km) * km
            if half_step_km + d_km * math.radians(half_deg) <= max_tol_km:
                options.append((style, unit))
    return options


def relative_wording(
    lat: float, lon: float, places: Places, i: int, style: str, unit: str, rng
) -> Wording:
    """``40 km NW of Kearney, Nebraska`` or ``Kearney, Nebraska, bearing 312, 40 km``."""
    a_lat, a_lon = float(places.lat[i]), float(places.lon[i])
    d_km = float(distance_km(a_lat, a_lon, lat, lon))
    theta = float(bearing_deg(a_lat, a_lon, lat, lon))
    km = KM_PER_UNIT[unit]
    step = _distance_step(d_km / km)
    dist = round(d_km / km / step) * step
    dist_text = f"{dist:.0f} {UNIT_WORDS[unit][rng.integers(2)]}"
    place = places.label(i, abbreviate=bool(rng.integers(2)))
    if style == "compass":
        point = round(theta / 22.5) % 16
        s_theta, half_deg = point * 22.5, 11.25
        name = COMPASS16[point] if rng.random() < 0.6 else COMPASS_WORDS[COMPASS16[point]]
        text = f"{dist_text} {name} of {place}"
    else:
        s_theta, half_deg = float(round(theta) % 360), 0.5
        text = (
            f"a point {dist_text} from {place} on a bearing of {s_theta:03.0f} degrees"
            if rng.random() < 0.5
            else f"{place}, bearing {s_theta:03.0f}, {dist_text}"
        )
    if style == "compass" and rng.random() < 0.4:
        text = "the area " + text
    tol = 0.5 * step * km + d_km * math.radians(half_deg) + 1e-3
    return Wording(text, f"relative_{style}", tol, destination(a_lat, a_lon, s_theta, dist * km))


def named_wording(places: Places, i: int, radius_km: float, rng) -> Wording:
    """The place itself, for a centre within ``radius_km`` of it."""
    place = places.label(i, abbreviate=bool(rng.integers(2)))
    text = (place, f"near {place}", f"the {places.name[i]} area, {places.region[i]}")[
        rng.choice(3, p=(0.5, 0.3, 0.2))
    ]
    return Wording(text, "named", radius_km, (float(places.lat[i]), float(places.lon[i])))


def render_location(
    lat: float,
    lon: float,
    rng: np.random.Generator,
    places: Optional[Places] = None,
    max_tol_km: float = 25.0,
    named_km: float = 10.0,
    anchor_km: float = 600.0,
) -> Tuple[Wording, Optional[str]]:
    """A wording of ``(lat, lon)`` and the IANA time zone of the place it names, if any.

    Half the time a place is used: in 15% of all requests the nearest place itself (only if
    within ``named_km``), otherwise a wording relative to a place within ``anchor_km``
    (larger places likelier; only wordings within ``max_tol_km``). The rest, and remote
    centres with no usable place, are coordinates, as a real request would be.
    """
    u = rng.random()
    if places is not None and u < 0.5:
        idx, dist = places.nearby(lat, lon, anchor_km)
        if u < 0.15 and len(idx) and dist[0] <= named_km:
            i = int(idx[0])
            return named_wording(places, i, named_km, rng), str(places.timezone[i])
        keep = dist >= 3.0  # "0 km N of" reads as the place itself
        idx = idx[keep][:40]
        if len(idx):
            weight = np.sqrt(places.population[idx]) + 1.0  # some GeoNames entries are 0
            for i in rng.choice(idx, size=len(idx), replace=False, p=weight / weight.sum()):
                options = relative_options(lat, lon, places, int(i), max_tol_km)
                if options:
                    style, unit = options[rng.integers(len(options))]
                    wording = relative_wording(lat, lon, places, int(i), style, unit, rng)
                    return wording, str(places.timezone[i])
    if rng.random() < 0.6:
        digits = int(rng.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.3, 0.2]))
        return decimal_wording(lat, lon, digits, int(rng.integers(4))), None
    precision = str(rng.choice(["sec", "decmin", "min"]))
    return dms_wording(lat, lon, precision, bool(rng.integers(2))), None


# ---------------------------------------------------------------------------------------- #
# Window
# ---------------------------------------------------------------------------------------- #


def _minutes(m: int, rng) -> str:
    if m < 60 or rng.random() < 0.5:
        return f"{m} {'min' if rng.random() < 0.4 else 'minutes' if m != 1 else 'minute'}"
    h, mm = divmod(m, 60)
    hours = f"{h} hour{'s' if h > 1 else ''}" if rng.random() < 0.5 else f"{h} h"
    return hours if mm == 0 else f"{hours} {mm} min"


def _clock_utc(t: datetime, style: int) -> str:
    return (f"{t:%H:%M}Z", f"{t:%H%M}Z", f"{t:%H:%M} UTC")[style]


def _clock_local(t: datetime) -> str:
    return f"{t.hour % 12 or 12}:{t:%M} {'AM' if t.hour < 12 else 'PM'}"


def render_window(
    first: int,
    last: int,
    issued: datetime,
    rng: np.random.Generator,
    zone_name: Optional[str] = None,
    step_seconds: float = 60.0,
) -> Wording:
    """A wording of the window ``[first, last]`` for a request issued at step 0 (``issued``
    must be timezone-aware); ``zone_name`` enables local clock times."""
    if issued.tzinfo is None:
        raise ValueError("issued must be timezone-aware (UTC)")
    start = issued + timedelta(seconds=first * step_seconds)
    stop = issued + timedelta(seconds=last * step_seconds)
    a, b = round(first * step_seconds / 60), round(last * step_seconds / 60)
    exact = (start, stop)
    forms = ["relative", "relative_duration", "utc", "utc_duration"]
    zone = None
    if zone_name is not None:
        try:
            zone = ZoneInfo(zone_name)
            forms.append("local")
        except (ZoneInfoNotFoundError, ValueError):
            pass
    if first * step_seconds <= 60:
        forms.append("within")
    form = forms[rng.integers(len(forms))]

    if form == "relative":
        text = (
            f"between {a} and {b} minutes from now",
            f"from T+{a} to T+{b} min",
            f"starting {_minutes(a, rng)} from now and ending {_minutes(b, rng)} from now",
        )[rng.integers(3)]
    elif form == "relative_duration":
        text = f"starting in {_minutes(a, rng)}, for {_minutes(b - a, rng)}"
    elif form == "utc":
        style = int(rng.integers(3))
        s, e = _clock_utc(start, style), _clock_utc(stop, style)
        date = f" on {start:%d %b}" if rng.random() < 0.3 else ""
        text = (f"between {s} and {e}{date}", f"{s}-{e}{date}", f"from {s} to {e}{date}")[
            rng.integers(3)
        ]
    elif form == "utc_duration":
        text = f"starting at {_clock_utc(start, int(rng.integers(3)))} for {_minutes(b - a, rng)}"
    elif form == "local":
        s, e = start.astimezone(zone), stop.astimezone(zone)
        abbr = f"{s:%Z}"
        suffix = f" ({abbr})" if abbr.isalpha() and rng.random() < 0.5 else ""
        text = f"between {_clock_local(s)} and {_clock_local(e)} local time{suffix}"
    else:  # within: opens at issue, a step early
        text = (f"within the next {_minutes(b, rng)}", f"in the next {_minutes(b, rng)}")[
            rng.integers(2)
        ]
        return Wording(text, form, 1.0, (issued, stop))
    return Wording(text, form, 0.0, exact)


# ---------------------------------------------------------------------------------------- #
# Priority and the whole request
# ---------------------------------------------------------------------------------------- #


def priority_tier(weight: float) -> str:
    """The tier whose weight is nearest ``weight`` on a log scale."""
    return min(TIERS, key=lambda t: abs(math.log(TIERS[t]) - math.log(max(weight, 1e-9))))


def _fill(template: str, fields: Dict[str, str]) -> str:
    out = template
    for key, value in fields.items():
        out = out.replace("{^" + key + "}", value[:1].upper() + value[1:])
        out = out.replace("{" + key + "!upper}", value.upper())
        out = out.replace("{" + key + "}", value)
    return out


def steps_from_times(
    start: datetime, stop: datetime, issued: datetime, step_seconds: float = 60.0
) -> Tuple[int, int]:
    """Inverse of the window convention: ``(first, last)`` steps for ``start`` / ``stop``."""
    first = (start - issued).total_seconds() / step_seconds
    last = (stop - issued).total_seconds() / step_seconds
    return round(first), round(last)


def render_request(
    event: Event,
    issued: datetime,
    rng: np.random.Generator,
    places: Optional[Places] = None,
    step_seconds: float = 60.0,
    max_tol_km: float = 25.0,
) -> dict:
    """One JSON-ready request record for ``event``: text, raw label, tolerances, readings."""
    loc, zone = render_location(event.lat, event.lon, rng, places, max_tol_km=max_tol_km)
    win = render_window(event.first, event.last, issued, rng, zone, step_seconds)
    tier = priority_tier(event.priority)
    prefixes, suffixes = PRIORITY_PHRASES[tier]
    template = TEMPLATES[rng.integers(len(TEMPLATES))]
    text = _fill(
        template,
        {
            "P": prefixes[rng.integers(len(prefixes))],
            "S": suffixes[rng.integers(len(suffixes))],
            "V": VERBS[rng.integers(len(VERBS))],
            "L": loc.text,
            "T": win.text,
        },
    )
    start = issued + timedelta(seconds=event.first * step_seconds)
    stop = issued + timedelta(seconds=event.last * step_seconds)
    return {
        "text": text,
        "issued_utc": _iso(issued),
        "label": {
            "lat": event.lat,
            "lon": event.lon,
            "start_utc": _iso(start),
            "stop_utc": _iso(stop),
            "priority": event.priority,
        },
        "steps": [event.first, event.last],
        "tol": {"km": round(math.ceil(loc.tol * 10) / 10, 1), "min": win.tol},
        "stated": {
            "lat": round(loc.stated[0], 6),
            "lon": round(loc.stated[1], 6),
            "start_utc": _iso(win.stated[0]),
            "stop_utc": _iso(win.stated[1]),
        },
        "meta": {
            "cluster": event.cluster,
            "loc_form": loc.form,
            "time_form": win.form,
            "tier": tier,
            "template": TEMPLATES.index(template),
        },
    }


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _minutes_apart(a: str, b: str) -> float:
    return abs((parse_iso(a) - parse_iso(b)).total_seconds()) / 60.0


def score(pred: dict, record: dict) -> dict:
    """Errors of a parsed ``pred`` (``lat, lon, start_utc, stop_utc, priority``) against a
    record's raw label, and whether each is within the record's tolerance."""
    label = record["label"]
    km = float(distance_km(pred["lat"], pred["lon"], label["lat"], label["lon"]))
    start_min = _minutes_apart(pred["start_utc"], label["start_utc"])
    stop_min = _minutes_apart(pred["stop_utc"], label["stop_utc"])
    tier_ok = priority_tier(pred["priority"]) == priority_tier(label["priority"])
    ok = km <= record["tol"]["km"] and max(start_min, stop_min) <= record["tol"]["min"]
    return {
        "km": km,
        "start_min": start_min,
        "stop_min": stop_min,
        "priority_ok": tier_ok,
        "ok": ok and tier_ok,
    }
