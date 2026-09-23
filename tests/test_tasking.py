"""Tasking requests (sarsat.tasking) carry the raw event numbers and honest tolerances."""

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sarsat.tasking import (
    Target,
    extract_events,
    extract_targets,
    render_request,
    score,
    steps_from_times,
)
from sarsat.tasking.geo import Places, destination, distance_km
from sarsat.tasking.read import read_request, rebuild_state
from sarsat.tasking.render import (
    decimal_wording,
    dms_wording,
    parse_iso,
    relative_options,
    relative_wording,
    render_window,
)
from sarsat.windows import WindowedSarSat

SMALL = dict(num_satellites=20, max_targets=600, hotspots=8, hotspot_targets=25, planes=5)
ISSUED = datetime(2026, 9, 23, 22, 41, tzinfo=timezone.utc)  # windows cross midnight

ZONES = [
    "America/Chicago",
    "America/Chicago",
    "Africa/Nairobi",
    "Arctic/Longyearbyen",
    "Pacific/Fiji",
]
PLACES = Places(
    name=np.array(["Omaha", "Kearney", "Mombasa", "Longyearbyen", "Suva"], dtype=object),
    region=np.array(["Nebraska", "Nebraska", "Kenya", "Svalbard", "Fiji"], dtype=object),
    region_code=np.array(["NE", "NE", "KE", "SJ", "FJ"], dtype=object),
    country_code=np.array(["US", "US", "KE", "SJ", "FJ"], dtype=object),
    lat=np.array([41.2565, 40.6995, -4.0547, 78.2232, -18.1416]),
    lon=np.array([-95.9345, -99.0815, 39.6636, 15.6267, 178.4419]),
    population=np.array([486051.0, 33790.0, 799668.0, 2060.0, 77366.0]),
    timezone=np.array(ZONES, dtype=object),
)


def _points(n: int, seed: int = 0) -> np.ndarray:
    """Random points plus the awkward ones: poles, the dateline, the equator."""
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.uniform(-89, 89, n), rng.uniform(-180, 180, n)], axis=-1)
    edge = [(88.99, 179.99), (-88.99, -179.99), (0.0004, -0.0004), (-0.0004, 179.9999)]
    return np.concatenate([pts, np.array(edge)])


def test_events_are_the_exact_cluster_centres_windows_and_weight() -> None:
    env = WindowedSarSat(**SMALL, hotspot_radius_km=1e-3, hotspot_weight=7.0, window_steps=(10, 30))
    key = jax.random.PRNGKey(11)
    state = jax.jit(env.reset)(key)[0]
    events = extract_events(env, key)
    assert len(events) == env.hotspots
    latlon = np.asarray(state.target_latlon)
    window = np.asarray(state.target_window)
    for e in events:
        members = np.arange(e.cluster, env.hotspots * env.hotspot_targets, env.hotspots)
        assert np.abs(latlon[members] - [e.lat, e.lon]).max() < 1e-3
        assert (window[members] == [e.first, e.last]).all()
        assert e.priority == pytest.approx(7.0, rel=1e-4)
        assert 1 <= e.first and e.last - e.first + 1 in range(10, 31)


@pytest.mark.parametrize("digits", [1, 2, 3, 4])
@pytest.mark.parametrize("style", [0, 1, 2, 3])
def test_decimal_wordings_read_back_within_tolerance(digits: int, style: int) -> None:
    number = r"(-?\d+\.\d+)([NSEW]?)"
    for lat, lon in _points(200):
        w = decimal_wording(lat, lon, digits, style)
        (a, ha), (b, hb) = re.findall(number, w.text)
        read = (float(a) * (-1 if ha == "S" else 1), float(b) * (-1 if hb == "W" else 1))
        assert read == pytest.approx(w.stated, abs=1e-12)
        assert distance_km(lat, lon, *w.stated) <= w.tol


@pytest.mark.parametrize("precision", ["sec", "decmin", "min"])
@pytest.mark.parametrize("ascii_marks", [False, True])
def test_dms_wordings_read_back_within_tolerance(precision: str, ascii_marks: bool) -> None:
    for lat, lon in _points(200, seed=1):
        w = dms_wording(lat, lon, precision, ascii_marks)
        read = []
        for part in w.text.split():
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", part)]
            value = sum(x / 60.0**k for k, x in enumerate(nums))
            read.append(-value if part[-1] in "SW" else value)
        assert tuple(read) == pytest.approx(w.stated, abs=1e-9)
        assert distance_km(lat, lon, *w.stated) <= w.tol


def test_relative_wordings_stay_within_their_tolerance() -> None:
    rng = np.random.default_rng(2)
    seen = set()
    for i in range(len(PLACES.name)):
        for _ in range(300):
            # Points up to ~600 km from the place, across the dateline for Suva.
            d, th = rng.uniform(3, 600), rng.uniform(0, 360)
            lat, lon = destination(PLACES.lat[i], PLACES.lon[i], th, d)
            for style, unit in relative_options(lat, lon, PLACES, i, max_tol_km=25.0):
                w = relative_wording(lat, lon, PLACES, i, style, unit, rng)
                seen.add((style, unit))
                assert w.tol <= 25.0 + 1e-6
                assert distance_km(lat, lon, *w.stated) <= w.tol
    assert len(seen) == 6  # every bearing style in every unit was exercised


def _next_clock(hhmm: str, after: datetime, zone=timezone.utc) -> datetime:
    """The parser's rule: a dateless clock time is its next occurrence at or after ``after``."""
    h, m = int(hhmm[:2]), int(hhmm[-2:])
    local = after.astimezone(zone)
    t = local.replace(hour=h, minute=m, second=0, microsecond=0)
    return (t if t >= local else t + timedelta(days=1)).astimezone(timezone.utc)


def test_window_wordings_are_exact_and_read_back() -> None:
    rng = np.random.default_rng(3)
    forms = set()
    for _ in range(3000):
        length = int(rng.integers(10, 31))
        first = int(rng.integers(1, 180 - length + 2))
        last = first + length - 1
        w = render_window(first, last, ISSUED, rng, "America/Chicago")
        forms.add(w.form)
        start, stop = ISSUED + timedelta(minutes=first), ISSUED + timedelta(minutes=last)
        assert steps_from_times(start, stop, ISSUED) == (first, last)
        errors = [
            abs((a - b).total_seconds()) for a, b in zip(w.stated, (start, stop), strict=True)
        ]
        assert max(errors) <= 60 * w.tol
        if w.form == "utc":
            clocks = re.findall(r"(\d\d):?(\d\d)", w.text)
            read = tuple(_next_clock(h + m, ISSUED) for h, m in clocks)
            assert read == w.stated
        if w.form == "local":
            clocks = re.findall(r"(\d{1,2}):(\d\d) (AM|PM)", w.text)
            zone = ZoneInfo("America/Chicago")
            read = tuple(
                _next_clock(f"{int(h) % 12 + (12 if p == 'PM' else 0):02d}{m}", ISSUED, zone)
                for h, m, p in clocks
            )
            assert read == w.stated
        if w.text.startswith("from T+"):
            a, b = map(int, re.findall(r"T\+(\d+)", w.text))
            assert (a, b) == (first, last)
    assert forms >= {"relative", "relative_duration", "utc", "utc_duration", "local", "within"}


def test_requests_are_reproducible_and_their_readings_score_ok() -> None:
    env = WindowedSarSat(**SMALL, window_steps=(10, 30), background_windows=False)
    events = extract_events(env, jax.random.PRNGKey(5))
    for e in events:
        r1 = render_request(e, ISSUED, np.random.default_rng([5, e.cluster]), PLACES)
        r2 = render_request(e, ISSUED, np.random.default_rng([5, e.cluster]), PLACES)
        assert r1 == r2
        assert r1["label"]["priority"] == pytest.approx(10.0, rel=1e-4)
        start, stop = parse_iso(r1["label"]["start_utc"]), parse_iso(r1["label"]["stop_utc"])
        assert steps_from_times(start, stop, parse_iso(r1["issued_utc"])) == tuple(r1["steps"])
        pred = dict(r1["stated"], priority=r1["label"]["priority"])
        assert score(pred, r1)["ok"]
        far = dict(pred, lat=pred["lat"] + 1.0)
        assert not score(far, r1)["ok"]


def test_load_places_names_us_states() -> None:
    pytest.importorskip("geonamescache")
    from sarsat.tasking import load_places

    places = load_places(15000)
    idx, dist = places.nearby(41.2565, -95.9345, 20.0)
    names = {places.label(int(i)) for i in idx}
    assert "Omaha, Nebraska" in names and dist[0] < 20.0
    assert places.label(int(idx[list(places.name[idx]).index("Omaha")]), True) == "Omaha, NE"


def test_targets_cover_every_slot_with_their_own_window_and_weight() -> None:
    env = WindowedSarSat(**SMALL, window_steps=(10, 30), background_windows=False)
    key = jax.random.PRNGKey(5)
    targets = extract_targets(env, key)
    events = {e.cluster: e for e in extract_events(env, key)}
    assert [t.slot for t in targets] == list(range(env.max_targets))
    for t in targets:
        if t.cluster < 0:
            assert (t.first, t.last, t.priority) == (0, env.time_limit, pytest.approx(1.0))
        else:
            e = events[t.cluster]
            assert (t.first, t.last) == (e.first, e.last)
            assert t.priority == pytest.approx(env.hotspot_weight, rel=1e-4)


def _near_places(n: int, seed: int) -> list:
    """Targets scattered up to 600 km around the fixture places, with random windows."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        i = k % len(PLACES)
        lat, lon = destination(
            PLACES.lat[i], PLACES.lon[i], rng.uniform(0, 360), rng.uniform(0, 600) ** 1.5 / 24
        )
        first = int(rng.choice([0, 1, int(rng.integers(2, 150))]))
        last = 180 if first == 0 else first + int(rng.integers(9, 30))
        out.append(
            Target(k, -1 if first == 0 else 0, lat, lon, first, last, [1.0, 10.0][first > 0])
        )
    return out


def test_every_wording_reads_back_to_its_exact_reading() -> None:
    loc_forms, time_forms = set(), set()
    targets = _near_places(3000, seed=7)
    for places in (PLACES, None):
        for t in targets[: 3000 if places else 600]:
            rng = np.random.default_rng([7, t.slot])
            r = render_request(t, ISSUED, rng, places)
            p = read_request(r["text"], ISSUED, places)
            loc_forms.add(r["meta"]["loc_form"])
            time_forms.add(r["meta"]["time_form"])
            assert (p["lat"], p["lon"]) == pytest.approx(
                (r["stated"]["lat"], r["stated"]["lon"]), abs=1e-5
            ), r["text"]
            assert (p["start_utc"], p["stop_utc"]) == (
                r["stated"]["start_utc"],
                r["stated"]["stop_utc"],
            ), r["text"]
            assert p["priority"] == pytest.approx(t.priority), r["text"]
            assert score(p, r)["ok"], r["text"]
    assert loc_forms >= {"named", "relative_compass", "relative_bearing", "dms_sec", "decimal2"}
    assert time_forms >= {"relative", "utc", "utc_duration", "local", "within", "until"}


def test_ambiguous_place_labels_are_never_used() -> None:
    two = Places(
        name=np.array(["Springfield", "Springfield", "Kearney"], dtype=object),
        region=np.array(["Illinois", "Illinois", "Nebraska"], dtype=object),
        region_code=np.array(["IL", "IL", "NE"], dtype=object),
        country_code=np.array(["US", "US", "US"], dtype=object),
        lat=np.array([39.80, 39.90, 40.70]),
        lon=np.array([-89.64, -89.60, -99.08]),
        population=np.array([1e5, 1e5, 3e4]),
        timezone=np.array(["America/Chicago"] * 3, dtype=object),
    )
    assert two.lookup("Springfield", "Illinois") is None
    assert two.lookup("Kearney", "NE") is not None
    rng = np.random.default_rng(0)
    for _ in range(300):
        lat, lon = destination(39.85, -89.62, rng.uniform(0, 360), rng.uniform(0, 60))
        t = Target(0, -1, lat, lon, 0, 180, 1.0)
        assert "Springfield" not in render_request(t, ISSUED, rng, two)["text"]


def test_rebuilt_episode_matches_the_original() -> None:
    env = WindowedSarSat(**SMALL, window_steps=(10, 30), background_windows=False)
    key = jax.random.PRNGKey(9)
    records = [
        render_request(t, ISSUED, np.random.default_rng([9, t.slot]), PLACES)
        for t in extract_targets(env, key)
    ]
    readings = [read_request(r["text"], ISSUED, PLACES) for r in records]
    original = jax.jit(env.reset)(key)[0]
    rebuilt = rebuild_state(env, key, readings, ISSUED)
    km = distance_km(*np.asarray(original.target_latlon, np.float64).T,
                     *np.asarray(rebuilt.target_latlon, np.float64).T)  # fmt: skip
    assert (km <= np.array([r["tol"]["km"] for r in records]) + 1e-3).all()
    window_err = np.abs(np.asarray(original.target_window) - np.asarray(rebuilt.target_window))
    assert (window_err.max(axis=1) <= np.array([r["tol"]["min"] for r in records])).all()
    assert np.allclose(original.target_priority, rebuilt.target_priority, rtol=1e-5)
    assert jnp.array_equal(original.orbit.raan, rebuilt.orbit.raan)
    action = jnp.zeros(env.action_spec.shape).at[:, -1].set(1.0)
    _, timestep = jax.jit(env.step)(rebuilt, action)  # the rebuilt state is a valid state
    assert np.isfinite(np.asarray(timestep.reward)).all()
