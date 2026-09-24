# ruff: noqa: E501
"""Measure the parser: a synthetic set built from the gazetteer plus hand-written phrases.

    PYTHONPATH=. live/.venv/Scripts/python -m live.nl.evaluate --gazetteer data/geonames \\
        [--model models/qwen2.5-1.5b-instruct-q4_k_m.gguf] [--n 300]

A case passes when the location lands within 50 km of the truth and priority and deadline
match exactly. Reports the pass rate per field and per source (rules / gazetteer / model),
and prints the failures so the templates or the prompt can be fixed.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

from live.nl import parse
from live.nl.geocode import Gazetteer, Place, offset_point

PRIORITY_PHRASES = {
    "high": ["high priority", "urgent", "top priority", "critical", "priority 1", "emergency"],
    "low": ["low priority", "routine", "when convenient", "no rush", "not urgent", "opportunistic"],
    "normal": ["", "normal priority", "standard priority", ""],
}
DEADLINE_PHRASES = [
    ("within {n} minutes", lambda n: n),
    ("in the next {n} min", lambda n: n),
    ("{n} minute window", lambda n: n),
    ("deadline {n} minutes", lambda n: n),
    ("no later than {n} min", lambda n: n),
    ("within half an hour", lambda n: 30),
    ("asap", lambda n: 5),
    ("", lambda n: 30),
]
TEMPLATES = [
    "image {place}{sep}{prio}{sep}{dl}",
    "please collect {place} {prio} {dl}",
    "{prio}: SAR pass over {place} {dl}",
    "task a satellite on the port of {place}, {prio}, {dl}",
    "we need {place} imaged {dl}, {prio}",
    "{place} {dl} {prio}",
    "get me a scene of {place} ({prio}) {dl}",
    "observe {place} {dl}; {prio}",
]
OFFSET_TEMPLATES = [
    "image {km} km {bearing} of {place} {prio} {dl}",
    "collect the area {km} km {bearing} of {place}, {prio}, {dl}",
]
HANDWRITTEN: List[Tuple[str, Dict]] = [
    ("Image the port of Rotterdam, high priority, within 20 minutes", dict(place="Rotterdam", priority="high", deadline=20)),
    ("urgent: we need Tokyo imaged asap", dict(place="Tokyo", priority="high", deadline=5)),
    ("routine collection over Nairobi in the next half hour", dict(place="Nairobi", priority="low", deadline=30)),
    ("SAR scene of Reykjavik please, no rush", dict(place="Reykjavik", priority="low", deadline=30)),
    ("get Buenos Aires within 15 minutes", dict(place="Buenos Aires", priority="normal", deadline=15)),
    ("Take a look at Cape Town, critical, 10 minute window", dict(place="Cape Town", priority="high", deadline=10)),
    ("Perth, Australia, low priority, within 25 minutes", dict(place="Perth", priority="low", deadline=25)),
    ("image 51.5, -0.12 in 12 minutes, high priority", dict(lat=51.5, lon=-0.12, priority="high", deadline=12)),
    ("lat 35.7 lon 139.7 normal priority within 30 min", dict(lat=35.7, lon=139.7, priority="normal", deadline=30)),
    ("100 km north of Lima, urgent, within 20 minutes", dict(place="Lima", offset=(100, "north"), priority="high", deadline=20)),
    ("collect Mumbai", dict(place="Mumbai", priority="normal", deadline=30)),
    ("Anchorage within 2 hours", dict(place="Anchorage", priority="normal", deadline=30)),
    ("emergency imaging of Manila within the next 8 minutes", dict(place="Manila", priority="high", deadline=8)),
    ("we'd like Seville (Spain) imaged, standard priority, 30 min", dict(place="Sevilla", priority="normal", deadline=30)),
    ("São Paulo now", dict(place="Sao Paulo", priority="normal", deadline=5)),
]  # fmt: skip


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def synthetic(
    gaz: Gazetteer, n: int, seed: int = 0, min_population: int = 100_000
) -> List[Tuple[str, Dict]]:
    rng = random.Random(seed)
    cities = [p for p in gaz.places if p.kind == "city" and p.population >= min_population]
    cases = []
    for _ in range(n):
        p = rng.choice(cities)
        prio = rng.choice(list(PRIORITY_PHRASES))
        prio_phrase = rng.choice(PRIORITY_PHRASES[prio])
        dl_tpl, dl_fn = rng.choice(DEADLINE_PHRASES)
        minutes = rng.choice([5, 8, 10, 12, 15, 20, 25, 30])
        dl_phrase, deadline = dl_tpl.format(n=minutes), dl_fn(minutes)
        expected = dict(place=p.name, lat=p.lat, lon=p.lon, priority=prio, deadline=deadline)
        if rng.random() < 0.15:
            km, bearing = (
                rng.choice([20, 50, 80, 120]),
                rng.choice(["north", "south", "east", "west"]),
            )
            text = rng.choice(OFFSET_TEMPLATES).format(
                km=km, bearing=bearing, place=p.name, prio=prio_phrase, dl=dl_phrase
            )
            lat, lon = offset_point(p.lat, p.lon, km, bearing)
            expected.update(lat=lat, lon=lon, offset=(km, bearing))
        else:
            sep = rng.choice([", ", " ", " - "])
            text = rng.choice(TEMPLATES).format(
                place=p.name, prio=prio_phrase, dl=dl_phrase, sep=sep
            )
        cases.append((" ".join(text.split()).strip(" ,;:-"), expected))
    return cases


def resolve_expected(
    gaz: Optional[Gazetteer], exp: Dict
) -> Tuple[Optional[float], Optional[float]]:
    if "lat" in exp:
        return exp["lat"], exp["lon"]
    if gaz is None:
        return None, None
    hit: Optional[Place] = gaz.lookup(exp["place"])
    if hit is None:
        return None, None
    lat, lon = hit.lat, hit.lon
    if exp.get("offset"):
        lat, lon = offset_point(lat, lon, *exp["offset"])
    return lat, lon


def run(
    cases: List[Tuple[str, Dict]],
    gaz: Optional[Gazetteer],
    model,
    verbose: bool = True,
    prefer_model: bool = False,
) -> Dict[str, float]:
    ok = {"location": 0, "priority": 0, "deadline": 0, "all": 0}
    by_source: Dict[str, List[int]] = {}
    failures = []
    t0 = time.time()
    for text, exp in cases:
        p = parse(text, geocoder=gaz, model=model, prefer_model=prefer_model)
        lat, lon = resolve_expected(gaz, exp)
        loc = p.located and lat is not None and haversine_km(p.lat, p.lon, lat, lon) <= 50
        pri = p.priority == exp["priority"]
        dl = p.deadline_minutes == exp["deadline"]
        ok["location"] += loc
        ok["priority"] += pri
        ok["deadline"] += dl
        ok["all"] += loc and pri and dl
        by_source.setdefault(p.source, []).append(int(loc and pri and dl))
        if not (loc and pri and dl):
            failures.append((text, exp, p))
    n = len(cases)
    rates = {k: v / n for k, v in ok.items()}
    if verbose:
        print(f"{n} cases in {time.time() - t0:.1f}s")
        for k, v in rates.items():
            print(f"  {k:9s} {100 * v:5.1f}%")
        for src, hits in by_source.items():
            print(f"  source {src:12s} n={len(hits):4d} pass {100 * sum(hits) / len(hits):5.1f}%")
        for text, exp, p in failures[:40]:
            print(
                f"  FAIL {text!r}\n       expected {exp}\n       got place={p.place!r} ({p.lat}, {p.lon}) "
                f"{p.priority} {p.deadline_minutes} via {p.source} {p.notes}"
            )
    return rates


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gazetteer", required=True)
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int)
    ap.add_argument(
        "--model-first",
        action="store_true",
        help="ask the model before the sentence search, to measure the fallback path",
    )
    args = ap.parse_args()
    gaz = Gazetteer(args.gazetteer)
    model = None
    if args.model:
        from live.nl.llm import LocalModel

        model = LocalModel(args.model, n_threads=args.threads)
    print("hand-written:")
    run(HANDWRITTEN, gaz, model, prefer_model=args.model_first)
    print("synthetic:")
    run(synthetic(gaz, args.n, args.seed), gaz, model, prefer_model=args.model_first)


if __name__ == "__main__":
    main()
