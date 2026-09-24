"""Offline geocoding from a GeoNames extract (no network, no API).

Point ``Gazetteer`` at a directory holding any of the GeoNames dumps (download once from
https://download.geonames.org/export/dump/, CC BY 4.0):

* ``cities15000.txt`` (or ``cities5000.txt`` / ``cities500.txt``): populated places;
* ``countryInfo.txt``: countries, resolved to their capital;
* ``admin1CodesASCII.txt``: states and provinces, resolved to their most populous city.

Lookups go exact-first (names, ASCII names and alternate names, diacritics stripped), then
fuzzy (``rapidfuzz``), with population breaking ties and an optional ``", Country"`` suffix
narrowing the search. :meth:`Gazetteer.find_in_text` picks the place out of a whole request
sentence when no language model is available.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

_BEARINGS = {"n": 0, "north": 0, "ne": 45, "northeast": 45, "north-east": 45, "e": 90, "east": 90}
_BEARINGS |= {"se": 135, "southeast": 135, "south-east": 135, "s": 180, "south": 180}
_BEARINGS |= {"sw": 225, "southwest": 225, "south-west": 225, "w": 270, "west": 270}
_BEARINGS |= {"nw": 315, "northwest": 315, "north-west": 315}
_BEARINGS |= {"nne": 22.5, "ene": 67.5, "ese": 112.5, "sse": 157.5}
_BEARINGS |= {"ssw": 202.5, "wsw": 247.5, "wnw": 292.5, "nnw": 337.5}
EARTH_RADIUS_KM = 6371.0

# Words that carry no place information in a tasking request.
_FILLER = re.compile(
    r"\b(please|image|imaging|collect|collection|capture|take|get|grab|shoot|look at|look|observe|"
    r"observation|acquire|acquisition|task|tasking|request|sar|picture|photo|scene|pass over|pass|"
    r"of|the|a|an|at|on|over|near|around|in|to|for|with|and|city|town|port|harbour|harbor|"
    r"airport|area|region|vicinity|downtown|centre|center|asap|urgent(?:ly)?|priority|high|low|"
    r"normal|medium|critical|routine|emergency|within|minutes?|mins?|hours?|hrs?|next|deadline|"
    r"by|before|window|max(?:imum)?|now|immediately|possible|soon|as|km|kilomet(?:er|re)s?|"
    r"miles?|mi|north|south|east|west|northeast|northwest|southeast|southwest|"
    r"n|s|e|w|ne|nw|se|sw)\b",
    re.I,
)
_OFFSET = re.compile(
    r"(\d+(?:\.\d+)?)\s*(km|kilomet(?:er|re)s?|mi|miles?)\s+"
    r"(north|south|east|west|northeast|northwest|southeast|southwest|north-east|north-west|"
    r"south-east|south-west|n|s|e|w|ne|nw|se|sw)\s+(?:of|from)\s+",
    re.I,
)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def _squeeze(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    country: str
    population: int
    kind: str  # city | country | admin1
    score: float = 1.0


def find_offset(text: str) -> Optional[Tuple[float, str]]:
    """``(km, bearing)`` from "N km <bearing> of ..." in ``text``, or ``None``."""
    m = _OFFSET.search(text)
    if not m:
        return None
    km = float(m.group(1)) * (1.609344 if m.group(2).lower().startswith("mi") else 1.0)
    return km, m.group(3).lower()


def offset_point(lat: float, lon: float, km: float, bearing: str) -> Tuple[float, float]:
    """Move ``km`` along a great circle in the direction ``bearing`` (a compass word)."""
    theta = math.radians(_BEARINGS[bearing.strip().lower()])
    d = km / EARTH_RADIUS_KM
    phi1, lam1 = math.radians(lat), math.radians(lon)
    phi2 = math.asin(math.sin(phi1) * math.cos(d) + math.cos(phi1) * math.sin(d) * math.cos(theta))
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(d) * math.cos(phi1),
        math.cos(d) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), (math.degrees(lam2) + 540) % 360 - 180


class Gazetteer:
    def __init__(self, directory: str, max_alternates: int = 60) -> None:
        self.directory = directory
        self.places: List[Place] = []
        self._exact: Dict[str, List[int]] = {}  # every name and alternate spelling
        self._primary: Dict[str, List[int]] = {}  # names and ASCII names only
        self._country_names: Dict[str, str] = {}  # folded name -> ISO code
        self._by_country_code: Dict[str, str] = {}  # ISO -> country name
        self._admin1: Dict[str, Tuple[str, str]] = {}  # "CC.ADM1" -> (name, ascii)
        cities = next(
            (
                os.path.join(directory, f)
                for f in ("cities500.txt", "cities5000.txt", "cities15000.txt")
                if os.path.exists(os.path.join(directory, f))
            ),
            None,
        )
        countries = os.path.join(directory, "countryInfo.txt")
        admin1 = os.path.join(directory, "admin1CodesASCII.txt")
        if os.path.exists(countries):
            self._load_countries(countries)
        if os.path.exists(admin1):
            self._load_admin1(admin1)
        if cities is None:
            raise FileNotFoundError(f"no cities*.txt in {directory}")
        self._load_cities(cities, max_alternates)
        self._ascii_names = [_fold(p.name) for p in self.places]

    # ---------------------------------------------------------------- loading
    def _add(self, place: Place, names: Iterable[str], primary: int = 2) -> None:
        """Index ``place`` under ``names``; the first ``primary`` of them are the place's
        own names (the rest alternate spellings, which only an explicit lookup may use)."""
        idx = len(self.places)
        self.places.append(place)
        seen = set()
        for i, raw in enumerate(names):
            n = _fold(raw) if raw else ""
            if not n or n in seen:
                continue
            seen.add(n)
            self._exact.setdefault(n, []).append(idx)
            if i < primary:
                self._primary.setdefault(n, []).append(idx)

    def _load_countries(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.rstrip("\n").split("\t")
                iso, name, capital = cols[0], cols[4], cols[5]
                self._country_names[_fold(name)] = iso
                self._by_country_code[iso] = name
                self._capitals = getattr(self, "_capitals", {})
                self._capitals[iso] = (name, capital, int(cols[7] or 0))

    def _load_admin1(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 3:
                    self._admin1[cols[0]] = (cols[1], cols[2])

    def _load_cities(self, path: str, max_alternates: int) -> None:
        best_in_admin1: Dict[str, Tuple[int, int]] = {}
        capitals: Dict[str, int] = {}
        caps = getattr(self, "_capitals", {})
        with open(path, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 15:
                    continue
                name, ascii_name, alternates = cols[1], cols[2], cols[3]
                lat, lon, cc, adm1 = float(cols[4]), float(cols[5]), cols[8], cols[10]
                population = int(cols[14] or 0)
                alts = [a for a in alternates.split(",") if a and a.isascii()][:max_alternates]
                place = Place(name, lat, lon, self._by_country_code.get(cc, cc), population, "city")
                idx = len(self.places)
                self._add(place, [name, ascii_name, *alts])
                key = f"{cc}.{adm1}"
                if population > best_in_admin1.get(key, (0, -1))[0]:
                    best_in_admin1[key] = (population, idx)
                if cc in caps and _fold(caps[cc][1]) in (_fold(name), _fold(ascii_name)):
                    if (
                        population > self.places[capitals[cc]].population
                        if cc in capitals
                        else True
                    ):
                        capitals[cc] = idx
        for cc, (cname, _capital, cpop) in caps.items():
            if cc in capitals:
                c = self.places[capitals[cc]]
                self._add(
                    Place(
                        f"{c.name}, {cname}", c.lat, c.lon, cname, cpop or c.population, "country"
                    ),
                    [cname],
                )
        for key, (pop, idx) in best_in_admin1.items():
            if key in self._admin1:
                name, ascii_name = self._admin1[key]
                c = self.places[idx]
                self._add(
                    Place(f"{name} ({c.name})", c.lat, c.lon, c.country, pop, "admin1"),
                    [name, ascii_name],
                )

    # ---------------------------------------------------------------- lookup
    def _pick(self, indices: List[int], country: Optional[str], score: float) -> Optional[Place]:
        cands = [self.places[i] for i in indices]
        if country:
            code = self._country_names.get(country)
            cname = self._by_country_code.get(code, "") if code else ""
            narrowed = [p for p in cands if _fold(p.country) in (country, _fold(cname))]
            cands = narrowed or cands
        if not cands:
            return None
        best = max(cands, key=lambda p: (p.kind == "city", p.population))
        return Place(best.name, best.lat, best.lon, best.country, best.population, best.kind, score)

    def lookup(self, query: str, fuzzy_cutoff: int = 86) -> Optional[Place]:
        """The place named by ``query`` (``"Rotterdam"``, ``"port of Rotterdam"``,
        ``"Paris, France"``), or ``None``."""
        q = _squeeze(query)
        country = None
        if "," in q:
            head, tail = q.rsplit(",", 1)
            if _fold(tail) in self._country_names:
                q, country = head, _fold(tail)
        for candidate in (q, _FILLER.sub(" ", q)):
            key = _fold(_squeeze(candidate))
            if key in self._exact:
                return self._pick(self._exact[key], country, 1.0)
        # "Port of Rotterdam, the Netherlands": the longest exact name inside the phrase.
        hit = self._exact_ngrams(_fold(q).split(), country)
        if hit is not None:
            return hit
        cleaned = _fold(_squeeze(_FILLER.sub(" ", q))) or _fold(q)
        return self._fuzzy(cleaned, country, fuzzy_cutoff)

    def _fuzzy(self, cleaned: str, country: Optional[str], cutoff: int) -> Optional[Place]:
        """A spelling-tolerant match on the whole of ``cleaned`` ("Rotterdamm"). The
        character ratio, not a partial-token score: "biggest dutch port" must not become
        some city that shares a few letters."""
        if not cleaned:
            return None
        try:
            from rapidfuzz import fuzz, process
        except ImportError:  # pragma: no cover - rapidfuzz is a declared dependency
            return None
        hit = process.extractOne(cleaned, self._ascii_names, scorer=fuzz.ratio, score_cutoff=cutoff)
        if hit is None:
            return None
        _, score, idx = hit
        # Several places may share the matched name: choose among them by population.
        same = self._exact.get(self._ascii_names[idx], [idx])
        return self._pick(same, country, score / 100.0)

    def _exact_ngrams(self, words: List[str], country: Optional[str] = None) -> Optional[Place]:
        """The longest exact place name among the word n-grams, cities first. Single words
        match a place's own name, or an alternate spelling only when long enough not to be
        an everyday word ("Seville" yes; "No", an alternate name of Ho, Ghana, no)."""
        # A city beats a country or region however long its name ("Port of Rotterdam, the
        # Netherlands" is Rotterdam); among cities the longer name wins ("Santo Domingo de
        # los Colorados" over "Santo Domingo"), then the more populous.
        best: Optional[Tuple[Tuple[bool, int, int], Place]] = None
        for n in range(min(6, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                key = " ".join(words[i : i + n])
                if n == 1 and (len(key) < 3 or _FILLER.fullmatch(key)):
                    continue
                index = self._exact if n > 1 or len(key) >= 5 else self._primary
                hits = index.get(key)
                if hits is None and n == 1:
                    hits = self._primary.get(key)
                if hits:
                    cand = self._pick(hits, country, 1.0)
                    if cand is None:
                        continue
                    rank = (cand.kind == "city", n, cand.population)
                    if best is None or rank > best[0]:
                        best = (rank, cand)
        return best[1] if best else None

    def find_in_text(self, text: str) -> Tuple[Optional[Place], Optional[Tuple[float, str]]]:
        """The place a whole request sentence refers to, and any ``(km, bearing)`` offset.

        Longest exact n-gram first (so "San Jose" beats "Jose", and "Perth, Australia" is
        Perth because a city beats a country). If nothing matches exactly, a spelling-
        tolerant match is tried only when at most three non-filler words remain
        ("Rotterdamm"); a sentence like "the biggest Dutch port" returns nothing, leaving it
        to the language model.
        """
        offset = find_offset(text)
        m = _OFFSET.search(text)
        if m:
            text = text[m.end() :]
        text = re.sub(r"[-+]?\d+(?:\.\d+)?", " ", text)  # numbers are never place names
        hit = self._exact_ngrams(_fold(text).split())
        if hit is not None:
            return hit, offset
        residue = _fold(_squeeze(_FILLER.sub(" ", text)))
        if 0 < len(residue.split()) <= 3:
            return self._fuzzy(residue, None, 86), offset
        return None, offset
