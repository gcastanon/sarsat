"""Spherical-Earth geometry and a table of named places for wording locations.

The Earth is the same sphere the environment uses (``EARTH_RADIUS_KM``), so a distance
computed here is the one the environment would compute.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from sarsat.orbits import EARTH_RADIUS_KM

KM_PER_DEG = np.pi * EARTH_RADIUS_KM / 180.0
KM_PER_UNIT = {"km": 1.0, "mi": 1.609344, "nmi": 1.852}
COMPASS16 = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)  # fmt: skip


def distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance [km]; broadcasts over arrays."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing from point 1 to point 2 [deg clockwise from north, in [0, 360)]."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.rad2deg(np.arctan2(y, x)) % 360.0


def destination(lat: float, lon: float, bearing: float, dist_km: float) -> Tuple[float, float]:
    """The point ``dist_km`` from ``(lat, lon)`` along initial ``bearing`` [deg]."""
    p1, l1 = np.deg2rad(lat), np.deg2rad(lon)
    th, d = np.deg2rad(bearing), dist_km / EARTH_RADIUS_KM
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(th))
    l2 = l1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return float(np.rad2deg(p2)), float((np.rad2deg(l2) + 180.0) % 360.0 - 180.0)


class Places:
    """Named places as parallel arrays, sorted by latitude; ``region`` is the US state or the
    country name. A label (``Kearney, Nebraska``, or ``Kearney, NE`` for US places) is used
    in a request only if it names exactly one place, so reading it back is unambiguous."""

    def __init__(
        self,
        name: np.ndarray,
        region: np.ndarray,
        region_code: np.ndarray,  # US state code (e.g. ``NE``) or ISO country code
        country_code: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        population: np.ndarray,
        timezone: np.ndarray,  # IANA zone name
    ) -> None:
        order = np.argsort(np.asarray(lat, np.float64), kind="stable")
        self.name = np.asarray(name, dtype=object)[order]
        self.region = np.array([r.strip() for r in np.asarray(region)[order]], dtype=object)
        self.region_code = np.asarray(region_code, dtype=object)[order]
        self.country_code = np.asarray(country_code, dtype=object)[order]
        self.lat = np.asarray(lat, np.float64)[order]
        self.lon = np.asarray(lon, np.float64)[order]
        self.population = np.asarray(population, np.float64)[order]
        self.timezone = np.asarray(timezone, dtype=object)[order]
        self._index: Dict[Tuple[str, str], List[int]] = {}
        for i in range(len(self.name)):
            for region in {self._region(i, False), self._region(i, True)}:
                self._index.setdefault((self.name[i], region), []).append(i)
        # Regions a reader looks for after ", ", longest first ("Korea, North" before "Korea").
        self.regions = sorted({k[1] for k in self._index}, key=len, reverse=True)

    def __len__(self) -> int:
        return len(self.name)

    def _region(self, i: int, abbreviate: bool) -> str:
        us = self.country_code[i] == "US"
        return self.region_code[i] if abbreviate and us else self.region[i]

    def nearby(self, lat: float, lon: float, radius_km: float) -> Tuple[np.ndarray, np.ndarray]:
        """Indices of the places within ``radius_km``, nearest first, and their distances."""
        band = radius_km / KM_PER_DEG
        lo, hi = np.searchsorted(self.lat, [lat - band, lat + band])
        d = distance_km(lat, lon, self.lat[lo:hi], self.lon[lo:hi])
        idx = np.flatnonzero(d <= radius_km)
        order = np.argsort(d[idx], kind="stable")
        return idx[order] + lo, d[idx][order]

    def label(self, i: int, abbreviate: bool = False) -> str:
        """``Kearney, Nebraska`` (``Kearney, NE`` abbreviated, US only) or ``Mombasa, Kenya``."""
        return f"{self.name[i]}, {self._region(i, abbreviate)}"

    def unique(self, i: int, abbreviate: bool = False) -> bool:
        """Whether place ``i``'s label names it alone (and can be read back)."""
        return (
            "," not in self.name[i]
            and len(self._index[(self.name[i], self._region(i, abbreviate))]) == 1
        )

    def lookup(self, name: str, region: str) -> Optional[int]:
        """The one place labelled ``name, region``, or None."""
        hits = self._index.get((name, region), [])
        return hits[0] if len(hits) == 1 else None


def load_places(min_population: int = 5000) -> Places:
    """GeoNames places of at least ``min_population`` (1000, 5000 or 15000) via the
    ``geonamescache`` package (``pip install -e ".[tasking]"``). GeoNames data is CC BY 4.0."""
    import geonamescache

    gc = geonamescache.GeonamesCache(min_city_population=min_population)
    countries = {code: c["name"] for code, c in gc.get_countries().items()}
    states = {code: s["name"] for code, s in gc.get_us_states().items()}
    rows = []
    for c in gc.get_cities().values():
        if c["countrycode"] == "US" and c["admin1code"] in states:
            region, code = states[c["admin1code"]], c["admin1code"]
        else:
            region, code = countries.get(c["countrycode"], c["countrycode"]), c["countrycode"]
        rows.append(
            (c["name"], region, code, c["countrycode"], c["latitude"], c["longitude"],
             c["population"], c["timezone"])
        )  # fmt: skip
    rows.sort(key=lambda r: (r[4], r[5], r[0]))  # deterministic order whatever the cache's
    cols = list(zip(*rows, strict=True))
    return Places(
        name=np.array(cols[0], dtype=object),
        region=np.array(cols[1], dtype=object),
        region_code=np.array(cols[2], dtype=object),
        country_code=np.array(cols[3], dtype=object),
        lat=np.array(cols[4], dtype=np.float64),
        lon=np.array(cols[5], dtype=np.float64),
        population=np.array(cols[6], dtype=np.float64),
        timezone=np.array(cols[7], dtype=object),
    )
