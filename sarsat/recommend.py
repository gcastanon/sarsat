"""How to change a request so the policy collects it without dropping anything else.

Given a provisional request (:class:`sarsat.live.Preview`) and the forecast of the world
without it (the *base*), find variants of the request -- a later deadline, another priority,
a looser image quality, a nearby location -- that the policy would collect while every
target the base collects is still collected (the same set; a shift in step or satellite is
reported, not held against the variant).

Two stages, because the policy is a learned function and the only way to know what it does
is to run it:

1. *Generate* (cheap, geometry only). A variant is worth a forecast if some satellite can
   reach its point at a step inside its window, at which the base forecast has that
   satellite idle (not sensing) and charged. Variants are ordered by how many parameters
   they change, then by their earliest such idle pass. Changing the priority alone changes no
   geometry; it is kept because it changes what the policy wants.
2. *Confirm* (one forecast each, ``max_forecasts`` at most). Each variant in order is
   previewed and forecast with ``LiveWorld.whatif``; it is an alternative if the request is
   collected and the set of other collected targets is unchanged.

The candidate test uses the same reach as ``LiveWorld.feasibility`` (the cheapest mode's
geometry and a battery forecast), so for a variant needing a demanding mode it can propose
passes the forecast then refuses; the forecast has the last word. Splitting a request in two
is not generated. DECISIONS.md issue 40.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from sarsat.forecast import Forecast
from sarsat.live import MAX_DEADLINE_STEPS, PRIORITY_WEIGHT, LiveWorld, Preview
from sarsat.orbits import EARTH_RADIUS_KM
from sarsat.targets import latlon_to_unit_vector

PRIORITIES = ("low", "normal", "high")
DEGREES_OF_FREEDOM = ("deadline", "priority", "quality", "location")


@dataclass
class Variant:
    """One way of changing the request, and what came of it."""

    lat: float
    lon: float
    priority: str
    deadline_steps: int
    quality: Optional[float]
    changes: List[str] = field(default_factory=list)  # "deadline 10 -> 20", ...
    passes: int = 0  # idle, reachable (satellite, step) pairs in the base forecast
    first_pass: Optional[int] = None  # earliest such global step
    confirmed: Optional[bool] = None  # None: not forecast (out of budget)
    reason: str = ""  # why it was rejected or skipped
    whatif: Optional[Dict[str, Any]] = None  # summary of the confirming what-if

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "priority": self.priority,
            "deadline_steps": self.deadline_steps,
            "quality": self.quality,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.params,
            "changes": list(self.changes),
            "passes": self.passes,
            "first_pass": self.first_pass,
            "confirmed": self.confirmed,
            "reason": self.reason,
            "whatif": self.whatif,
        }


def _offset(lat: float, lon: float, km: float, bearing_deg: float) -> Tuple[float, float]:
    """The point ``km`` away along ``bearing_deg`` (great circle)."""
    d = km / EARTH_RADIUS_KM
    b = math.radians(bearing_deg)
    la, lo = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(b))
    lo2 = lo + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(la), math.cos(d) - math.sin(la) * math.sin(la2)
    )
    lat2 = max(-89.0, min(89.0, math.degrees(la2)))
    lon2 = (math.degrees(lo2) + 180.0) % 360.0 - 180.0
    return round(lat2, 4), round(lon2, 4)


def summary(whatif: Dict[str, Any]) -> Dict[str, Any]:
    """The what-if without its two full schedules: the request's fate and the impact."""
    return {
        "request_fate": whatif["request_fate"],
        "same_set": whatif["same_set"],
        "lost": whatif["lost"],
        "gained": whatif["gained"],
        "shifted": whatif["shifted"],
        "reward_delta": whatif["reward_delta"],
        "items": whatif["items"],
    }


def acceptable(whatif: Dict[str, Any], max_lost: int = 0) -> bool:
    """Collected, and no other collected target lost (``max_lost`` relaxes that: at most so
    many lost, for a user who will trade; the default is the strict same-set test)."""
    if not whatif["request_fate"]["collected"]:
        return False
    return bool(whatif["same_set"]) if max_lost <= 0 else int(whatif["lost"]) <= int(max_lost)


class Recommender:
    """Variants of a request that the policy collects at no cost to the rest of the plan."""

    def __init__(
        self,
        world: LiveWorld,
        policy: Any,
        allow: Sequence[str] = DEGREES_OF_FREEDOM,
        deadline_step: int = 5,
        radii_km: Sequence[float] = (25.0, 50.0),
        bearings: int = 8,
        max_forecasts: int = 6,
        max_results: int = 3,
        max_changes: int = 2,
        max_lost: int = 0,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        unknown = set(allow) - set(DEGREES_OF_FREEDOM)
        if unknown:
            raise ValueError(f"unknown degrees of freedom {sorted(unknown)}")
        self.world, self.policy = world, policy
        self.allow = tuple(allow)
        self.deadline_step = int(deadline_step)
        self.radii_km = tuple(float(r) for r in radii_km)
        self.bearings = int(bearings)
        self.max_forecasts = int(max_forecasts)
        self.max_results = int(max_results)
        self.max_changes = int(max_changes)
        self.max_lost = int(max_lost)
        # Called as the search goes: {"stage": "variant", "done", "budget", "candidates",
        # "changes", "found"} before each confirming forecast, and {"stage": "step", "done",
        # "total"} every few steps inside one (through ``LiveWorld.whatif``).
        self.progress = progress

    # ------------------------------------------------------------- generate

    def _idle(self, base: Forecast) -> np.ndarray:
        """``(H, N)``: at each forecast step, satellites not sensing and able to pay for a
        look (charge before the step, as ``LiveWorld.feasibility`` forecasts it)."""
        env = self.world.env
        before = np.concatenate(
            [np.asarray(self.world.state.battery)[None], base.battery[:-1]], axis=0
        )
        charged = before >= env.sense_cost - 1e-6
        return ~base.sensing & charged

    def _reach(self, lat: float, lon: float, horizon: int) -> np.ndarray:
        """``(horizon, N)`` geometric reach of the point over the next ``horizon`` steps."""
        w = self.world
        step = int(w.state.step)
        steps = step + 1 + jnp.arange(horizon)
        point = EARTH_RADIUS_KM * latlon_to_unit_vector(jnp.asarray([lat, lon], jnp.float32))
        return np.asarray(w._jit_access(w.state.orbit, point, steps))

    def _passes(self, v: Variant, idle: np.ndarray, reach_cache: Dict) -> None:
        """Fill ``v.passes`` / ``v.first_pass`` from the base forecast's idle satellites."""
        w = self.world
        step = int(w.state.step)
        window = min(v.deadline_steps, w.T - step, len(idle))
        if window <= 0:
            v.passes, v.first_pass = 0, None
            return
        key = (v.lat, v.lon)
        if key not in reach_cache:
            reach_cache[key] = self._reach(v.lat, v.lon, min(MAX_DEADLINE_STEPS, w.T - step))
        ok = reach_cache[key][:window] & idle[:window]
        hits = np.flatnonzero(ok.any(axis=1))
        v.passes = int(ok.sum())
        v.first_pass = int(w.global_step + 1 + hits[0]) if hits.size else None

    def _axis_values(self, r) -> Dict[str, List[Tuple[Any, str]]]:
        """Per degree of freedom, the alternative values and how to describe the change."""
        w = self.world
        step = int(w.state.step)
        axes: Dict[str, List[Tuple[Any, str]]] = {}
        if "deadline" in self.allow:
            vals = []
            cap = min(MAX_DEADLINE_STEPS, w.T - step)
            d = r.deadline_steps + self.deadline_step
            while d <= cap:
                vals.append((d, f"deadline {r.deadline_steps} -> {d} steps"))
                d += self.deadline_step
            if cap > r.deadline_steps and (not vals or vals[-1][0] != cap):
                vals.append((cap, f"deadline {r.deadline_steps} -> {cap} steps"))
            axes["deadline"] = vals
        if "priority" in self.allow:
            axes["priority"] = [
                (p, f"priority {r.priority} -> {p}") for p in PRIORITIES if p != r.priority
            ]
        if "quality" in self.allow and w._has_quality and r.quality is not None:
            lower = sorted({m.quality for m in w.env.modes if m.quality < r.quality}, reverse=True)
            vals = [(q, f"quality {r.quality:g} -> {q:g}") for q in lower]
            vals.append((None, f"quality {r.quality:g} -> any"))
            axes["quality"] = vals
        if "location" in self.allow:
            vals = []
            for km in self.radii_km:
                for k in range(self.bearings):
                    bearing = 360.0 * k / self.bearings
                    lat, lon = _offset(r.lat, r.lon, km, bearing)
                    vals.append(((lat, lon), f"location moved {km:g} km at {bearing:g} deg"))
            axes["location"] = vals
        return {k: v for k, v in axes.items() if v}

    def variants(self, preview: Preview) -> List[Variant]:
        """Every variant changing up to ``max_changes`` degrees of freedom, one value each,
        in change-count order (unranked: see :meth:`rank`)."""
        r = preview.request
        axes = self._axis_values(r)
        names = list(axes)
        out: List[Variant] = []
        for k in range(1, min(self.max_changes, len(names)) + 1):
            for combo in itertools.combinations(names, k):
                for values in itertools.product(*[axes[n] for n in combo]):
                    v = Variant(r.lat, r.lon, r.priority, r.deadline_steps, r.quality)
                    for name, (value, text) in zip(combo, values, strict=True):
                        if name == "deadline":
                            v.deadline_steps = value
                        elif name == "priority":
                            v.priority = value
                        elif name == "quality":
                            v.quality = value
                        else:
                            v.lat, v.lon = value
                        v.changes.append(text)
                    out.append(v)
        return out

    def rank(self, variants: Iterable[Variant], base: Forecast) -> List[Variant]:
        """Score every variant's idle passes against ``base`` and order them: fewer changes
        first, then the earliest idle pass, then more passes. Variants without a pass are
        dropped unless they only change the priority (no geometry changed)."""
        idle = self._idle(base)
        cache: Dict = {}
        kept = []
        for v in variants:
            self._passes(v, idle, cache)
            if v.passes == 0 and any(not c.startswith("priority") for c in v.changes):
                v.reason = "no satellite is both able to reach it and idle inside its window"
                continue
            kept.append(v)
        kept.sort(
            key=lambda v: (
                len(v.changes),
                v.first_pass if v.first_pass is not None else 10**9,
                -v.passes,
            )
        )
        return kept

    # ------------------------------------------------------------- confirm

    def recommend(
        self,
        preview: Preview,
        base: Optional[Forecast] = None,
        original: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Alternatives to ``preview``'s request. ``base`` is the forecast without the request
        (made here if not given) and ``original`` the request's own what-if (likewise)."""
        t0 = time.perf_counter()
        w = self.world
        base = w.forecast(self.policy) if base is None else base
        original = w.whatif(self.policy, preview, base) if original is None else original
        forecasts = 0
        result: Dict[str, Any] = {
            "request": preview.request.to_dict(),
            "original": summary(original),
            "alternatives": [],
            "rejected": [],
            "untried": 0,
            "candidates": 0,
            "forecasts": 0,
            "allow": list(self.allow),
            "max_lost": self.max_lost,
        }
        if acceptable(original, self.max_lost):
            result["verdict"] = "the request is collected as it is, at no cost to the rest"
            result["seconds"] = round(time.perf_counter() - t0, 2)
            return result

        ranked = self.rank(self.variants(preview), base)
        steps = None
        if self.progress is not None:
            steps = lambda done, total: self.progress(  # noqa: E731
                {"stage": "step", "done": done, "total": total}
            )
        result["candidates"] = len(ranked)
        r = preview.request
        for v in ranked:
            if len(result["alternatives"]) >= self.max_results or forecasts >= self.max_forecasts:
                result["untried"] += 1
                continue
            alt = w.preview_request(
                v.lat,
                v.lon,
                v.priority,
                v.deadline_steps,
                text=r.text,
                place=r.place,
                quality=v.quality,
            )
            if self.progress is not None:
                self.progress(
                    {
                        "stage": "variant",
                        "done": forecasts,
                        "budget": self.max_forecasts,
                        "candidates": len(ranked),
                        "changes": list(v.changes),
                        "found": len(result["alternatives"]),
                    }
                )
            out = w.whatif(self.policy, alt, base, progress=steps)
            forecasts += 1
            v.whatif = summary(out)
            v.confirmed = acceptable(out, self.max_lost)
            if v.confirmed:
                result["alternatives"].append(v.to_dict())
            else:
                fate = out["request_fate"]
                if not fate["collected"]:
                    v.reason = "the policy still does not collect it"
                else:
                    v.reason = f"collected, but {out['lost']} other target(s) would be lost"
                result["rejected"].append(v.to_dict())
        result["forecasts"] = forecasts
        # The rejected variants, least damaging first, so a caller can see the near misses.
        result["rejected"].sort(
            key=lambda d: (
                not d["whatif"]["request_fate"]["collected"],
                d["whatif"]["lost"],
                -d["whatif"]["gained"],
            )
        )
        if result["alternatives"]:
            result["verdict"] = f"{len(result['alternatives'])} way(s) to fit it in"
        elif result["untried"]:
            result["verdict"] = (
                f"none of the {forecasts} variants forecast fits; {result['untried']} not tried "
                f"(budget {self.max_forecasts} forecasts)"
            )
        else:
            result["verdict"] = "no variant within the allowed changes fits without a loss"
        result["seconds"] = round(time.perf_counter() - t0, 2)
        return result


__all__ = [
    "DEGREES_OF_FREEDOM",
    "PRIORITIES",
    "PRIORITY_WEIGHT",
    "Recommender",
    "Variant",
    "acceptable",
    "summary",
]
