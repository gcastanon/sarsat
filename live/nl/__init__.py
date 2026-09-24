"""Natural-language tasking requests -> (location, priority, deadline), fully offline.

Three stages, each optional after the first:

1. :mod:`live.nl.rules` -- regular expressions for explicit coordinates, priority words and
   deadline phrases. If coordinates are present the location is settled here.
2. :mod:`live.nl.llm` -- a small local instruct model (llama.cpp) under a JSON-schema grammar
   extracts the place name (and offsets such as "50 km east of ...") when the rules found
   no coordinates.
3. :mod:`live.nl.geocode` -- an offline GeoNames gazetteer turns the place name into
   coordinates; the model's own coordinate guess is only a flagged fallback.

:func:`parse` runs the pipeline with whatever is installed: without a model or gazetteer,
only coordinate requests resolve, and the result says so.
"""

from __future__ import annotations

from typing import Optional

from live.nl.rules import parse_rules
from live.nl.schema import Parsed, clamp_deadline, normalise_priority

_geocoder = None
_model = None


def configure(geocoder=None, model=None) -> None:
    """Install the (optional) gazetteer and language model used by :func:`parse`."""
    global _geocoder, _model
    _geocoder, _model = geocoder, model


def parse(text: str, geocoder=None, model=None) -> Parsed:
    geocoder = _geocoder if geocoder is None else geocoder
    model = _model if model is None else model
    p = parse_rules(text)
    if p.located:
        return p
    extraction = None
    if model is not None:
        try:
            extraction = model.extract(text)
        except Exception as err:  # the model is optional; say what happened
            p.notes.append(f"model failed: {err}")
    if extraction:
        # The rules win where they found something; the model fills the rest.
        rules_priority = p.extraction.get("priority")
        rules_deadline = p.extraction.get("deadline_minutes")
        p.extraction.update(extraction)
        p.place = (extraction.get("place") or "").strip()
        if rules_priority is None and extraction.get("priority"):
            p.priority = normalise_priority(extraction["priority"])
        if rules_deadline is None and extraction.get("deadline_minutes"):
            p.deadline_minutes = clamp_deadline(extraction["deadline_minutes"])
    if not p.place and geocoder is not None:
        # No model (or it found no place): pick the place out of the sentence itself.
        hit, offset = geocoder.find_in_text(text)
        if hit is not None:
            p.lat, p.lon, p.place = hit.lat, hit.lon, hit.name
            p.source, p.confidence = "gazetteer", hit.score
            if offset is not None:
                _apply_offset(p, {"offset_km": offset[0], "bearing": offset[1]})
            return p
    if p.place and geocoder is not None:
        hit = geocoder.lookup(p.place)
        if hit is not None:
            p.lat, p.lon, p.place = hit.lat, hit.lon, hit.name
            p.source, p.confidence = "gazetteer", hit.score
            _apply_offset(p, extraction)
            return p
        p.notes.append(f"'{p.place}' not found in the gazetteer")
    if extraction and extraction.get("lat") is not None and extraction.get("lon") is not None:
        p.lat, p.lon = float(extraction["lat"]), float(extraction["lon"])
        p.source, p.confidence = "model", 0.3
        p.notes.append("coordinates guessed by the model (approximate)")
        _apply_offset(p, extraction)
        return p
    p.notes.append("no location found")
    return p


def _apply_offset(p: Parsed, extraction: Optional[dict]) -> None:
    """Shift by "N km <bearing> of <place>" when the model extracted one."""
    if not extraction or not extraction.get("offset_km") or not extraction.get("bearing"):
        return
    from live.nl.geocode import offset_point

    p.lat, p.lon = offset_point(p.lat, p.lon, float(extraction["offset_km"]), extraction["bearing"])
    p.notes.append(f"{extraction['offset_km']:g} km {extraction['bearing']} of {p.place}")


__all__ = ["Parsed", "configure", "parse"]
