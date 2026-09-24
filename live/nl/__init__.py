"""Natural-language tasking requests -> (location, priority, deadline), fully offline.

Stages, each optional after the first:

1. :mod:`live.nl.rules` -- regular expressions settle explicit coordinates, priority words
   and deadline phrases. Priority and deadline come from here alone: the rules score 100%
   on the evaluation set, whereas a small model guesses when nothing is stated.
2. :mod:`live.nl.geocode` -- an offline GeoNames gazetteer picks the place out of the
   sentence (longest exact name first) and reads "N km <bearing> of" offsets.
3. :mod:`live.nl.llm` -- a small local instruct model (llama.cpp, JSON-schema grammar) is
   the fallback when the gazetteer finds no name in the sentence: it names the place, the
   gazetteer geocodes it, and only if that fails too are the model's own coordinates used,
   flagged as approximate.

:func:`parse` runs whatever is installed: without a gazetteer only coordinate requests
resolve, and the result says so. ``prefer_model=True`` asks the model before the sentence
search (for measuring the fallback; ``live/nl/evaluate.py --model-first``).
"""

from __future__ import annotations

from typing import Optional

from live.nl.rules import parse_rules
from live.nl.schema import Parsed

_geocoder = None
_model = None


def configure(geocoder=None, model=None) -> None:
    """Install the (optional) gazetteer and language model used by :func:`parse`."""
    global _geocoder, _model
    _geocoder, _model = geocoder, model


def parse(text: str, geocoder=None, model=None, prefer_model: bool = False) -> Parsed:
    geocoder = _geocoder if geocoder is None else geocoder
    model = _model if model is None else model
    p = parse_rules(text)
    if p.located:
        return p
    if geocoder is None and model is None:
        p.notes.append("no location found (no gazetteer or model installed)")
        return p

    if geocoder is not None and not prefer_model and _from_sentence(p, text, geocoder):
        return p

    extraction = None
    if model is not None:
        try:
            extraction = model.extract(text)
        except Exception as err:  # the model is optional; say what happened
            p.notes.append(f"model failed: {err}")
    if extraction:
        p.extraction.update({f"model_{k}": v for k, v in extraction.items()})
        p.place = (extraction.get("place") or "").strip()
        if p.place and geocoder is not None:
            hit = geocoder.lookup(p.place)
            if hit is not None:
                p.lat, p.lon, p.place = hit.lat, hit.lon, hit.name
                p.source, p.confidence = "gazetteer", hit.score
                _apply_offset(p, text, extraction)
                return p
            p.notes.append(f"'{p.place}' not found in the gazetteer")

    if geocoder is not None and prefer_model and _from_sentence(p, text, geocoder):
        return p

    if extraction and extraction.get("lat") is not None and extraction.get("lon") is not None:
        p.lat, p.lon = float(extraction["lat"]), float(extraction["lon"])
        p.source, p.confidence = "model", 0.3
        p.notes.append("coordinates guessed by the model (approximate)")
        _apply_offset(p, text, extraction)
        return p
    p.notes.append("no location found")
    return p


def _from_sentence(p: Parsed, text: str, geocoder) -> bool:
    hit, offset = geocoder.find_in_text(text)
    if hit is None:
        return False
    p.lat, p.lon, p.place = hit.lat, hit.lon, hit.name
    p.source, p.confidence = "gazetteer", hit.score
    if offset is not None:
        _shift(p, offset[0], offset[1])
    return True


def _apply_offset(p: Parsed, text: str, extraction: Optional[dict]) -> None:
    """Shift by "N km <bearing> of <place>": from the text's own words when present, else
    from what the model extracted."""
    from live.nl.geocode import find_offset

    offset = find_offset(text)
    if offset is None and extraction and extraction.get("offset_km") and extraction.get("bearing"):
        offset = (float(extraction["offset_km"]), str(extraction["bearing"]))
    if offset is not None:
        _shift(p, offset[0], offset[1])


def _shift(p: Parsed, km: float, bearing: str) -> None:
    from live.nl.geocode import offset_point

    p.lat, p.lon = offset_point(p.lat, p.lon, km, bearing)
    p.notes.append(f"{km:g} km {bearing} of {p.place}")


__all__ = ["Parsed", "configure", "parse"]
