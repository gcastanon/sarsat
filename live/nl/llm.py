"""A small local instruct model, run in-process by llama.cpp, that turns a request into
structured fields under a JSON-schema grammar. It never geocodes: the place *name* goes to
the gazetteer; its own coordinates are only kept as an "approximate" fallback.

Any GGUF chat model works; Qwen2.5-1.5B-Instruct (Q4_K_M, ~1 GB) answers in a few seconds
on CPU, Qwen2.5-3B-Instruct (~2 GB) is more robust to unusual phrasing. Install with
``pip install llama-cpp-python`` (CPU wheels: ``--extra-index-url
https://abetlen.github.io/llama-cpp-python/whl/cpu``).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional

_BEARINGS = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
_FIELDS = ["place", "country", "lat", "lon", "offset_km", "bearing", "priority", "deadline_minutes"]
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "place": {"type": ["string", "null"]},
        "country": {"type": ["string", "null"]},
        "lat": {"type": ["number", "null"]},
        "lon": {"type": ["number", "null"]},
        "offset_km": {"type": ["number", "null"]},
        "bearing": {"type": ["string", "null"], "enum": [None, *_BEARINGS]},
        "priority": {"type": ["string", "null"], "enum": [None, "low", "normal", "high"]},
        "deadline_minutes": {"type": ["integer", "null"]},
    },
    "required": _FIELDS,
}

SYSTEM = (
    "You extract satellite imaging tasking requests into JSON. Fields: place (the named "
    "location, e.g. 'Rotterdam' or 'Port of Rotterdam'; null if none), country (if stated or "
    "obvious, else null), lat/lon (your best estimate of the place's coordinates in decimal "
    "degrees, or null), offset_km and bearing (only for phrases like '50 km east of X'), "
    "priority (low/normal/high; 'urgent' or 'critical' mean high, 'routine' or 'when "
    "convenient' mean low; null if unstated), deadline_minutes (the requested time limit in "
    "minutes, null if unstated). Answer with the JSON object only."
)


class LocalModel:
    def __init__(self, path: str, n_threads: Optional[int] = None, n_ctx: int = 2048) -> None:
        from llama_cpp import Llama

        self.path = path
        self._llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
        self._lock = threading.Lock()  # one completion at a time

    def extract(self, text: str, max_tokens: int = 160) -> Dict[str, Any]:
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                response_format={"type": "json_object", "schema": SCHEMA},
                temperature=0.0,
                max_tokens=max_tokens,
            )
        raw = out["choices"][0]["message"]["content"]
        data = json.loads(raw)
        return _normalise(data)


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    place = data.get("place")
    if place and data.get("country") and "," not in str(place):
        place = f"{place}, {data['country']}"
    out["place"] = place or None
    for key in ("lat", "lon", "offset_km"):
        v = data.get(key)
        out[key] = float(v) if isinstance(v, (int, float)) else None
    if out["lat"] is not None and not -90 <= out["lat"] <= 90:
        out["lat"] = out["lon"] = None
    if out["lon"] is not None and not -180 <= out["lon"] <= 180:
        out["lat"] = out["lon"] = None
    bearing = data.get("bearing")
    out["bearing"] = str(bearing).lower() if bearing else None
    priority = data.get("priority")
    out["priority"] = str(priority).lower() if priority else None
    d = data.get("deadline_minutes")
    out["deadline_minutes"] = int(d) if isinstance(d, (int, float)) and d > 0 else None
    return out
