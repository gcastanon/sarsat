"""Natural-language tasking requests written from a windowed scenario's raw event numbers.

``extract_events`` reads the events (centre, window, priority) of an episode;
``render_request`` writes one request per event with its raw label, tolerances and exact
readings. ``scripts/make_requests.py`` writes a JSONL data set; DECISIONS.md issue 28.
"""

from sarsat.tasking.events import Event, cluster_centres, extract_events
from sarsat.tasking.geo import Places, load_places
from sarsat.tasking.render import render_request, score, steps_from_times

__all__ = [
    "Event",
    "Places",
    "cluster_centres",
    "extract_events",
    "load_places",
    "render_request",
    "score",
    "steps_from_times",
]
