"""Natural-language tasking requests written from a windowed scenario's raw numbers.

``extract_events`` reads the events (centre, window, priority) of an episode and
``extract_targets`` every single target; ``render_request`` writes one request for either,
with its raw label, tolerance and exact reading; ``sarsat.tasking.read`` reads requests
back and rebuilds an episode from them. ``scripts/make_requests.py`` and
``scripts/seed_requests.py`` write the data sets; DECISIONS.md issues 28 and 29.
"""

from sarsat.tasking.events import Event, Target, cluster_centres, extract_events, extract_targets
from sarsat.tasking.geo import Places, load_places
from sarsat.tasking.render import render_request, score, steps_from_times

__all__ = [
    "Event",
    "Places",
    "Target",
    "cluster_centres",
    "extract_events",
    "extract_targets",
    "load_places",
    "render_request",
    "score",
    "steps_from_times",
]
