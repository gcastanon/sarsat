# Tasking requests

`events500_seeds1000-1015.jsonl`: one natural-language request for each of the 100 events
of `sarsat-500sat-events` on each paired test seed 1000-1015 (1,600 lines), written from
the raw event numbers by `sarsat.tasking` (DECISIONS.md issue 28). Regenerate with

```
pip install -e ".[tasking]"
python scripts/make_requests.py --scenario events500 --seed-start 1000 --seed-end 1015 \
    --output data/requests/events500_seeds1000-1015.jsonl
```

Every choice is seeded, so the rerun is byte-identical. Other seeds or scenarios
(`events200`) take the same command.

| Field | Meaning |
|---|---|
| `id`, `scenario`, `seed` | `events500-1003-017` is cluster 17 of `jax.random.PRNGKey(1003)` |
| `text` | the request |
| `issued_utc` | when it was sent: step 0 of the episode, a whole minute of 2026 drawn per seed |
| `label` | the raw event: centre `lat` / `lon` [deg], `start_utc` / `stop_utc` (first and last step, inclusive), `priority` (weight relative to a background target) |
| `steps` | the window as environment steps `[first, last]`; step `s` is `issued_utc + s` minutes |
| `tol` | the most an exact reading of the text can differ from `label`: `km`, `min` |
| `stated` | that exact reading (the rounded coordinates, the point "40 km NW of" a town) |
| `meta` | `cluster`, `loc_form`, `time_form`, `tier`, `template`, for breaking errors down |

A parser's output `{lat, lon, start_utc, stop_utc, priority}` is scored with
`sarsat.tasking.score(pred, record)`: great-circle km and minutes against `label`, `ok`
when both are within `tol` and the priority tier matches. The reading conventions a
parser must follow (dateless clock times are the next occurrence, local times are the
named place's zone, bearings are from the place) are in `sarsat/tasking/render.py`.

Place names: GeoNames (CC BY 4.0, <https://www.geonames.org>) via `geonamescache`.
