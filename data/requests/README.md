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

## Every target of one seed

`events500_seed1000_targets.jsonl.gz`: one request per target of seed 1000 (20,000 lines,
slot order): the 15,000 background targets as routine requests open all episode and the
5,000 event members with their event's window (DECISIONS.md issue 29). Same fields as
above, plus `meta.slot`; `meta.cluster` is -1 for a background target.

```
python scripts/seed_requests.py --scenario events500 --seed 1000 \
    --output data/requests/events500_seed1000_targets.jsonl.gz --policies greedy_beam
```

writes the file, reads every text back (`sarsat.tasking.read.read_request`, which sees
only the text and `issued_utc`), rebuilds the episode from the readings
(`rebuild_state`), and reports the errors and each policy's return on the original and
the rebuilt episode (seed 1000: `greedy_beam` 0.3652 / 0.3650, `coop_plan` 0.8747 /
0.8748). To rebuild an episode from the file yourself:

```python
readings = [read_request(r["text"], parse_iso(r["issued_utc"]), places) for r in records]
state = rebuild_state(env, jax.random.PRNGKey(seed), readings, parse_iso(records[0]["issued_utc"]))
```
