# Example evaluation output

`8sat-100tg-seed0/` holds the four CSV logs of one reproducible episode: 8 satellites,
100 targets, 180 steps, the greedy baseline policy, seed 0. It images 52 of the 100
targets (return 0.520) in 52 looks.

Use it to try the viewer without running anything: open `sarsat/viewer.html`, click
**Load CSVs…** and select all four files (or drag them onto the page).

Regenerate with:

```bash
python -m sarsat.evaluate --satellites 8 --targets 100 --seed 0 --policy greedy \
    --out examples/8sat-100tg-seed0
```

That also writes an `episode.html`, which is not kept here because the viewer template
plus these CSVs already cover it. `tests/test_evaluate.py` checks that a fresh run still
reproduces these files, so a failure there means the environment changed and the
examples need regenerating.

## Best trained policies vs the references

Six more episodes, all seed 1000, written by `scripts/export_viewer_runs.py` (which also
writes an `episode.html` per run into `runs/viewer/`; those are not kept here). Within a
scenario the orbits and targets are identical, so the three runs can be compared step by
step. Load any folder's four CSVs into `sarsat/viewer.html` as above.

| Folder | Scenario | Policy | Return |
|---|---|---|---|
| `hotspots100-greedy_beam-seed1000/` | `sarsat-100sat-hotspots` | `greedy_beam` | 0.799 |
| `hotspots100-coop_plan-seed1000/` | `sarsat-100sat-hotspots` | `coop_plan` | 0.927 |
| `hotspots100-mappo_v5-seed1000/` | `sarsat-100sat-hotspots` | best MAPPO (`mappo_v5`) | 0.918 |
| `events200-greedy_beam-seed1000/` | `sarsat-200sat-events` | `greedy_beam` | 0.405 |
| `events200-coop_plan-seed1000/` | `sarsat-200sat-events` | `coop_plan` | 0.675 |
| `events200-ev_mappo_e-seed1000/` | `sarsat-200sat-events` | best MAPPO (`ev_mappo_e`) | 0.658 |

The `events200` runs' `targets.csv` carries `window_open` / `window_close`, so the viewer
shows each event cluster only while its window is open (DECISIONS.md issue 24). The
trained-policy runs need the checkpoints in `~/marl/runs/` to regenerate; the reference
runs regenerate from the environment alone.
