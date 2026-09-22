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
