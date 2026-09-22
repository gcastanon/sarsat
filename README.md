# SarSat

A small, fast multi-agent reinforcement-learning environment in JAX. Agents are
satellites in low Earth orbit that cooperate to image ground targets. It follows the
[Jumanji](https://github.com/instadeepai/jumanji) API and plugs into MAPX through a thin
wrapper (`mapx_integration/`).

* Design rationale, issue by issue: [DECISIONS.md](DECISIONS.md)
* What was deliberately left out: [IMPROVEMENTS.md](IMPROVEMENTS.md)

## The task

| | |
|---|---|
| Agents | 1-1000 satellites on random circular orbits; they never manoeuvre |
| Step | 60 s of simulated time |
| Action | `[incidence, squint, side, sense]` per satellite; two continuous slots in `[-1, 1]`, two binary switches |
| Beam | rectangular, +-4 x +-4 degrees by default |
| Access | side-looking both sides: incidence 20-45 degrees, squint +-25 degrees, nadir hole |
| Battery | sensing costs 10% per step, 2% recharges every step -> 20% sustainable duty cycle |
| Targets | fixed lat/lon points, 80% on land / 20% on water, priority `1 / num_targets` |
| Reward | shared: the priority of every target newly caught in a sensing beam; each pays once |
| Observation | egocentric: cross-track/squint tracks of accessible targets over the next N=2 steps, az/el/distance of the K=5 nearest satellites, own battery |
| Episode | ends when every target is imaged (termination) or after `time_limit` steps (truncation) |

The episode return is the fraction of targets imaged, in `[0, 1]`.

## Usage

```python
import jax
from sarsat import SarSat

env = SarSat(num_satellites=8, max_targets=100)
state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(0))        # 100 targets
state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(0), 60)    # 60 targets, no recompile per count
action = env.action_spec.generate_value()                          # (8, 3)
state, timestep = jax.jit(env.step)(state, action)
```

`max_targets` is the static array capacity; `reset` accepts any `num_targets` up to it.
All constructor arguments are documented on `SarSat.__init__`.

### Access geometry

The sensor is side-looking, as a SAR is, and cannot image at nadir. A target is
accessible when its incidence angle -- measured at the target, between the line of sight
and the local vertical -- falls in `incidence_deg`, its squint (the along-track angle off
broadside) is within `max_squint_deg`, and it is above the horizon. With the defaults at
550 km altitude that is:

| | |
|---|---|
| Incidence 20-45 degrees | look angle 18.4-40.6 degrees at the satellite |
| Cross-track reach | 183-488 km either side of the ground track |
| Along-track reach | +-317 km, against 447 km travelled per step |
| Accessible area | ~404,000 km^2, 0.079% of the globe, per satellite |

The action addresses that region as `[incidence, squint, side, sense]`: `action[0]`
sweeps from the inner to the outer edge of the band, `action[1]` sets the squint, and
`side` is a binary switch (0 left of the ground track, 1 right). Every action is legal.
Because squint and cross-track look combine into the off-nadir angle, the cross-track
range available depends on the squint commanded with it. In MAPX the wrapper declares
this as a hybrid action -- a 2-D Gaussian plus two Bernoullis -- so `HybridActionHead`
handles it unchanged. `side_switch=False` folds the side into the sign of the first
slot instead (DECISIONS.md issue 21 explains why the switch is the default).

Observation layout (`obs_dim = 4 * slots * N + 3 * K + 1`, 96 by default):

```
[ slot 0: (accessible, incidence, squint, side) @ t+1, same @ t+2 | slot 1 ... | slot 9 ... ]
[ neighbour 0: (az, el, distance) | ... | neighbour 4 ... ]
[ battery ]
```

Each slot stores the target's pointing in the action's own encoding, so a slot's three
pointing features followed by its `accessible` flag *is* the action that images it. That
is the whole greedy baseline. That is exactly
what `scripts/greedy_baseline.py` does:

| Scenario (180 steps) | Random policy | Greedy baseline |
|---|---|---|
| 8 satellites, 100 targets | 0.009 | 0.54 |
| 8 satellites, 1000 targets | | 0.32 |
| 64 satellites, 1000 targets | | 0.98 |

## Making cooperation matter

Random orbits over uniform targets give independent greedy satellites little to
coordinate about: leftovers flow to the next satellite and a centralised planner gains
5%. Two constructor options change that (DECISIONS.md issue 22 has the measurements):

```python
env = SarSat(num_satellites=100, max_targets=8000,
             hotspots=40, hotspot_targets=50, hotspot_weight=10.0,  # high-value clusters
             planes=10)                                             # Walker constellation
```

Hotspots make the battery bind *and* make targets differ in value, so which satellite
spends charge on what matters; a planner that knows what teammates will cover beats the
best independent policy by about 10%, while an equally far-sighted *solo* planner gains
almost nothing. `planes=50, train_spacing_s=3.0` flies formation pairs that share one
view, where independent policies waste looks on duplicates and coordination is worth
about 40%. `scripts/cooperation_gap.py` measures the gap for any configuration.

## Evaluation logs and map viewer

```bash
python -m sarsat.evaluate --satellites 8 --targets 100 --seed 0 --policy greedy --out runs/seed0
```

runs one seeded episode and writes, into `runs/seed0/`:

| File | One row per | Columns |
|---|---|---|
| `satellites.csv` | step x satellite | time, sub-satellite lat/lon, altitude, battery, whether it sensed, commanded az/el |
| `looks.csv` | honoured sensing look | commanded az/el, boresight ground point, the four footprint corners, targets hit |
| `targets.csv` | target | lat/lon, land or water, priority, step it was imaged (-1 if never) and by which satellite |
| `captures.csv` | (target, satellite) capture | step, time, position, priority, cumulative return |
| `episode.html` | - | self-contained interactive map of the episode (open directly in a browser) |

A ready-made run ships in `examples/8sat-100tg-seed0/` (the four CSVs of the seed-0
greedy episode below), so the viewer has something to open before you run anything.

There are two ways to get an episode into the viewer:

1. **Generated page.** Every evaluation run writes `episode.html` with the episode
   embedded; double-click it. Nothing else is needed.
2. **Standalone viewer + CSVs.** Open `sarsat/viewer.html` (a copy of it lives in the
   package), click **Load CSVs…** and select the four CSV files of a run, or drag the
   four files onto the page. The file names must contain `satellites`, `looks`, `targets`
   and `captures`. This also works on an `episode.html` to swap in another run.

Programmatically, `write_html(run_episode(env, policy, seed), "episode.html")` produces
the generated page from any `EpisodeLog`.

The viewer animates ground tracks, active looks (footprint, boresight ring and look
direction; green when the look imaged a target), targets flipping from orange to green
as they are imaged, battery levels and the cumulative-return timeline. Hover a
satellite or target for details. Steps are numbered so that the look logged at step `t`
is the action chosen from the observation at `t - 1`, matching the environment.

`run_episode(env, policy, seed)` is the programmatic entry point; `policy` is any
function `Observation -> action` (e.g. a trained MAPX actor wrapped in a closure) or the
name of a `sarsat.reference` yardstick -- `greedy_beam`, `solo_plan`, `coop_dedup`,
`coop_plan` -- which acts from the full environment state. The CLI takes the same
scenario options as the constructor, so the cooperative benchmark's two reference
policies log as:

```bash
python -m sarsat.evaluate --satellites 100 --targets 8000 --hotspots 40 --planes 10 \
    --policy greedy_beam --out runs/hotspots-greedy_beam
python -m sarsat.evaluate --satellites 100 --targets 8000 --hotspots 40 --planes 10 \
    --policy coop_plan --out runs/hotspots-coop_plan
``` The HTML embeds the whole log, so it grows with `steps x satellites`
(~1 MB per 8 steps at 1000 satellites).

## Performance

`python scripts/benchmark.py`, 2-core CPU, no accelerator, random actions:

| Satellites | Targets | Parallel envs | Env steps / s | Agent steps / s |
|---|---|---|---|---|
| 8 | 100 | 1024 | 176,000 | 1.4 M |
| 64 | 1000 | 64 | 6,000 | 381 k |
| 1000 | 1000 | 1 | 460 | 460 k |

## Layout

```
sarsat/env.py        the environment (reset, step, observation)
sarsat/coop.py       CoopSarSat: beam-value / look-ahead observation for MARL training
sarsat/orbits.py     closed-form orbit kinematics, az/el convention, beam test
sarsat/targets.py    land / water target sampling
sarsat/types.py      State, Orbit, Observation
sarsat/windows.py    time-windowed targets: events over a persistent background (DECISIONS 24)
tests/               behavioural tests (pytest)
sarsat/evaluate.py   seeded evaluation: CSV logs and the HTML viewer
sarsat/viewer.html   template for the map viewer (plain canvas, no dependencies)
scripts/             greedy baseline, benchmark, cooperation-gap yardstick, land-mask builder
scripts/baseline_seeds.py  reference-policy returns per seed, for eval_checkpoint.py
scripts/coop_check.py      CoopSarSat sanity check: slot-0 policy vs greedy_beam
scripts/coop_profile.py    per-piece timing of a CoopSarSat step
scripts/gpu_alloc_probe.py how much GPU memory JAX can get under WSL2
scripts/scenario_sweep.py  cooperation gap of candidate scenarios (reference policies)
examples/            CSV logs of one reproducible episode, for the viewer
mapx_integration/    wrapper, Hydra configs and tests to drop into MAPX
```

## Development

```bash
pip install -e ".[dev]"
pytest -q && ruff check . && ruff format --check .
```
