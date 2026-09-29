# Status

_Last updated: 2026-09-29 (issue 30). Read this first when picking the project up. Keep it
short and current: history lives in git, reasoning in DECISIONS.md, results in README.md._

## Where things live

| What | Where | Notes |
|---|---|---|
| Source of truth | this git repository | pull at the start of a session, push at the end |
| Overview and results | `README.md` | quick start, scenarios, the results table |
| Design rationale | `DECISIONS.md` (issues 1-30) | append an issue for any non-obvious choice |
| Codebase guide | `docs/index.html` | code map, data flow, experiment / training / evaluation walkthrough; update it when those change |
| Deferred ideas | `IMPROVEMENTS.md` | move an item out when it is built |
| Training | `mapx_integration/` | overlay files for a MAPX checkout, launch and evaluation scripts; see its README |
| Live demo | `sarsat/live.py` here; server, web UI and NL parser in [sarsat-live](https://github.com/gcastanon/sarsat-live) | rolling episodes stay here (tested); the demo pins a sarsat release (issue 29) |
| Rented-GPU workflow | `CLAUDE.md` | Runpod setup, pinned versions, measured throughput and cost |
| Claude Project knowledge | copies of the docs above | search only; not the source of truth |

## Current state

* **Environment.** `sarsat.SarSat` is a Jumanji-style JAX environment: circular LEO orbits,
  two-sided SAR access (20-45 degree incidence, +-25 degree squint, nadir hole), a 4 x 4
  degree beam, a battery duty cycle and a shared team reward. `CoopSarSat` (issue 23) adds
  the observation and compact global state the learners train on; `sarsat.windows` adds
  time-windowed event targets (issue 24); `sarsat.announce` reveals each request at a random
  time before its window opens (issue 30).
* **Benchmarks.** `sarsat-100sat-hotspots` (issue 22), `sarsat-200sat-events` (issue 24)
  and `sarsat-500sat-events` (issue 26, where both planning ahead and cooperating pay).
  Reference policies in `sarsat.reference`; at 500 satellites they need
  `lean_precompute` (~4 GB of tables).
* **Learners.** MAPX `rec_mappo` / `rec_ippo` with the `launch.sh` recipe (slot-mixture
  head, `credit_mix=0.5`) reach 97-99% of the centralised planner `coop_plan` on all three
  benchmarks and beat every independent reference on 16/16 test seeds (issues 23, 25, 27;
  numbers in README.md). The 500-satellite run trained on a rented A40 in 2.4 h for $1.19.
* **Announced requests** (issue 30). `sarsat-500sat-announced` is the 500-satellite field
  with every request windowed (events 10-30 steps, background 30-90) and announced at least
  30 minutes ahead; the observation, actor and critic alike, uses nothing about a request
  before then. References on seeds 1000-1015: `greedy_beam` 0.372, `coop_dedup` 0.397,
  `solo_plan` 0.645, `coop_plan` 0.792 (the last two clairvoyant). **No learner trained
  yet**: the planned run is `ann500_mappo_a` (`mapx_integration/campaign_plan_ann500.json`:
  one A40, ~3.6 h, ~$1.76, cap 7 h), launched from the PC, which holds SSH, MAPX and the
  spending ledger.
* **Live demo** (issues 28-29). `sarsat.live.LiveWorld` turns the episodic environment
  into a world that never ends; typed requests become time-windowed target clusters. The
  server (60x real time, 83 ms per 500-satellite step on a Windows CPU), web UI and
  offline NL parser live in the `sarsat-live` repository, which drives `LiveWorld` from a
  sarsat release and reads the parameter snapshot from this checkout's `runs/`.
  Environment-side changes for the demo are made here.
* **Tooling.** `python -m sarsat.evaluate` writes CSV logs and a self-contained HTML map
  viewer; `examples/` holds recorded episodes. The results deck
  `reports/sarsat_marl_results.html` is built by `scripts/collect_results.py` +
  `scripts/build_deck.py` from `reports/results.json`.
* **Checks.** `pytest -q` passes (`mapx_integration/tests/` run inside a MAPX checkout);
  `ruff check .` and `ruff format --check .` are clean.

## Working conventions

1. Start: pull, `pip install -e ".[dev]"`, `pytest -q`. Skim `DECISIONS.md`.
2. Any behavioural change: update or add a DECISIONS issue, regenerate
   `examples/8sat-100tg-seed0/` (the command is in `examples/README.md`), refresh the
   numbers in `README.md`, rerun the suite.
3. Keep new environment features back-compatible: add a subclass or a flag (as
   `sarsat/windows.py` does) rather than change existing observation layouts, so trained
   checkpoints keep loading.
4. End: commit with a message that says *why*, push, and update this file.
5. Release: bump `version` in `pyproject.toml` and `__version__` in `sarsat/__init__.py` in
   a PR; on merge `.github/workflows/tag-release.yml` tags the commit `vX.Y.Z` (Claude
   sessions cannot push tags themselves). Raise `sarsat-live`'s `sarsat>=` pin with it.

## Open questions

* **Announced requests.** Does MAPPO still get close to `coop_plan` when requests are
  revealed over time (issue 30)? Launch `ann500_mappo_a` and replay it on seeds 1000-1015.
* **Replication.** Most recipes have one training seed; the hotspot recipe has three
  (0.894-0.898). Replicate before relying on a small difference.
* **`credit_mix`.** Pure team reward gets MAPPO only a bit over half way from
  `greedy_beam` to `coop_plan` at 100 satellites; blending in each satellite's own share
  closes it (issue 23). Would a much larger batch make it unnecessary?
* **Battery at episode end.** The trained 500-satellite policy ends episodes at ~8%
  charge, so the continuous live world loses ~4% per episode against the offline score
  (issue 28). Training from a random initial charge should fix it.
* **Side switch vs signed action encoding.** No training evidence either way (issue 21).
* **Realism next steps.** Look-side switching is free (a roll cost or per-pass side
  would be more honest); reward is flat across the incidence band. Both are listed in
  `IMPROVEMENTS.md`.
* **NL tasking.** Rules + gazetteer score 100% on hand-written phrases and 96.7% on a
  synthetic set, where every miss is a homonym (Birmingham UK vs Alabama); details in the
  `sarsat-live` README.
