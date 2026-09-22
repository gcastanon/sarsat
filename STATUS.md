# Status — read this first in a new chat

_Last updated: 2026-09-22 (issue 25). Keep this file short and current; history lives in git and
the reasoning lives in DECISIONS.md._

## Where things live

| What | Where | Notes |
|---|---|---|
| Source of truth | this git repository | clone it at the start of a chat, push at the end |
| Design rationale | `DECISIONS.md` (issues 1-21) | append an issue for any non-obvious choice |
| Deferred ideas | `IMPROVEMENTS.md` | move an item out when it is built |
| MAPX integration | `mapx_integration/` | drop-in files for the MAPX repo, see its README |
| Claude Project knowledge | mirrors of the three docs above + this file | search only; not the source of truth |

## Current state

* `sarsat` is a Jumanji-style JAX environment: circular LEO orbits, side-looking SAR
  access (two-sided 20-45 degree incidence band, +-25 degree squint, nadir hole),
  4 x 4 degree beam, battery duty cycle, shared team reward.
* Action `[incidence, squint, side, sense]` (hybrid: 2 continuous + 2 binary);
  `side_switch=False` gives the older signed 3-slot form for comparison.
* Tests pass (`pytest -q`), lint clean (`ruff check . && ruff format --check .`).
* Greedy baseline, 8 satellites / 100 targets / 180 steps: 0.52-0.54. Random: 0.009.
* Cooperative benchmark: `hotspots=40, planes=10` at 100 satellites -- coordination is
  worth ~10% over the best independent policy; formation pairs ~40%
  (`scripts/cooperation_gap.py`, DECISIONS issue 22).
* `sarsat.coop.CoopSarSat` gives a MARL learner a beam-value / look-ahead observation and
  a compact ~133-number global state (vs the 9,600 the plain wrapper's joint observation
  would need at 100 satellites); `mapx_integration/train_sarsat.py` trains `rec_ippo` /
  `rec_mappo` on it with an anchored actor. On seeds 1000-1015 (`greedy_beam` 0.786,
  `coop_plan` 0.904): pure team reward, `rec_mappo` 0.847 (16/16 over `greedy_beam`; IPPO
  stalls at ~0.74); with `+env.credit_mix=0.5` both IPPO and MAPPO reach 0.89-0.90, best
  0.898 with the slot-mixture head (`SARSAT_HEAD=mixture`). DECISIONS issue 23.
* `sarsat.windows` adds time-windowed targets on top of the existing classes; the
  `sarsat-200sat-events` scenario (events over a persistent background, 10% duty cycle)
  gives coop_plan +49-56% over greedy_beam with solo_plan gaining nothing (DECISIONS
  issue 24). Learners reach 0.649 vs coop_plan 0.670 and greedy_beam 0.440 (issue 25).
* Results deck: `reports/sarsat_marl_results.html` (built by `scripts/collect_results.py`
  + `scripts/build_deck.py` from `reports/results.json`).
* `python -m sarsat.evaluate` writes CSV logs and a self-contained HTML map viewer;
  `examples/8sat-100tg-seed0/` holds one reproducible run and a test guards it.

## Working conventions

1. Start: `git clone`, `pip install -e ".[dev]"`, `pytest -q`. Read `DECISIONS.md`.
2. Any behavioural change: update or add a DECISIONS issue, regenerate
   `examples/8sat-100tg-seed0/` (the command is in `examples/README.md`), refresh the
   baseline numbers in `README.md`, rerun the suite.
3. End: commit with a message that says *why*, push, and update this file.

## Open questions

* Side switch vs signed encoding: no training evidence yet (DECISIONS issue 21).
* Can a MARL learner actually capture the ~10% cooperation gap on the hotspot scenario?
  Yes, to within ~0.006 of `coop_plan`, but only once each satellite's own share of the
  team reward is mixed in (`credit_mix`); with the literally shared reward MAPPO gets a bit
  over half way. One training seed per configuration -- replicate before relying on it.
  What is left is same-step duplication and unspent end-of-episode battery (issue 23).
* Does a much larger batch (no WSL2 4 GB allocation cap) make `credit_mix` unnecessary?
* Look-side switching is free; a roll cost or per-pass side is the next realism step.
* Reward is flat across the incidence band; incidence-dependent quality is a one-liner
  in `step` if wanted.
* MAPX proper (Python >= 3.12, JAX >= 0.11) has not been run against the wrapper; only
  a faithful stub of its common wrappers has.
