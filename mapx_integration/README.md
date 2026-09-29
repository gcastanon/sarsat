# Training SarSat with MAPX

Everything here runs from this folder on Linux or WSL2 with a CUDA JAX. `train_sarsat.py`
trains `CoopSarSat` (the observation and global state built for multi-agent learning,
DECISIONS.md issue 23) with MAPX's `rec_ippo` / `rec_mappo`; the `mapx/` tree beside it is
an overlay of wrappers, networks and Hydra configs that the launch script copies into the
MAPX checkout, so no MAPX file is edited by hand.

## 1. Setup

MAPX needs Python 3.12. Its repository is private: ask its owner for access.

```bash
git clone git@github.com:Chulabhaya/mapx.git ~/marl/mapx
uv venv ~/marl/.venv --python 3.12 && source ~/marl/.venv/bin/activate
uv pip install "jax[cuda12]==0.11.2" flax==0.12.9 optax==0.2.8 chex==0.1.92 jumanji==1.1.2 \
  hydra-core==1.4.0.dev10 omegaconf==2.4.0.dev15 orbax-checkpoint==0.12.4 \
  tfp-nightly==0.26.0.dev20260921 numpy==2.5.3 absl-py==2.5.0 colorama==0.4.6
uv pip install -e ~/marl/mapx -e /path/to/sarsat
```

This is the known-good version set of the Runpod runs (every run used JAX 0.11.2); plain
`pip` works too. For a rented GPU (Runpod), `CLAUDE.md` has the same setup with the paths used
there, plus measured throughput and cost.

| Variable | Default | Meaning |
|---|---|---|
| `MAPX_ROOT` | `~/marl/mapx` | MAPX checkout; `run.sh` copies `mapx/` into it on every launch |
| `VENV` | `~/marl/.venv` | virtualenv activated by `run.sh` |
| `RUNS` | `~/marl/runs` | one folder per run: `train.log`, `checkpoints/`, Hydra `outputs/` |

## 2. Train

```bash
./launch.sh ippo  ippo_v1  sarsat-100sat-hotspots
./launch.sh mappo mappo_v1 sarsat-200sat-events
```

`launch.sh <ippo|mappo> <run> <scenario> [hydra overrides...]` bakes in the recipe that
reached the results below; anything after the scenario is a Hydra override.

| Baked in | Value |
|---|---|
| Action head | slot mixture (`SARSAT_HEAD=mixture`; `anchored` for the older slot-0 head) |
| Credit assignment | `+env.credit_mix=0.5`: each satellite's own share blended into the team reward |
| Batch | 32 envs x 180-step rollouts (16 envs at 200 satellites, 4 at 500), 4 minibatches, 4 epochs |
| Schedule | 1500 updates, evaluation every 30, learning-rate decay, gamma 0.995 |
| Checkpoint | the best 32-episode evaluation is kept under `checkpoints/` |

| Scenario (`mapx/configs/env/scenario/`) | Use |
|---|---|
| `sarsat-100sat-hotspots` | benchmark, DECISIONS.md issues 22-23 |
| `sarsat-200sat-events` | benchmark with time-windowed events, issues 24-25 |
| `sarsat-500sat-events` | largest benchmark, issues 26-27. `launch.sh` defaults to 4 environments, a guess at the WSL2 allocation cap; the trained run used `arch.num_envs=16` on a 48 GB GPU (~35 GB used) |
| `sarsat-500sat-announced` | the 500-satellite field with every request windowed and announced at random at least 30 minutes ahead, issue 30; not trained yet, the run plan is `campaign_plan_ann500.json` |
| `sarsat-100sat-pairs` | 50 formation pairs 3 s apart: a coordination stress test, not a realistic constellation (issue 22) |
| `sarsat-8sat-100tg`, `sarsat-64sat-1000tg`, `sarsat-1000sat-1000tg` | plain uniform targets, for smoke tests and scaling |

`run.sh` is the same launcher without the recipe, for raw overrides. Follow a run with:

```bash
./watch_runs.sh ippo_v1            # one line per new evaluation, flags crashes and OOMs
./status.sh ippo_v1                # actor / evaluator / trainer history so far
./wait_evals.sh 10 ippo_v1         # block until 10 evaluations have been logged
```

Expected on `sarsat-100sat-hotspots`: `Fraction imaged mean` above `greedy_beam` (0.79)
within the first evaluations, 0.89-0.91 by update 300. On the paired seeds 1000-1015 the
best IPPO and MAPPO checkpoints score 0.895-0.898 against `coop_plan` 0.904; on
`sarsat-200sat-events`, 0.645-0.649 against `coop_plan` 0.670 and `greedy_beam` 0.440.
DECISIONS.md issues 23-25 have every run.

## 3. Evaluate

```bash
JAX_PLATFORMS=cpu ./eval_runs.sh ippo ippo_v1           # -> ~/marl/runs/ippo_v1/eval_seeds.json
JAX_PLATFORMS=cpu python ../scripts/export_viewer_runs.py --run ~/marl/runs/ippo_v1 --system ippo
JAX_PLATFORMS=cpu python diagnose_policy.py --run-dir ~/marl/runs/ippo_v1 --system ippo --workers 1
```

`eval_runs.sh` replays the best checkpoint greedily on seeds 1000-1015 (`SEED_START`,
`SEED_END`, `SEED_TAG` change the set) and prints a per-seed table against `greedy_beam`,
`solo_plan`, `coop_dedup` and `coop_plan`, seeded identically so the numbers are directly comparable. The export writes
the four CSVs and `episode.html` for the map viewer into `runs/viewer/`. `diagnose_policy.py`
says where a policy loses value against `coop_plan`: hotspot vs background coverage,
battery timing, same-step duplicates. All three run on the CPU, so they are safe beside a
training job. The reference returns they compare against come from
`scripts/baseline_seeds.py` and live in `runs/baselines/`.

## 4. Manage runs

| Script | Does |
|---|---|
| `stop_run.sh <run> <snapshot>` | copies the run's checkpoints to `~/marl/snapshots/<snapshot>` and stops it |
| `resume_from.sh <src> <dst> <ippo\|mappo>` | stages `src`'s checkpoint so `dst` starts from it (then launch `dst` with `logger.checkpointing.load_model=True logger.checkpointing.load_args.checkpoint_uid=00_start`) |
| `collect_runs.sh <run...>` | copies logs, configs, checkpoints and evaluation JSONs into the repo's git-ignored `runs/marl/` |
| `tests/` | wrapper tests; copy them into MAPX's `tests/` and run them there |

## 5. Limits worth knowing

* **Batch size is set by WSL2, not the environment.** A single allocation is capped near
  4 GB under WSL2, which is what pins `num_envs` to 32 (16 at 200 satellites). A Linux host
  or a dedicated GPU can raise it. `run.sh` already selects the platform allocator, without
  which JAX reaches only about half the VRAM; `scripts/gpu_alloc_probe.py` re-measures the cap.
* **Never put a third process on the GPU.** Two trainings share a 16 GB card when their
  resident sets fit; a third GPU process beside one holding ~8 GB has killed the older
  run with `cudaErrorUnknown`. Evaluate with `JAX_PLATFORMS=cpu` while anything trains.
* **Host RAM is the other ceiling.** WSL2's default 15 GB is exhausted by two CPU jobs;
  run side jobs one at a time (`--workers 1`) or the OOM killer takes the trainings too.
* **Always pass one seed.** MAPX's default `system.seed` is a list of ten; `run.sh` sets
  `system.seed=42`. Restoring a checkpoint needs the `arch.num_envs` it was trained with.

## Plain `SarSat` without the CoopSarSat observation

`mapx/wrappers/sarsat.py` exposes the raw environment (`env=sarsat`) with its hybrid
action spec, so MAPX picks `HybridActionHead` itself. Register it from a launch script
with `make_env.register_env_factory("SarSat", make_sarsat_envs)` before Hydra runs, or add
the equivalent branch to `mapx/utils/make_env.py`, then
`python -m mapx.systems.ppo.anakin.rec_ippo env=sarsat env/scenario=sarsat-64sat-1000tg`.
Prefer the independent-critic systems beyond a few tens of satellites: the centralised
critics' global state is the joint observation (issue 18).
