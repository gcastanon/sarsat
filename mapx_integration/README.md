# Using SarSat from MAPX

1. Install the environment next to MAPX: `pip install -e path/to/sarsat`.
2. Copy this folder's `mapx/` tree over the MAPX repository and the tests into MAPX's
   `tests/` folder (nothing existing is overwritten):

   ```
   mapx/wrappers/sarsat.py
   mapx/wrappers/sarsat_coop.py
   mapx/networks/sarsat.py
   mapx/configs/env/sarsat.yaml
   mapx/configs/env/sarsat_coop.yaml
   mapx/configs/env/scenario/sarsat-{8sat-100tg,64sat-1000tg,1000sat-1000tg}.yaml
   mapx/configs/env/scenario/sarsat-100sat-{hotspots,pairs}.yaml
   mapx/configs/network/sarsat.yaml
   tests/test_sarsat_wrapper.py
   tests/test_sarsat_coop_wrapper.py
   ```

3. Hook the factory into `mapx/utils/make_env.py`. Either

   **(a) built-in** - add these lines to `make()` next to the other families:

   ```python
   if env_name == "SarSat":
       from mapx.wrappers.sarsat import make_sarsat_envs

       return apply_common_wrappers(*make_sarsat_envs(config, add_global_state), config)
   ```

   **(b) no MAPX edits** - register from your own launch script before Hydra runs:

   ```python
   from mapx.systems.ppo.anakin.rec_ippo import hydra_entry_point
   from mapx.utils.make_env import register_env_factory
   from mapx.wrappers.sarsat import make_sarsat_envs

   register_env_factory("SarSat", make_sarsat_envs)
   hydra_entry_point()
   ```

4. Train:

   ```bash
   python -m mapx.systems.ppo.anakin.rec_ippo env=sarsat
   python -m mapx.systems.ppo.anakin.rec_ippo env=sarsat env/scenario=sarsat-64sat-1000tg
   ```

Notes

* The action spec is a `HybridArray` (2 continuous pointing slots + 1 binary sense slot),
  so MAPX picks `HybridActionHead` automatically; the sense slot is masked while the
  battery is too low.
* `eval_metric: episode_return` is the fraction of targets imaged.
* Prefer the independent-critic systems (`rec_ippo`, `rec_imaspo`, `rec_imars`) beyond a
  few tens of satellites: the centralised critics' global state is the joint observation
  (`N * obs_dim` per agent). See DECISIONS.md issue 18.
* Any `SarSat` constructor argument can be set under `kwargs:` in `sarsat.yaml` or under
  `task_config:` in a scenario file.

## Training on the cooperative benchmark

`train_sarsat.py` trains `CoopSarSat` (`sarsat/coop.py`) -- the beam-value / look-ahead
observation and compact global state built for multi-agent learning, DECISIONS.md
issue 23 -- rather than the plain `SarSat` wrapper above. It does not need step 3's
`make_env.py` edit: it registers the `SarSatCoop` env factory
(`mapx.wrappers.sarsat_coop.make_sarsat_coop_envs`) and swaps in the anchored actor and
action head from `mapx.networks.sarsat` itself, before handing off to MAPX's own
`rec_ippo` / `rec_mappo` entry point. Copy the `mapx/` tree over as in step 2 above, then
run everything from *this* folder (`run.sh` re-syncs `mapx/` on every launch, so edits
here are picked up without a manual copy):

```bash
./run.sh ippo  ippo_v1  <hydra overrides...>
./run.sh mappo mappo_v1 <hydra overrides...>
```

`run.sh <system> <run-name> [overrides...]` copies `mapx/` into `$MAPX_ROOT` (default
`~/marl/mapx`), activates `$VENV` (default `~/marl/.venv`), and runs
`train_sarsat.py <system> system.seed=42 [overrides...]` from `$RUNS/<run-name>` (default
`~/marl/runs/<run-name>`), teeing output to `train.log`. The override set used for the
runs reported in DECISIONS.md issue 23:

```bash
./run.sh mappo mappo_v1 \
    arch.num_envs=32 system.rollout_length=180 system.recurrent_chunk_size=30 \
    system.update_batch_size=1 system.num_updates=3000 \
    arch.num_evaluation=60 arch.num_eval_episodes=32 \
    arch.evaluation_greedy=True arch.absolute_metric=False \
    system.gamma=0.995 system.policy_clip_eps=0.1 system.value_clip_eps=10.0 \
    system.actor_lr=3e-4 system.critic_lr=5e-4 \
    logger.checkpointing.save_model=True
```

`arch.num_envs=32` and `system.rollout_length=180` are set by the WSL/GPU constraint
below, not by anything about the environment; a machine without it can raise both.
`env=sarsat_coop` and `network=sarsat` are appended automatically by `train_sarsat.py`
when not given explicitly, so overrides only need to touch `system.*` / `arch.*` /
`logger.*`, or `env.kwargs` / `env.scenario` to change the scenario
(`env/scenario=sarsat-100sat-pairs` for the formation-flying stress test).

**Options that mattered** (DECISIONS.md issue 23 has the measurements):

```bash
# Credit assignment: blend each satellite's own share into the team reward. Off by
# default; with it both IPPO and MAPPO reach coop_plan's level instead of ~0.85 / ~0.74.
./run.sh ippo ippo_credit <overrides above> +env.credit_mix=0.5

# Slot-mixture head (the policy picks which ranked beam to aim at) plus the
# lower-index-goes-first contention feature; best result so far. Evaluate such runs with
# the same SARSAT_HEAD=mixture in front of eval_checkpoint.py / diagnose_policy.py.
SARSAT_HEAD=mixture ./run.sh mappo mappo_mix <overrides above> \
    +env.credit_mix=0.5 +env.kwargs.ranked_contention=True system.decay_learning_rates=True

# Continue from another run's best checkpoint (MAPX keeps the best-evaluated one):
./resume_from.sh mappo_v3 mappo_v4 mappo
./run.sh mappo mappo_v4 <overrides> logger.checkpointing.load_model=True \
    logger.checkpointing.load_args.checkpoint_uid=00_start
```

`stop_run.sh <run> <snapshot-name>` copies a run's checkpoints to `~/marl/snapshots/` and
stops it; `wait_evals.sh <count> <run> [runs...]` blocks until a run has logged that many
evaluations (or anything crashed) and prints every listed run's evaluation history;
`diagnose_policy.py` compares a checkpoint with `greedy_beam` / `coop_plan` look by look
(hotspot vs background coverage, battery use, duplicates). Keep CPU side jobs to one
process (`--workers 1`) while trainings run: WSL2's default 15 GB of host memory is
easily exhausted, and the OOM killer takes the trainings with it.

**GPU co-tenancy on this machine.** Two trainings share the RTX 5070 Ti only when their
resident sets fit with margin: two 100-satellite runs at 24-32 environments, or two
200-satellite runs at 12. Starting a third process (a training, or an evaluation on the
GPU) next to a run holding ~8 GB has twice killed the older run with `cudaErrorUnknown`.
Evaluate on the CPU (`JAX_PLATFORMS=cpu`) while anything trains. `launch.sh` bundles the
issue-23 recipe; `eval_runs.sh` scores several runs' best checkpoints
(`SEED_START/SEED_END/SEED_TAG` select the seed set); `collect_runs.sh` copies run
artefacts into `runs/marl/`. The results deck is rebuilt with
`python scripts/collect_results.py && python scripts/build_deck.py`.

**Watching a run.**

```bash
./status.sh mappo_v1                # actor/evaluator/trainer log lines, tail -100
TAIL=20 ./status.sh mappo_v1 ippo_v1  # several runs, fewer lines each
./watch_runs.sh mappo_v1 ippo_v1     # polls every 30s, prints one line per new eval
```

Both read `$RUNS/<run-name>/train.log` (same `$RUNS` default as `run.sh`) and strip ANSI
colour codes; `watch_runs.sh` also flags a traceback or an out-of-memory line as soon as
one appears.

**Evaluating a checkpoint.** `eval_checkpoint.py` rebuilds the actor exactly as
`learner_setup` does (same anchored network, same hydra config the run was launched
with), restores the latest orbax checkpoint read-only, and replays one full episode per
seed acting greedily -- seeded identically to the reference policies below (the raw
`CoopSarSat` reset with `jax.random.PRNGKey(seed)`, bypassing the wrapper's own key
split), so the numbers are directly comparable:

```bash
JAX_PLATFORMS=cpu python eval_checkpoint.py \
    --run-dir ~/marl/runs/mappo_v1 --system mappo \
    --seeds-start 1000 --seeds-end 1015 --out /tmp/mappo_v1_eval.json
```

It prints a per-seed table against `greedy_beam` / `coop_plan` (loaded from
`runs/baselines/*.jsonl`) plus the mean return and how many of the seeds it beat
`greedy_beam` on, and optionally writes the same data as JSON via `--out`. Must run on
CPU (`JAX_PLATFORMS=cpu`, enforced by the script even if the caller forgets) -- the GPU is
reserved for the training job.

**Baseline reference returns.** `scripts/baseline_seeds.py` (in the `sarsat` repository
root, not this folder) generates the `greedy_beam` / `solo_plan` / `coop_plan` per-seed
numbers `eval_checkpoint.py` and DECISIONS.md issue 23 compare against, on the fixed
`sarsat-100sat-hotspots` scenario:

```bash
python scripts/baseline_seeds.py --seed-start 1000 --seed-end 1015 \
    --output runs/baselines/part0.jsonl
```

It appends one JSON line per `(seed, policy)` to the output file, flushed after every
line, so it can be run in the background and monitored while in flight; `runs/baselines/`
already holds `part0.jsonl`..`part3.jsonl` covering seeds 1000-1015.

**WSL / GPU notes.** Everything above runs from WSL2 against a GPU shared with other
training jobs:

* `run.sh` sets `XLA_PYTHON_CLIENT_ALLOCATOR=platform`. WSL2's default (BFC) allocator
  cannot reserve more than about half the VRAM it is offered, up front or by growing;
  the platform allocator reaches all of it, and a fully jitted learner makes few enough
  allocations that the platform allocator's own overhead does not matter.
* Even with that, a single allocation is capped near 4 GB under WSL2, which is what
  limits a run to `arch.num_envs=32` with `system.rollout_length=180` -- not compute, and
  not anything about `CoopSarSat`. Raise `num_envs` freely on a Linux host or a dedicated
  GPU.
* `scripts/gpu_alloc_probe.py` (in the `sarsat` repository root) measures how many 1 GB
  chunks the current allocator/environment combination can actually hold, if the cap
  needs re-checking on a different machine.
* Always set `JAX_PLATFORMS=cpu` for anything run alongside a training job (tests,
  `eval_checkpoint.py`, ad hoc scripts) -- the GPU is busy training and must not be
  touched by anything else.
