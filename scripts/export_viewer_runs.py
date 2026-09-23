"""Export map-viewer episode logs (CSVs + ``episode.html``) for trained MARL policies and
reference (``sarsat.reference``) policies, one folder per (scenario, policy).

For a fixed ``--seed`` (default 1000) this writes, under ``--out-dir`` (default
``runs/viewer``):

* ``hotspots100-greedy_beam``, ``hotspots100-coop_plan`` -- reference policies on the
  100-satellite hotspot scenario (plain :class:`sarsat.env.SarSat`, so ``targets.csv`` has
  no window columns).
* ``hotspots100-mappo_v5`` -- the trained checkpoint at ``~/marl/runs/mappo_v5``.
* ``events200-greedy_beam``, ``events200-coop_plan`` -- reference policies on the
  200-satellite windowed-events scenario (:class:`sarsat.windows.WindowedSarSat`, so
  ``targets.csv`` carries ``window_open``/``window_close``).
* ``events200-ev_mappo_e`` -- the trained checkpoint at ``~/marl/runs/ev_mappo_e``.

The trained policies were both trained with the mixture action head
(``SARSAT_HEAD=mixture``), so this script sets that before importing anything from
``mapx_integration`` (which reads it once, at import time). The scenario kwargs and the
lean (int32/float32) ``reference.precompute`` monkeypatch are reused from
``scripts/baseline_seeds.py``, but only applied for events200: the lean tables trade
precision for memory, and at 100 satellites the recorded ``runs/baselines`` values for
hotspots100 (used below to verify ``coop_plan``) turn out to have been generated with the
default full-precision (float64) ``precompute`` -- switching it on for hotspots100 too
reproduces a slightly different ``coop_plan`` trajectory (same target field, a handful of
different capture choices) whose return is off by ~2e-4, just over this script's 1e-4
tolerance. At 200 satellites the full-precision tables are what risk the 15 GB host
budget, so events200 keeps the lean version, matching how its baselines were produced.

Run (CPU only -- the GPU may be busy training)::

    JAX_PLATFORMS=cpu python scripts/export_viewer_runs.py --seed 1000

``--run <run-dir> --system ippo|mappo [--scenario hotspots100|events200]`` exports just
that training run's best checkpoint instead, into ``<scenario>-<run-dir name>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Both trained runs below were trained with the mixture head; ``eval_checkpoint`` reads
# this env var exactly once, at import time, so it must be set first.
os.environ.setdefault("SARSAT_HEAD", "mixture")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sarsat.reference as reference  # noqa: E402

_default_precompute = reference.precompute  # captured before baseline_seeds monkeypatches it

from baseline_seeds import SCENARIOS, lean_precompute  # noqa: E402 (see module docstring)

from sarsat.env import SarSat  # noqa: E402
from sarsat.evaluate import run_episode, write_csv, write_html  # noqa: E402
from sarsat.windows import WindowedSarSat  # noqa: E402

# ``eval_checkpoint.py`` lives in mapx_integration/, not a package; its own sys.path fix
# (dropping mapx_integration back out) runs the moment it is imported, so "mapx.*" below
# resolves to the real installed package rather than the mapx_integration/mapx overlay.
sys.path.insert(0, str(REPO_ROOT / "mapx_integration"))
import eval_checkpoint as ec  # noqa: E402
from mapx.types import Observation, ObservationGlobalState  # noqa: E402

SCENARIO_TITLE = {
    "hotspots100": "sarsat-100sat-hotspots",
    "events200": "sarsat-200sat-events",
    "events500": "sarsat-500sat-events",
}
# Only the 100-satellite references use the full-precision planner tables (see above).
LEAN_SCENARIOS = ("events200", "events500")
TRAINED_RUNS = {
    "hotspots100": ("mappo_v5", "~/marl/runs/mappo_v5"),
    "events200": ("ev_mappo_e", "~/marl/runs/ev_mappo_e"),
}
REFERENCE_POLICIES = ("greedy_beam", "coop_plan")


def build_reference_env(scenario: str):
    """Plain ``SarSat`` for hotspots100 (so ``targets.csv`` carries no window columns),
    ``WindowedSarSat`` for events200 / events500 (whose hotspots are windowed events)."""
    kwargs = dict(SCENARIOS[scenario])
    if scenario == "hotspots100":
        return SarSat(time_limit=180, **kwargs)
    return WindowedSarSat(time_limit=180, **kwargs)


class TrainedPolicy:
    """A MAPX recurrent actor wrapped as a ``sarsat.evaluate.StatefulPolicy``.

    Mirrors ``mapx.evaluator.make_rec_eval_act_fn``'s greedy path (these runs were saved
    with ``arch.evaluation_greedy: true``) and ``eval_checkpoint.make_episode_fn``'s
    batching exactly: the actor is applied under a leading ``(time=1, env=1)`` axis pair
    over its own hidden state, and the distribution's mode is taken.
    """

    def __init__(
        self,
        actor_apply,
        actor_params,
        num_agents: int,
        hidden_state_dim: int,
        has_global_state: bool,
    ) -> None:
        # ``actor_apply`` returns a distribution object (e.g. ``SlotMixtureDistribution``),
        # which isn't a valid jit output; take the mode inside the jitted function instead
        # and only cross the jit boundary with arrays.
        def apply_and_mode(params, hidden_state, ac_in):
            hidden_state, pi = actor_apply(params, hidden_state, ac_in)
            return hidden_state, pi.mode()

        self._apply = jax.jit(apply_and_mode)
        self._params = actor_params
        self._num_agents = num_agents
        self._hidden_state_dim = hidden_state_dim
        self._has_global_state = has_global_state
        self._hidden_state = None

    def reset(self) -> None:
        self._hidden_state = ec.ScannedRNN.initialize_carry(
            (1, self._num_agents), self._hidden_state_dim
        )

    def __call__(self, obs) -> jax.Array:
        if self._has_global_state:
            wrapped = ObservationGlobalState(
                obs.agents_view, obs.action_mask, obs.global_state, obs.step_count
            )
        else:
            wrapped = Observation(obs.agents_view, obs.action_mask, obs.step_count)
        env_batched = jax.tree_util.tree_map(lambda x: x[jnp.newaxis], wrapped)  # (1, N, ...)
        done = jnp.zeros((1, self._num_agents), dtype=bool)  # never done mid-episode here
        ac_in = jax.tree_util.tree_map(lambda x: x[jnp.newaxis], (env_batched, done))
        self._hidden_state, action = self._apply(self._params, self._hidden_state, ac_in)
        return action[0, 0]


def build_trained_policy(run_dir: str, system: str = "mappo"):
    """Rebuild the actor and restore its params exactly as ``eval_checkpoint.main`` does;
    returns ``(policy, env, checkpoint_dir, step)`` where ``env`` is the raw
    ``CoopSarSat``/``WindowedCoopSarSat`` (``wrapper._env``)."""
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    model_name = f"rec_{system}"
    config_path = ec.find_hydra_config(run_dir)
    cfg = ec.OmegaConf.load(config_path)
    ec.OmegaConf.set_struct(cfg, False)
    checkpoint_dir = ec.find_checkpoint_dir(run_dir, model_name)
    actor_params, step = ec.restore_actor_params(checkpoint_dir)

    add_global_state = system == "mappo"
    _, eval_env = ec.make_sarsat_coop_envs(cfg, add_global_state=add_global_state)
    actor_network = ec.build_actor(cfg, eval_env)
    hidden_state_dim = int(cfg.network.hidden_state_dim)
    policy = TrainedPolicy(
        actor_network.apply, actor_params, eval_env.num_agents, hidden_state_dim, add_global_state
    )
    return policy, eval_env._env, checkpoint_dir, step


def load_trained_expected(run_dir: str, seed: int) -> Optional[float]:
    """The saved ``policy_return`` for ``seed`` in a run's ``eval_seeds.json``, if any."""
    path = Path(os.path.expanduser(run_dir)) / "eval_seeds.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    for row in payload.get("results", []):
        if int(row["seed"]) == seed:
            return float(row["policy_return"])
    return None


def export_one(scenario: str, policy_name: str, env, policy, seed: int, out_dir: Path) -> dict:
    title = f"{SCENARIO_TITLE[scenario]} · {policy_name} · seed {seed}"
    folder = out_dir / f"{scenario}-{policy_name}"
    start = time.perf_counter()
    log = run_episode(env, policy, seed=seed)
    elapsed = time.perf_counter() - start
    paths = write_csv(log, folder)
    paths.append(write_html(log, folder / "episode.html", title))
    return {
        "scenario": scenario,
        "policy": policy_name,
        "return": float(log.reward.sum()),
        "elapsed_s": elapsed,
        "paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/viewer"))
    parser.add_argument(
        "--run",
        default=None,
        help="Export only this training run's best checkpoint (a run directory such as "
        "~/marl/runs/ippo_v1), named after the directory, instead of the built-in set.",
    )
    parser.add_argument("--system", choices=["ippo", "mappo"], default="mappo")
    parser.add_argument("--scenario", choices=sorted(SCENARIO_TITLE), default="hotspots100")
    args = parser.parse_args()

    rows = []
    if args.run is not None:
        scenario = args.scenario
        reference.precompute = (
            lean_precompute if scenario in LEAN_SCENARIOS else _default_precompute
        )
        label = Path(os.path.expanduser(args.run)).name
        print(f"running {scenario} / {label} (loading {args.run}) ...", flush=True)
        policy, trained_env, checkpoint_dir, step = build_trained_policy(args.run, args.system)
        print(f"  checkpoint {checkpoint_dir} (step {step})", flush=True)
        row = export_one(scenario, label, trained_env, policy, args.seed, args.out_dir)
        row["expected"] = load_trained_expected(args.run, args.seed)
        report([row], args.seed)
        return

    for scenario in ("hotspots100", "events200"):
        # See the module docstring: only the event scenarios need the lean tables.
        reference.precompute = (
            lean_precompute if scenario in LEAN_SCENARIOS else _default_precompute
        )
        env = build_reference_env(scenario)
        for policy_name in REFERENCE_POLICIES:
            print(f"running {scenario} / {policy_name} ...", flush=True)
            rows.append(
                export_one(scenario, policy_name, env, policy_name, args.seed, args.out_dir)
            )

        policy_label, run_dir = TRAINED_RUNS[scenario]
        print(f"running {scenario} / {policy_label} (loading {run_dir}) ...", flush=True)
        policy, trained_env, checkpoint_dir, step = build_trained_policy(run_dir, system="mappo")
        print(f"  checkpoint {checkpoint_dir} (step {step})", flush=True)
        row = export_one(scenario, policy_label, trained_env, policy, args.seed, args.out_dir)
        row["expected"] = load_trained_expected(run_dir, args.seed)
        rows.append(row)
    report(rows, args.seed)


def report(rows: list, seed: int) -> None:
    """Print each exported episode's return next to its known value, if any."""
    # Known seed-1000 returns to check against: greedy_beam/coop_plan from
    # runs/baselines/*.jsonl, the trained policy from its run's eval_seeds.json.
    for row in rows:
        if "expected" not in row:
            baselines = ec.load_baselines(str(REPO_ROOT), row["scenario"])
            row["expected"] = baselines.get(seed, {}).get(row["policy"])

    print()
    header = f"{'scenario':12s} {'policy':12s} {'return':>10s} {'expected':>10s} {'files':>7s}"
    print(header)
    print("-" * len(header))
    mismatches = []
    for row in rows:
        sizes = sum(p.stat().st_size for p in row["paths"])
        expected = row["expected"]
        exp_str = f"{expected:.4f}" if expected is not None else "n/a"
        flag = ""
        if expected is not None and abs(row["return"] - expected) > 1e-4:
            flag = "  MISMATCH"
            mismatches.append(row)
        print(
            f"{row['scenario']:12s} {row['policy']:12s} {row['return']:>10.4f} "
            f"{exp_str:>10s} {sizes / 1024:>6.0f}k{flag}"
        )
        for p in row["paths"]:
            print(f"    {p} ({p.stat().st_size:,} bytes)")

    if mismatches:
        print(f"\n{len(mismatches)} scenario/policy pair(s) mismatched expected returns.")
    else:
        print("\nAll returns matched their expected values within 1e-4.")


if __name__ == "__main__":
    main()
