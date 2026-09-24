"""Evaluate a MAPX checkpoint on CoopSarSat with fixed, reproducible per-seed episodes.

Rebuilds the actor network exactly as ``rec_mappo``/``rec_ippo`` build it in
``learner_setup`` (same pre_torso/post_torso/action_head via ``hydra.utils.instantiate``,
same ``AnchoredRecurrentActor``/``AnchoredHybridHead`` overlay used by ``train_sarsat.py``),
restores the actor params from the run's latest orbax checkpoint, and replays one full
episode per seed acting greedily. Episodes are seeded identically to the ``runs/baselines``
reference policies: the underlying ``CoopSarSat`` is reset with exactly
``jax.random.PRNGKey(seed)`` (bypassing ``SarSatCoopWrapper.reset``, which would otherwise
split the key before resetting).

Usage (must run on CPU -- the GPU is reserved for training):

    JAX_PLATFORMS=cpu python eval_checkpoint.py \\
        --run-dir ~/marl/runs/mappo_v1 --system mappo \\
        --seeds-start 1000 --seeds-end 1015 --out /tmp/eval.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# The ``mapx/`` overlay next to this script would shadow the installed package
# (see the same fix at the top of ``train_sarsat.py``).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

# Belt-and-suspenders: the GPU is busy with a training job, force CPU even if the
# caller forgot to set this in the environment. Must happen before jax is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import hydra  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
from mapx.evaluator import make_rec_eval_act_fn  # noqa: E402
from mapx.networks import ScannedRNN  # noqa: E402
from mapx.networks.sarsat import AnchoredRecurrentActor, SlotMixtureActor  # noqa: E402

# Must match the SARSAT_HEAD the run was trained with (see train_sarsat.py).
MIXTURE = os.environ.get("SARSAT_HEAD", "anchored") == "mixture"
from mapx.utils.network_utils import NETWORK_TARGET_WHITELIST, get_action_head  # noqa: E402
from mapx.wrappers.sarsat_coop import CoopWrapperState, make_sarsat_coop_envs  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def find_hydra_config(run_dir: str) -> str:
    """Locate the run's ``.hydra/config.yaml`` copy under ``<run_dir>/outputs``."""
    pattern = os.path.join(run_dir, "outputs", "*", "*", ".hydra", "config.yaml")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No hydra config found under {pattern}")
    # Most recently modified, in case of multiple resumed/rerun outputs.
    return max(matches, key=os.path.getmtime)


def find_checkpoint_dir(run_dir: str, model_name: str) -> str:
    """Locate the (lexicographically latest, i.e. most recent) checkpoint run directory."""
    pattern = os.path.join(run_dir, "checkpoints", model_name, "*")
    matches = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not matches:
        raise FileNotFoundError(f"No checkpoint directories found under {pattern}")
    return sorted(matches)[-1]


def restore_actor_params(checkpoint_dir: str):
    """Read-only restore of the actor params from the latest step in ``checkpoint_dir``.

    Uses orbax directly (rather than ``mapx.utils.checkpointing.Checkpointer``) with
    ``read_only=True`` so this can never write to, clean up, or otherwise disturb a
    checkpoint directory that a live training job may still be writing to.
    """
    options = ocp.CheckpointManagerOptions(read_only=True)
    manager = ocp.CheckpointManager(directory=checkpoint_dir, options=options)
    try:
        step = manager.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoint steps found in {checkpoint_dir}")
        restored = manager.restore(step, args=ocp.args.StandardRestore())
        actor_params = restored["learner_state"]["params"]["actor_params"]
        return actor_params, step
    finally:
        manager.close()


def build_actor(cfg, eval_env):
    """Rebuild the actor network exactly as ``learner_setup`` does, with the anchored
    overlay (``AnchoredRecurrentActor`` / ``AnchoredHybridHead``) that ``train_sarsat.py``
    swaps in for training."""
    actor_pre_torso = hydra.utils.instantiate(
        cfg.network.actor_network.pre_torso, _execution_whitelist_=NETWORK_TARGET_WHITELIST
    )
    actor_post_torso = hydra.utils.instantiate(
        cfg.network.actor_network.post_torso, _execution_whitelist_=NETWORK_TARGET_WHITELIST
    )
    head_cfg, _ = get_action_head(eval_env.action_spec)
    head_name = "SlotMixtureHead" if MIXTURE else "AnchoredHybridHead"
    head_cfg = dict(head_cfg) | {
        "_target_": f"mapx.networks.sarsat.{head_name}",
        "slot_dim": eval_env._env.slot_dim,
    }
    actor_action_head = hydra.utils.instantiate(
        head_cfg, action_dim=eval_env.action_dim, _execution_whitelist_=NETWORK_TARGET_WHITELIST
    )
    return (SlotMixtureActor if MIXTURE else AnchoredRecurrentActor)(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=cfg.network.hidden_state_dim,
    )


def make_episode_fn(eval_env, eval_act_fn, num_agents: int, hidden_state_dim: int, time_limit: int):
    """One full greedy episode, keyed exactly like the baseline evaluations: the raw
    ``CoopSarSat`` is reset with ``jax.random.PRNGKey(seed)`` directly (bypassing
    ``SarSatCoopWrapper.reset``'s internal ``key, reset_key = jax.random.split(key)``)."""

    def run_episode(actor_params, seed: jax.Array) -> jax.Array:
        reset_key = jax.random.PRNGKey(seed)
        env_state, raw_timestep = eval_env._env.reset(reset_key)
        zero = jnp.zeros((), jnp.float32)
        pending = jnp.zeros_like(env_state.cluster_access[:0])  # unused without fused reset
        wrapper_state = CoopWrapperState(reset_key, env_state, zero, reset_key, pending)
        timestep = eval_env._convert(raw_timestep, zero)

        hidden_state = ScannedRNN.initialize_carry((1, num_agents), hidden_state_dim)
        actor_state = {"hidden_state": hidden_state}
        act_key = jax.random.PRNGKey(seed)  # only consumed by jax.random.split below; greedy
        # action selection itself is deterministic.

        def step_fn(carry, _):
            wrapper_state, timestep, actor_state, key = carry
            key, sub_key = jax.random.split(key)
            # Mirror the evaluator: add a size-1 "num_envs" axis (we run one env at a
            # time), then make_rec_eval_act_fn adds the leading time axis of 1 itself.
            batched_timestep = jax.tree_util.tree_map(lambda x: x[jnp.newaxis], timestep)
            action, actor_state = eval_act_fn(actor_params, batched_timestep, sub_key, actor_state)
            action = action[0]
            wrapper_state, timestep = eval_env.step(wrapper_state, action)
            return (wrapper_state, timestep, actor_state, key), None

        init_carry = (wrapper_state, timestep, actor_state, act_key)
        (final_wrapper_state, _, _, _), _ = jax.lax.scan(
            step_fn, init_carry, None, length=time_limit
        )
        # Unscaled cumulative team reward == the final `fraction_imaged` env metric.
        return final_wrapper_state.episode_return

    return jax.jit(run_episode)


def load_baselines(repo_root: str, scenario: str = "") -> dict:
    """Load per-seed baseline returns from ``runs/baselines/*.jsonl``; rows that carry a
    ``scenario`` field are kept only when it matches ``scenario``."""
    folder = os.path.join(repo_root, "runs", "baselines")
    candidates = sorted(glob.glob(os.path.join(folder, "*.jsonl")))
    # Rows without a scenario field predate the field and belong to the 100-satellite
    # hotspot benchmark; ``baseline_seeds.py`` names the scenarios ``hotspots100`` etc.
    aliases = {
        "sarsat-100sat-hotspots": "hotspots100",
        "sarsat-200sat-events": "events200",
        "sarsat-500sat-events": "events500",
        "sarsat-500sat-announced": "announced500",
    }
    wanted = aliases.get(scenario, scenario)
    baselines: dict[int, dict[str, float]] = {}
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if scenario and row.get("scenario", "hotspots100") != wanted:
                    continue
                seed = int(row["seed"])
                baselines.setdefault(seed, {})[row["policy"]] = float(row["return"])
    return baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", required=True, help="Hydra run directory, e.g. ~/marl/runs/mappo_v1"
    )
    parser.add_argument("--system", required=True, choices=["mappo", "ippo"])
    parser.add_argument("--seeds-start", type=int, required=True)
    parser.add_argument("--seeds-end", type=int, required=True, help="Inclusive.")
    parser.add_argument("--out", default=None, help="Optional path to write results as JSON.")
    args = parser.parse_args()

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    model_name = f"rec_{args.system}"

    config_path = find_hydra_config(run_dir)
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    print(f"Loaded hydra config from {config_path}")

    checkpoint_dir = find_checkpoint_dir(run_dir, model_name)
    print(f"Restoring actor params from {checkpoint_dir}")
    actor_params, step = restore_actor_params(checkpoint_dir)
    print(f"Restored checkpoint step {step}")

    add_global_state = args.system == "mappo"
    _, eval_env = make_sarsat_coop_envs(cfg, add_global_state=add_global_state)

    actor_network = build_actor(cfg, eval_env)
    eval_act_fn = make_rec_eval_act_fn(actor_network.apply, cfg)

    episode_fn = make_episode_fn(
        eval_env,
        eval_act_fn,
        num_agents=eval_env.num_agents,
        hidden_state_dim=int(cfg.network.hidden_state_dim),
        time_limit=eval_env.time_limit,
    )

    seeds = list(range(args.seeds_start, args.seeds_end + 1))
    scenario = str(cfg.env.scenario.get("task_name", ""))
    baselines = load_baselines(os.path.dirname(_here), scenario)

    results = []
    for seed in seeds:
        ret = float(episode_fn(actor_params, seed))
        base = baselines.get(seed, {})
        results.append(
            {
                "seed": seed,
                "policy_return": ret,
                "greedy_beam": base.get("greedy_beam"),
                "coop_plan": base.get("coop_plan"),
                "solo_plan": base.get("solo_plan"),
            }
        )

    header = (
        f"{'seed':>6} {'policy':>10} {'greedy_beam':>12} {'coop_plan':>10} {'beats_greedy':>13}"
    )
    print(header)
    print("-" * len(header))
    beats_greedy = 0
    n_with_greedy = 0
    for r in results:
        gb = r["greedy_beam"]
        beat = ""
        if gb is not None:
            n_with_greedy += 1
            won = r["policy_return"] > gb
            beats_greedy += int(won)
            beat = "yes" if won else "no"
        gb_str = f"{gb:.4f}" if gb is not None else "n/a"
        cp_str = f"{r['coop_plan']:.4f}" if r["coop_plan"] is not None else "n/a"
        print(f"{r['seed']:>6} {r['policy_return']:>10.4f} {gb_str:>12} {cp_str:>10} {beat:>13}")

    def mean(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    mean_policy = mean("policy_return")
    mean_gb = mean("greedy_beam")
    mean_cp = mean("coop_plan")
    print("-" * len(header))
    print(
        f"means: policy={mean_policy:.4f} "
        f"greedy_beam={'n/a' if mean_gb is None else f'{mean_gb:.4f}'} "
        f"coop_plan={'n/a' if mean_cp is None else f'{mean_cp:.4f}'}"
    )
    print(f"policy beats greedy_beam on {beats_greedy}/{n_with_greedy} seeds")

    if args.out:
        out_path = os.path.abspath(os.path.expanduser(args.out))
        payload = {
            "run_dir": run_dir,
            "system": args.system,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_step": step,
            "seeds_start": args.seeds_start,
            "seeds_end": args.seeds_end,
            "results": results,
            "summary": {
                "mean_policy_return": mean_policy,
                "mean_greedy_beam": mean_gb,
                "mean_coop_plan": mean_cp,
                "policy_beats_greedy_count": beats_greedy,
                "policy_beats_greedy_total": n_with_greedy,
            },
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
