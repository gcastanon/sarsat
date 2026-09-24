"""The trained MAPPO actor as a per-step policy, loaded without hydra or orbax.

``mapx_integration/eval_checkpoint.py`` rebuilds the actor by instantiating the run's hydra
config and restores it from a 1.7 GB orbax checkpoint. A live server needs neither: the
network is four modules with fixed sizes (``runs/<run>/outputs/*/.hydra/config.yaml``,
``network:``), and the params are the 5.4 MB flax msgpack snapshot the training pod wrote
(``runs/<run>/sync/params_<step>.msgpack``). The module and parameter names below match that
snapshot's tree (``pre_torso``, ``ScannedRNN_0``, ``post_torso``, ``action_head``).

Usage::

    env = make_env()                              # the checkpoint's scenario
    policy = load_policy("runs/ev500_mappo_a/sync/params_1728000.msgpack", env.num_agents)
    policy.reset()                                # at every episode start
    action = policy(observation)                  # CoopObservation -> (N, 4)

``python -m live.policy`` replays seed 1000 through ``sarsat.evaluate.run_episode`` and
compares the return with ``runs/ev500_mappo_a/eval_seeds.json``.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
from flax import serialization
from mapx.networks.base import ScannedRNN
from mapx.networks.sarsat import SarSatTorso, SlotMixtureActor, SlotMixtureHead
from mapx.networks.torsos import MLPTorso
from mapx.types import ActionBlock, ActionLayout, Observation

from sarsat.windows import WindowedCoopSarSat

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUN = os.path.join(REPO, "runs", "ev500_mappo_a")
DEFAULT_PARAMS = os.path.join(DEFAULT_RUN, "sync", "params_1728000.msgpack")

# scenario/sarsat-500sat-events.yaml task_config + env/sarsat_coop.yaml kwargs + the
# launch.sh override (ranked_contention); see runs/ev500_mappo_a/outputs/*/.hydra/config.yaml.
SCENARIO: dict[str, Any] = dict(
    num_satellites=500,
    max_targets=20000,
    hotspots=100,
    hotspot_targets=50,
    hotspot_weight=10.0,
    planes=50,
    window_steps=(10, 30),
    background_windows=False,
    recharge_rate=0.005,
    time_limit=180,
    num_slots=4,
    num_candidates=64,
    horizon=16,
    ranked_contention=True,
)

# network: in the same config.
NETWORK: dict[str, Any] = dict(
    hidden_state_dim=128,
    pre_layer_sizes=(256, 128),
    post_layer_sizes=(128,),
)


def make_env(**overrides: Any) -> WindowedCoopSarSat:
    """The checkpoint's environment; ``overrides`` replace scenario keys."""
    return WindowedCoopSarSat(**(SCENARIO | overrides))


def build_actor(env: WindowedCoopSarSat) -> SlotMixtureActor:
    """``SlotMixtureActor`` sized for ``env`` (slot width, slot count, horizon)."""
    layout = ActionLayout(
        (
            ActionBlock(size=2),
            ActionBlock(size=1, num_categories=2),
            ActionBlock(size=1, num_categories=2),
        )
    )
    return SlotMixtureActor(
        pre_torso=SarSatTorso(
            num_slots=env.num_slots,
            slot_dim=env.slot_dim,
            horizon=env.horizon,
            layer_sizes=tuple(NETWORK["pre_layer_sizes"]),
        ),
        post_torso=MLPTorso(
            layer_sizes=tuple(NETWORK["post_layer_sizes"]), use_layer_norm=True, activation="relu"
        ),
        action_head=SlotMixtureHead(
            action_dim=env.action_dim, layout=layout, num_slots=env.num_slots, slot_dim=env.slot_dim
        ),
        hidden_state_dim=int(NETWORK["hidden_state_dim"]),
    )


def load_actor_params(path: str = DEFAULT_PARAMS):
    """Actor params from a ``params_<step>.msgpack`` snapshot (``{params: {actor_params}}``)."""
    with open(path, "rb") as f:
        tree = serialization.msgpack_restore(f.read())
    params = tree["params"]["actor_params"] if "params" in tree else tree["actor_params"]
    return jax.tree_util.tree_map(jnp.asarray, params)


class TrainedPolicy:
    """Greedy recurrent actor as a ``sarsat.evaluate.StatefulPolicy``.

    Mirrors ``mapx.evaluator.make_rec_eval_act_fn`` (``arch.evaluation_greedy: true``): the
    actor runs under a leading ``(time=1, env=1)`` axis pair over its own GRU carry and the
    distribution's mode is the action. ``done`` is False mid-episode; :meth:`reset` zeroes
    the carry, which is what a ``done`` of True would do inside the network.
    """

    def __init__(self, actor: SlotMixtureActor, params, num_agents: int) -> None:
        self.num_agents = num_agents
        self.hidden_state_dim = int(actor.hidden_state_dim)
        self._params = params

        def apply_and_mode(params, carry, agents_view, action_mask):
            obs = Observation(agents_view[None, None], action_mask[None, None], None)
            done = jnp.zeros((1, 1, num_agents), bool)
            carry, pi = actor.apply(params, carry, (obs, done))
            return carry, pi.mode()[0, 0]

        self._apply = jax.jit(apply_and_mode)
        self.reset()

    def reset(self) -> None:
        self._carry = ScannedRNN.initialize_carry((1, self.num_agents), self.hidden_state_dim)

    def __call__(self, obs) -> jax.Array:
        self._carry, action = self._apply(
            self._params, self._carry, obs.agents_view, obs.action_mask
        )
        return action


def load_policy(path: str = DEFAULT_PARAMS, env: WindowedCoopSarSat | None = None) -> TrainedPolicy:
    env = make_env() if env is None else env
    return TrainedPolicy(build_actor(env), load_actor_params(path), env.num_agents)


def _main() -> None:
    import argparse
    import json
    import time

    from sarsat.evaluate import run_episode

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    env = make_env()
    policy = load_policy(args.params, env)
    t0 = time.time()
    log = run_episode(env, policy, seed=args.seed)
    ret = float(log.reward.sum())
    print(f"seed {args.seed}: return {ret:.4f} in {time.time() - t0:.0f}s ({log.num_steps} steps)")

    ref_path = os.path.join(os.path.dirname(os.path.dirname(args.params)), "eval_seeds.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            rows = {r["seed"]: r for r in json.load(f)["results"]}
        if args.seed in rows:
            ref = rows[args.seed]["policy_return"]
            print(f"reference (A40): {ref:.4f}; difference {ret - ref:+.4f}")


if __name__ == "__main__":
    _main()
