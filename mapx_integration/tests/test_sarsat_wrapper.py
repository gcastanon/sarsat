# Copyright 2026 MAPX contributors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SarSat -> MAPX wrapper contract (run inside the MAPX repository)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mapx.types import HybridArray, Observation, ObservationGlobalState
from mapx.utils import make_env
from mapx.utils.network_utils import get_action_head
from mapx.wrappers.sarsat import SarSatWrapper, make_sarsat_envs
from omegaconf import OmegaConf

from sarsat import SarSat

CONFIG_DIR = Path(make_env.__file__).parents[1] / "configs" / "env"


def _config(add_agent_id: bool = False) -> OmegaConf:
    env = OmegaConf.load(CONFIG_DIR / "sarsat.yaml")
    env.pop("defaults")
    env.scenario = OmegaConf.load(CONFIG_DIR / "scenario" / "sarsat-8sat-100tg.yaml")
    return OmegaConf.create({"env": env, "system": {"add_agent_id": add_agent_id}})


def test_wrapper_types_shapes_and_hybrid_action_spec() -> None:
    env = SarSatWrapper(SarSat(num_satellites=4, max_targets=20), has_global_state=True)
    state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(0))
    obs = timestep.observation

    assert isinstance(obs, ObservationGlobalState)
    assert obs.agents_view.shape == (4, env.obs_dim)
    assert obs.global_state.shape == (4, 4 * env.obs_dim)
    assert obs.action_mask.shape == (4, 4) and obs.step_count.shape == (4,)
    env.observation_spec.validate(obs)
    assert timestep.extras == {"env_metrics": {}}

    # [incidence, squint, side, sense]: a 2-D Gaussian plus two Bernoullis.
    assert isinstance(env.action_spec, HybridArray)
    assert env.action_spec.layout.continuous_slots == (0, 1)
    assert env.action_spec.layout.categorical_blocks == (((2,), 2), ((3,), 2))
    assert get_action_head(env.action_spec)[1] == "hybrid"

    signed = SarSatWrapper(SarSat(num_satellites=4, max_targets=20, side_switch=False))
    assert signed.action_spec.layout.categorical_blocks == (((2,), 2),)

    state, timestep = jax.jit(env.step)(state, env.action_spec.generate_value())
    assert timestep.reward.shape == (4,) and timestep.discount.shape == (4,)
    assert isinstance(
        SarSatWrapper(SarSat(4, 20)).reset(jax.random.PRNGKey(0))[1].observation, Observation
    )


def test_registered_factory_runs_through_the_common_wrappers() -> None:
    make_env.register_env_factory("SarSat", make_sarsat_envs)
    try:
        train_env, eval_env = make_env.make(_config(), add_global_state=False)
    finally:
        make_env.unregister_env_factory("SarSat")
    assert train_env.num_agents == eval_env.num_agents == 8
    assert train_env.time_limit == 180 and train_env.action_dim == 4

    # Auto-reset + episode metrics, vmapped over environments as the learners do.
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    states, timesteps = jax.jit(jax.vmap(train_env.reset))(keys)
    actions = jnp.zeros((4, 8, 4), dtype=jnp.float32)
    step = jax.jit(jax.vmap(train_env.step))
    for _ in range(181):  # crosses one auto-reset boundary
        states, timesteps = step(states, actions)
    assert timesteps.observation.agents_view.shape == (4, 8, train_env.unwrapped.obs_dim)
    np.testing.assert_array_equal(timesteps.extras["episode_metrics"]["episode_length"], 180)
    assert bool(jnp.all(jnp.isfinite(timesteps.observation.agents_view)))
