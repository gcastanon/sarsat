# Copyright 2026 MAPX contributors.
# SPDX-License-Identifier: Apache-2.0

"""Adapter from the SarSat satellite environment to MAPX's ``MarlEnv`` protocol.

SarSat already speaks the Jumanji API with MAPX's observation field names, so this
wrapper only (1) re-types observations as ``mapx.types`` tuples, optionally adding a
global state, (2) declares the action as hybrid -- two continuous pointing slots
(incidence, squint) followed by binary switches for the look side (when the environment
has ``side_switch`` on) and for sensing -- so MAPX selects ``HybridActionHead``, and
(3) adds the ``env_metrics`` extras entry the learners expect.

``make_sarsat_envs`` is a MAPX ``EnvFactory``; see ``mapx_integration/README.md`` for the
two ways of hooking it into ``mapx.utils.make_env``.
"""

from functools import cached_property
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
from mapx.types import (
    ActionBlock,
    ActionLayout,
    HybridArray,
    MarlEnv,
    Observation,
    ObservationGlobalState,
)
from omegaconf import DictConfig, OmegaConf

from sarsat import SarSat, State


class SarSatWrapper(Wrapper):
    """Expose a ``sarsat.SarSat`` environment to the MAPX learners."""

    def __init__(self, env: SarSat, has_global_state: bool = False) -> None:
        self.has_global_state = has_global_state  # read by the specs built in super().__init__
        super().__init__(env)
        self._env: SarSat
        self.num_agents = env.num_agents
        self.time_limit = env.time_limit
        self.action_dim = env.action_dim

    def _convert(self, timestep: TimeStep) -> TimeStep:
        obs = timestep.observation
        if self.has_global_state:
            # Joint observation, as in the CAMAR wrapper. Its width grows with the number
            # of satellites: use the independent-critic systems for large constellations.
            joint = obs.agents_view.reshape(-1)
            global_state = jnp.broadcast_to(joint, (self.num_agents, joint.shape[0]))
            obs = ObservationGlobalState(
                obs.agents_view, obs.action_mask, global_state, obs.step_count
            )
        else:
            obs = Observation(obs.agents_view, obs.action_mask, obs.step_count)
        return timestep.replace(observation=obs, extras={"env_metrics": {}})

    def reset(self, key: jax.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        return state, self._convert(timestep)

    def step(self, state: State, action: jax.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        return state, self._convert(timestep)

    @cached_property
    def observation_spec(self) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        env_spec = self._env.observation_spec
        fields = {
            "agents_view": env_spec.agents_view,
            "action_mask": env_spec.action_mask,
            "step_count": env_spec.step_count,
        }
        if not self.has_global_state:
            return specs.Spec(Observation, "ObservationSpec", **fields)
        fields["global_state"] = specs.Array(
            (self.num_agents, self.num_agents * self._env.obs_dim), jnp.float32, "global_state"
        )
        return specs.Spec(ObservationGlobalState, "ObservationSpec", **fields)

    @cached_property
    def action_spec(self) -> HybridArray:
        env_spec = self._env.action_spec
        # Two continuous pointing slots, then one binary categorical block per switch:
        # ``[incidence, squint, side, sense]`` or ``[cross-track, squint, sense]``.
        switches = self._env.action_dim - 2
        blocks = (ActionBlock(size=2), *([ActionBlock(size=1, num_categories=2)] * switches))
        return HybridArray(
            shape=env_spec.shape,
            dtype=env_spec.dtype,
            minimum=env_spec.minimum,
            maximum=env_spec.maximum,
            layout=ActionLayout(blocks),
            name="action",
        )


def make_sarsat_envs(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """MAPX ``EnvFactory``: build raw train / eval environments from a Hydra config."""
    kwargs = OmegaConf.to_container(config.env.get("kwargs", {}), resolve=True)
    kwargs |= OmegaConf.to_container(config.env.scenario.task_config, resolve=True)
    return (
        SarSatWrapper(SarSat(**kwargs), add_global_state),
        SarSatWrapper(SarSat(**kwargs), add_global_state),
    )
