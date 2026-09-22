# Copyright 2026 MAPX contributors.
# SPDX-License-Identifier: Apache-2.0

"""MAPX adapter for ``sarsat.coop.CoopSarSat`` (beam-value observation, compact global state).

Differences from the plain ``SarSatWrapper``:

* the global state handed to centralised critics is the environment's fixed-size team
  summary, not the joint observation, so ``rec_mappo`` stays cheap at 100 satellites;
* rewards are multiplied by ``reward_scale`` so returns are O(1-10) for the value loss
  (the true fraction imaged is logged as the ``fraction_imaged`` env metric);
* with ``fused_auto_reset`` the training environment resets *inside* ``step``, before the
  observation is built. Under ``vmap`` MAPX's ``AutoResetWrapper`` evaluates both a full
  ``reset`` and the step's observation every step; fusing builds one observation per step.
  Use :class:`PassthroughAutoReset` in place of ``AutoResetWrapper`` with it. Episodes only
  end on the time limit in practice, so the missing ``real_next_obs`` is never needed
  (``bootstrap_truncation`` must stay off, which is right for a fixed-horizon task whose
  observation includes the time remaining).
"""

from functools import cached_property
from typing import NamedTuple, Tuple, Union

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

from sarsat.coop import CoopSarSat, CoopState
from sarsat.windows import WindowedCoopSarSat

LAST_SLOT_DIM = CoopSarSat.slot_dim  # slot width of the most recently built env
_SLICES = 3  # schedule rows built per step; 3 * time_limit comfortably exceeds the schedule


class CoopWrapperState(NamedTuple):
    key: jax.Array
    env_state: CoopState
    episode_return: jax.Array  # unscaled return of the running episode
    next_key: jax.Array  # reset key of the next episode (fused auto-reset only)
    next_access: jax.Array  # its cluster schedule, filled a few steps at a time


class SarSatCoopWrapper(Wrapper):
    def __init__(
        self,
        env: CoopSarSat,
        has_global_state: bool = False,
        reward_scale: float = 1.0,
        fused_auto_reset: bool = False,
        credit_mix: float = 0.0,
    ) -> None:
        self.credit_mix = credit_mix
        self.has_global_state = has_global_state
        super().__init__(env)
        self._env: CoopSarSat
        self.num_agents = env.num_agents
        self.time_limit = env.time_limit
        self.action_dim = env.action_dim
        self.reward_scale = reward_scale
        self.fused_auto_reset = fused_auto_reset

    def _convert(self, timestep: TimeStep, fraction: jax.Array) -> TimeStep:
        obs = timestep.observation
        if self.has_global_state:
            obs = ObservationGlobalState(
                obs.agents_view, obs.action_mask, obs.global_state, obs.step_count
            )
        else:
            obs = Observation(obs.agents_view, obs.action_mask, obs.step_count)
        return timestep.replace(
            observation=obs,
            reward=timestep.reward * self.reward_scale,
            extras={"env_metrics": {"fraction_imaged": fraction}},
        )

    def reset(self, key: jax.Array) -> Tuple[CoopWrapperState, TimeStep]:
        key, reset_key = jax.random.split(key)
        env_state, timestep = self._env.reset(reset_key)
        zero = jnp.zeros((), jnp.float32)
        key, next_key = jax.random.split(key)
        pending = jnp.zeros_like(env_state.cluster_access[: self._schedule_rows])
        state = CoopWrapperState(key, env_state, zero, next_key, pending)
        return state, self._convert(timestep, zero)

    def step(self, state: CoopWrapperState, action: jax.Array) -> Tuple[CoopWrapperState, TimeStep]:
        env = self._env
        env_state, reward, all_imaged, credit = env.transition_with_credit(state.env_state, action)
        last = all_imaged | (env_state.step >= env.time_limit)
        fraction = state.episode_return + reward
        key, next_key, pending = state.key, state.next_key, state.next_access
        running = fraction
        if self.fused_auto_reset:
            # Under vmap a conditional reset runs every step, so it has to be cheap. Sampling
            # the next episode's orbits and targets is; its cluster schedule is not, so that
            # is built ``_SLICES`` steps at a time over the running episode and is complete
            # long before it is needed.
            base = env.base_state(next_key)
            fresh = env.assemble(base, pending)
            env_state = jax.tree.map(lambda a, b: jnp.where(last, a, b), fresh, env_state)
            running = jnp.where(last, 0.0, fraction)
            # On the hand-over step this writes row 0 for the episode that just started; the
            # first step of the new episode overwrites rows 0.._SLICES-1 for the right one.
            rows = jnp.clip(
                _SLICES * (env_state.step - 1) + jnp.arange(_SLICES), 0, env.schedule_length - 1
            )
            pending = pending.at[rows].set(env.cluster_schedule(base, rows))
            key, candidate = jax.random.split(key)
            next_key = jnp.where(last, candidate, next_key)
        env_state, timestep = env.make_timestep(env_state, reward, all_imaged, last)
        timestep = self._convert(timestep, fraction)
        if self.credit_mix:
            # Blend the team reward with each satellite's own share of it (mean over agents
            # is still the team reward). Off by default: the task's reward is the team's.
            mixed = (1.0 - self.credit_mix) * reward + self.credit_mix * self.num_agents * credit
            timestep = timestep.replace(reward=mixed * self.reward_scale)
        return CoopWrapperState(key, env_state, running, next_key, pending), timestep

    @property
    def _schedule_rows(self) -> int:
        return self._env.schedule_length if self.fused_auto_reset else 0

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
        fields["global_state"] = env_spec.global_state
        return specs.Spec(ObservationGlobalState, "ObservationSpec", **fields)

    @cached_property
    def action_spec(self) -> HybridArray:
        env_spec = self._env.action_spec
        blocks = (ActionBlock(size=2), *([ActionBlock(size=1, num_categories=2)] * 2))
        return HybridArray(
            shape=env_spec.shape,
            dtype=env_spec.dtype,
            minimum=env_spec.minimum,
            maximum=env_spec.maximum,
            layout=ActionLayout(blocks),
            name="action",
        )


class PassthroughAutoReset(Wrapper):
    """Stand-in for MAPX's ``AutoResetWrapper`` when the environment resets itself."""

    OBS_IN_EXTRAS_KEY = "real_next_obs"

    def __init__(self, env: MarlEnv):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.time_limit = env.time_limit
        self.action_dim = env.action_dim

    @staticmethod
    def _tag(state, timestep):
        return state, timestep.replace(
            extras=dict(timestep.extras) | {"real_next_obs": timestep.observation}
        )

    def reset(self, key: jax.Array):
        return self._tag(*self._env.reset(key))

    def step(self, state, action: jax.Array):
        return self._tag(*self._env.step(state, action))


def make_sarsat_coop_envs(
    config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """MAPX ``EnvFactory`` for the cooperative-observation environment."""
    kwargs = OmegaConf.to_container(config.env.get("kwargs", {}), resolve=True)
    kwargs |= OmegaConf.to_container(config.env.scenario.task_config, resolve=True)
    scale = float(config.env.get("reward_scale", 1.0))
    fused = bool(config.env.get("fused_auto_reset", True))
    mix = float(config.env.get("credit_mix", 0.0))
    # Time-windowed scenarios (sarsat.windows) widen each observation slot by one feature;
    # tell the networks, which are instantiated from this same config after the env.
    env_cls = WindowedCoopSarSat if "window_steps" in kwargs else CoopSarSat
    if "network" in config:
        for net in ("actor_network", "critic_network"):
            config.network[net].pre_torso.slot_dim = env_cls.slot_dim
    global LAST_SLOT_DIM
    LAST_SLOT_DIM = env_cls.slot_dim
    return (
        SarSatCoopWrapper(env_cls(**kwargs), add_global_state, scale, fused, credit_mix=mix),
        SarSatCoopWrapper(env_cls(**kwargs), add_global_state, scale, fused_auto_reset=False),
    )
