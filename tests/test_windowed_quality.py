"""Time-windowed events with quality requirements (DECISIONS.md issue 35)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from sarsat.modes import as_mode
from sarsat.quality import MODES, WindowedQualityCoopSarSat, WindowedQualitySarSat
from sarsat.windows import WindowedCoopSarSat, WindowedSarSat

KEY = jax.random.PRNGKey(0)
SCENARIO = Path(__file__).parents[1] / "mapx_integration/mapx/configs/env/scenario"
FIELD = dict(
    num_satellites=10,
    max_targets=60,
    hotspots=2,
    hotspot_targets=20,
    hotspot_radius_km=30.0,
    planes=5,
    time_limit=40,
    window_steps=(5, 15),
    background_windows=False,
)
QUALITY = dict(hotspot_quality=(1.0, 3.0, 5.0, 7.0), background_quality=(1.0, 3.0, 5.0))


def test_default_levels_reproduce_the_windowed_env() -> None:
    plain, quality = WindowedSarSat(**FIELD), WindowedQualitySarSat(**FIELD)
    s0, t0 = plain.reset(KEY)
    s1, t1 = quality.reset(KEY)
    np.testing.assert_array_equal(s0.target_window, s1.target_window)
    np.testing.assert_array_equal(s0.target_pos, s1.target_pos)
    for t in range(20):
        a = jax.random.uniform(jax.random.fold_in(KEY, t), (10, 4), minval=-1.0)
        a = a.at[:, 2:].set(jnp.round(0.5 * (a[:, 2:] + 1.0)))
        s0, t0 = plain.step(s0, a)
        s1, t1 = quality.step(s1, a)
        np.testing.assert_array_equal(t0.reward, t1.reward)
        np.testing.assert_array_equal(t0.observation.agents_view, t1.observation.agents_view)


def test_coop_slots_keep_the_window_feature_and_add_the_modes() -> None:
    env = WindowedQualityCoopSarSat(**FIELD, **QUALITY, modes=MODES)
    assert env.slot_dim == WindowedCoopSarSat.slot_dim + len(MODES)
    state, timestep = env.reset(KEY)
    assert np.asarray(state.target_quality).shape == (env.max_targets,)
    transition = jax.jit(env.transition)
    looked, rewarded = 0, 0.0
    for _ in range(env.time_limit):
        view = np.asarray(timestep.observation.agents_view)
        slots = view[:, : env.slot_dim * env.num_slots].reshape(env.num_agents, env.num_slots, -1)
        assert np.all(np.isfinite(view))
        np.testing.assert_array_equal(slots[..., -env.num_modes :].sum(-1), slots[..., 0])
        mode = np.argmax(slots[:, 0, -env.num_modes :], axis=-1) + 1
        look = np.where(slots[:, 0, 0] > 0.5, mode, 0)
        action = jnp.asarray(np.concatenate([slots[:, 0, 1:4], look[:, None]], -1), jnp.float32)
        _, reward, _ = transition(state, action)
        looked += int((look > 0).sum())
        rewarded += float(reward)
        state, timestep = env.step(state, action)
    assert looked > 0 and rewarded > 0.0


def test_a_target_outside_its_window_is_not_imaged_whatever_the_mode() -> None:
    env = WindowedQualitySarSat(**FIELD, **QUALITY, modes=MODES)
    state, _ = env.reset(KEY)
    closed = state._replace(target_window=jnp.zeros_like(state.target_window))  # never open
    action = jnp.zeros((env.num_agents, env.action_dim)).at[:, -1].set(1.0)
    assert not bool(env._apply_action(closed, closed.step + 1, action)[2].any())


def test_200_and_500_scenarios_build_with_the_documented_modes() -> None:
    for name in ("sarsat-200sat-modes", "sarsat-500sat-modes"):
        config = yaml.safe_load((SCENARIO / f"{name}.yaml").read_text())["task_config"]
        assert tuple(as_mode(m) for m in config["modes"]) == MODES
        small = config | dict(num_satellites=10, planes=5, max_targets=1000, hotspots=4)
        env = WindowedQualityCoopSarSat(**small, time_limit=40)
        state, timestep = env.reset(KEY)
        assert timestep.observation.agents_view.shape == (10, env.obs_dim)
        assert set(np.asarray(state.target_quality)) <= {1.0, 3.0, 5.0, 7.0}
