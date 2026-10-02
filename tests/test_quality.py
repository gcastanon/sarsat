"""Image-quality requirements (DECISIONS.md issue 32): each target requires a minimum
quality, drawn once per hotspot cluster and once per background target."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from sarsat import SarSat
from sarsat.modes import as_mode
from sarsat.quality import MODES, QualitySarSat
from test_modes import _action, _imaged, _single_target

KEY = jax.random.PRNGKey(0)
SCENARIO = Path(__file__).parents[1] / "mapx_integration/mapx/configs/env/scenario"
STRIPMAP, SPOTLIGHT = 1, 2  # look-slot values: mode k is look k + 1


def test_default_levels_reproduce_sarsat() -> None:
    plain = SarSat(num_satellites=6, max_targets=80, hotspots=2, hotspot_targets=10)
    quality = QualitySarSat(num_satellites=6, max_targets=80, hotspots=2, hotspot_targets=10)
    s0, t0 = plain.reset(KEY)
    s1, t1 = quality.reset(KEY)
    assert np.all(np.asarray(s1.target_quality) == 0.0)
    for t in range(20):
        a = jax.random.uniform(jax.random.fold_in(KEY, t), (6, 4), minval=-1.0)
        a = a.at[:, 2:].set(jnp.round(0.5 * (a[:, 2:] + 1.0)))
        s0, t0 = plain.step(s0, a)
        s1, t1 = quality.step(s1, a)
        np.testing.assert_array_equal(s0.battery, s1.battery)
        np.testing.assert_array_equal(s0.target_active, s1.target_active)
        np.testing.assert_array_equal(t0.reward, t1.reward)
        np.testing.assert_array_equal(t0.observation.agents_view, t1.observation.agents_view)


def test_clusters_share_a_level_and_the_background_follows_its_weights() -> None:
    kwargs = dict(num_satellites=4, max_targets=5000, hotspots=8, hotspot_targets=20)
    env = QualitySarSat(
        **kwargs,
        modes=MODES,
        hotspot_quality=(1.0, 7.0),
        background_quality=(1.0, 3.0, 5.0),
        background_quality_weights=(0.5, 0.5, 0.0),
    )
    state, _ = env.reset(KEY)
    plain, _ = SarSat(**kwargs, modes=MODES).reset(KEY)
    np.testing.assert_array_equal(state.target_pos, plain.target_pos)  # same field
    q = np.asarray(state.target_quality)
    hot = q[:160].reshape(20, 8)  # slot i is in cluster i % hotspots
    assert np.all(hot == hot[0]) and set(hot[0]) <= {1.0, 7.0}
    background = q[160:]
    assert set(background) == {1.0, 3.0}  # a zero weight is never drawn
    assert abs(np.mean(background == 1.0) - 0.5) < 0.05


@pytest.mark.parametrize("required, stripmap, spotlight", [(3.0, True, True), (5.0, False, True)])
def test_a_look_images_a_target_only_in_a_mode_good_enough(required, stripmap, spotlight) -> None:
    env = QualitySarSat(num_satellites=1, max_targets=1, modes=MODES)
    state = _single_target(env, 27.0)  # ~30 degrees incidence: inside both modes' bands
    state = state._replace(target_quality=jnp.full((1,), required, jnp.float32))
    assert _imaged(env, state, _action(env, state, 27.0, STRIPMAP)) == stripmap
    assert _imaged(env, state, _action(env, state, 27.0, SPOTLIGHT)) == spotlight
    # The observation's per-mode flags for the slot say the same.
    view = env._observe(state).agents_view[0]
    flags = np.asarray(view[1 + env._pointing_dim : 1 + env._pointing_dim + env.num_modes])
    assert bool(flags[STRIPMAP - 1]) == stripmap and bool(flags[SPOTLIGHT - 1]) == spotlight


def test_levels_need_weights_to_match() -> None:
    with pytest.raises(ValueError):
        QualitySarSat(num_satellites=2, max_targets=10, background_quality=(1.0, 3.0),
                      background_quality_weights=(1.0,))
    with pytest.raises(ValueError):
        QualitySarSat(num_satellites=2, max_targets=10, hotspot_quality=())


def test_modes_scenario_builds_with_the_documented_modes() -> None:
    config = yaml.safe_load((SCENARIO / "sarsat-100sat-modes.yaml").read_text())["task_config"]
    assert tuple(as_mode(m) for m in config["modes"]) == MODES
    # Cheapest first, so the greedy reference picks the cheapest mode that is good enough.
    assert [m.sense_cost for m in MODES] == sorted(m.sense_cost for m in MODES)
    config = config | dict(num_satellites=4, planes=2, max_targets=2100)  # scaled down
    env = QualitySarSat(**config)
    state, timestep = env.reset(KEY)
    assert env.num_modes == 4 and float(env.action_spec.maximum[-1]) == 4.0
    assert set(np.asarray(state.target_quality)) <= {1.0, 3.0, 5.0, 7.0}
    assert timestep.observation.agents_view.shape == (4, env.obs_dim)
