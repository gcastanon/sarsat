"""Collection modes (DECISIONS.md issue 31): one mode per look, each with its own battery
cost, image quality, duration, incidence band, squint limit and beam footprint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sarsat import SarSat, SensorMode, State
from sarsat.evaluate import greedy_policy_for, run_episode, write_csv
from sarsat.orbits import EARTH_RADIUS_KM, satellite_frames

KEY = jax.random.PRNGKey(0)

# At 500-600 km a look angle of 22 degrees is ~24 degrees of incidence, 36 is ~39.5.
NEAR = SensorMode(name="near", sense_cost=0.1, quality=0.3, incidence_deg=(20.0, 32.0))
FAR = SensorMode(name="far", sense_cost=0.3, quality=0.7, incidence_deg=(32.0, 45.0))


def _ground_point(env: SarSat, state: State, look_deg: float, squint_deg: float) -> jax.Array:
    """Where satellite 0's beam meets the Earth when pointed at ``(look, squint)`` at step 1."""
    position, frame = satellite_frames(state.orbit, env.step_seconds)
    look, squint = np.radians([look_deg, squint_deg])
    body = np.array([np.sin(squint), np.cos(squint) * np.sin(look), np.cos(squint) * np.cos(look)])
    origin = np.asarray(position[0], np.float64)
    ray = body @ np.asarray(frame[0], np.float64)
    b, c = ray @ origin, origin @ origin - EARTH_RADIUS_KM**2
    return jnp.asarray(origin + (-b - np.sqrt(b**2 - c)) * ray, jnp.float32)


def _single_target(env: SarSat, look_deg: float, squint_deg: float = 0.0) -> State:
    state, _ = env.reset(KEY, num_targets=1)
    point = _ground_point(env, state, look_deg, squint_deg)
    return state._replace(target_pos=state.target_pos.at[0].set(point))


def _action(env: SarSat, state: State, look_deg: float, look: int, squint_deg=0.0) -> jax.Array:
    """Every satellite points at ``(look_deg, squint_deg)``; ``look`` is the look slot."""
    n = env.num_agents
    angle = jnp.full((n, 1), np.radians(look_deg), jnp.float32)
    squint = jnp.full((n, 1), np.radians(squint_deg), jnp.float32)
    pointing = env._encode_pointing(state.orbit, angle, squint)[:, 0]
    return jnp.concatenate([pointing, jnp.full((n, 1), float(look), jnp.float32)], axis=-1)


def _imaged(env: SarSat, state: State, action: jax.Array) -> bool:
    return bool(env._apply_action(state, state.step + 1, action)[2][0, 0])


def test_default_is_one_mode_and_the_look_slot_is_the_sense_flag() -> None:
    env = SarSat(num_satellites=2, max_targets=10)
    assert env.num_modes == 1 and env.modes == (SensorMode(),)
    assert env.action_dim == 4 and float(env.action_spec.maximum[-1]) == 1.0
    moded = SarSat(num_satellites=2, max_targets=10, modes=[NEAR, FAR])
    assert moded.action_dim == 4  # one look slot whatever the number of modes
    np.testing.assert_array_equal(moded.action_spec.maximum, [1, 1, 1, 2])
    noside = SarSat(num_satellites=2, max_targets=10, modes=[NEAR, FAR], side_switch=False)
    np.testing.assert_array_equal(noside.action_spec.maximum, [1, 1, 2])


def test_explicit_default_mode_has_the_default_dynamics() -> None:
    plain = SarSat(num_satellites=6, max_targets=80)
    moded = SarSat(num_satellites=6, max_targets=80, modes=[SensorMode()])
    s0, t0 = plain.reset(KEY)
    s1, t1 = moded.reset(KEY)
    for t in range(30):
        a = jax.random.uniform(jax.random.fold_in(KEY, t), (6, 4), minval=-1.0)
        a = a.at[:, 2:].set(jnp.round(0.5 * (a[:, 2:] + 1.0)))
        a = a.at[:, :3].set(jnp.where(t % 2, a[:, :3], t0.observation.agents_view[:, 1:4]))
        s0, t0 = plain.step(s0, a)
        s1, t1 = moded.step(s1, a)
        np.testing.assert_array_equal(s0.battery, s1.battery)
        np.testing.assert_array_equal(s0.target_active, s1.target_active)
        np.testing.assert_array_equal(t0.reward, t1.reward)


def test_each_mode_has_its_own_incidence_band() -> None:
    env = SarSat(num_satellites=1, max_targets=1, modes=[NEAR, FAR])
    near, far = _single_target(env, 22.0), _single_target(env, 36.0)
    assert _imaged(env, near, _action(env, near, 22.0, look=1))
    assert not _imaged(env, near, _action(env, near, 22.0, look=2))
    assert _imaged(env, far, _action(env, far, 36.0, look=2))
    assert not _imaged(env, far, _action(env, far, 36.0, look=1))
    assert not _imaged(env, far, _action(env, far, 36.0, look=0))  # no look, no image


def test_each_mode_has_its_own_squint_limit() -> None:
    wide = SensorMode(name="wide", max_squint_deg=25.0)
    narrow = SensorMode(name="narrow", max_squint_deg=10.0)
    env = SarSat(num_satellites=1, max_targets=1, modes=[wide, narrow])
    state = _single_target(env, 30.0, squint_deg=18.0)
    assert _imaged(env, state, _action(env, state, 30.0, look=1, squint_deg=18.0))
    assert not _imaged(env, state, _action(env, state, 30.0, look=2, squint_deg=18.0))


def test_each_mode_has_its_own_beam_footprint() -> None:
    spot = SensorMode(name="spot", beam_half_width_deg=(1.0, 1.0))
    strip = SensorMode(name="strip", beam_half_width_deg=(6.0, 6.0))
    env = SarSat(num_satellites=1, max_targets=1, modes=[spot, strip])
    state = _single_target(env, 30.0)
    # Aim 3 degrees off the target: inside the strip footprint, outside the spot one.
    assert not _imaged(env, state, _action(env, state, 33.0, look=1))
    assert _imaged(env, state, _action(env, state, 33.0, look=2))
    assert _imaged(env, state, _action(env, state, 30.0, look=1))


def test_each_mode_draws_its_own_battery_cost() -> None:
    env = SarSat(num_satellites=1, max_targets=1, modes=[NEAR, FAR], recharge_rate=0.02)
    state, _ = env.reset(KEY, num_targets=1)
    for look, cost in ((1, 0.1), (2, 0.3), (0, 0.0)):
        state = state._replace(battery=jnp.full((1,), 0.5))
        after, _ = env.step(state, _action(env, state, 30.0, look=look))
        assert float(after.battery[0]) == pytest.approx(0.5 + 0.02 - cost, abs=1e-6)
    # Charge for the cheap mode only: the mask stays open, the dear look is ignored.
    state = state._replace(battery=jnp.full((1,), 0.2))
    assert bool(env._observe(state).action_mask[0, env._pointing_dim])
    after, _ = env.step(state, _action(env, state, 30.0, look=2))
    assert float(after.battery[0]) == pytest.approx(0.22, abs=1e-6)
    state = state._replace(battery=jnp.full((1,), 0.05))
    assert not bool(env._observe(state).action_mask[0, env._pointing_dim])


class _Demanding(SarSat):
    """Every target requires quality 0.5."""

    def _required_quality(self, state):
        return jnp.full((self.max_targets,), 0.5)


def test_a_look_only_images_targets_whose_quality_the_mode_meets() -> None:
    wide = (20.0, 45.0)
    low = SensorMode(name="low", quality=0.3, incidence_deg=wide)
    high = SensorMode(name="high", quality=0.7, incidence_deg=wide)
    env = _Demanding(num_satellites=1, max_targets=1, modes=[low, high])
    state = _single_target(env, 30.0)
    assert not _imaged(env, state, _action(env, state, 30.0, look=1))
    assert _imaged(env, state, _action(env, state, 30.0, look=2))
    flags = env._observe(state).agents_view[0, 4:6]  # slot 0, first step: (low, high)
    np.testing.assert_array_equal(flags, [0.0, 1.0])
    # Without requirements both modes image it.
    plain = SarSat(num_satellites=1, max_targets=1, modes=[low, high])
    assert _imaged(plain, state, _action(plain, state, 30.0, look=1))


@pytest.mark.parametrize("look_deg, flags", [(22.0, [1.0, 0.0]), (36.0, [0.0, 1.0])])
def test_observation_flags_say_which_modes_image_the_slot(look_deg, flags) -> None:
    env = SarSat(num_satellites=1, max_targets=1, modes=[NEAR, FAR])
    state = _single_target(env, look_deg)
    view = env._observe(state).agents_view[0]
    width = 1 + env._pointing_dim + env.num_modes
    assert env.obs_dim == width * env.max_observed_targets * env.lookahead_steps + 1
    assert view[0] == 1.0  # accessible to the envelope
    np.testing.assert_array_equal(view[4:width], flags)
    # Copying the slot's pointing and naming a flagged mode images the target.
    for k, flag in enumerate(flags):
        action = jnp.concatenate([view[None, 1:4], jnp.full((1, 1), k + 1.0)], axis=-1)
        assert _imaged(env, state, action) == bool(flag)


def test_duration_is_recorded_but_does_not_change_the_dynamics() -> None:
    quick = SarSat(num_satellites=4, max_targets=50, modes=[SensorMode(duration_s=1.0)])
    slow = SarSat(num_satellites=4, max_targets=50, modes=[SensorMode(duration_s=45.0)])
    assert slow.modes[0].duration_s == 45.0
    a, b = run_episode(quick, greedy_policy_for(quick)), run_episode(slow, greedy_policy_for(slow))
    np.testing.assert_array_equal(a.reward, b.reward)
    np.testing.assert_array_equal(a.battery, b.battery)


def test_modes_from_config_mappings_and_validation() -> None:
    env = SarSat(num_satellites=1, max_targets=1, modes=[{"name": "m", "incidence_deg": [25, 40]}])
    assert env.modes[0] == SensorMode(name="m", incidence_deg=(25.0, 40.0))
    with pytest.raises(ValueError):
        SarSat(modes=[])
    with pytest.raises(ValueError):
        SarSat(modes=[SensorMode(incidence_deg=(40.0, 30.0))])
    with pytest.raises(ValueError):
        SarSat(modes=[SensorMode(sense_cost=0.01)])  # below the recharge rate
    with pytest.raises(ValueError):  # the envelope's outer edge plus the widest beam
        SarSat(modes=[SensorMode(incidence_deg=(20, 80)), SensorMode(beam_half_width_deg=(12, 4))])
    with pytest.raises(ValueError):
        SensorMode(duration_s=-1.0)


def test_greedy_episode_logs_the_mode_of_every_look(tmp_path) -> None:
    env = SarSat(num_satellites=8, max_targets=150, modes=[NEAR, FAR])
    log = run_episode(env, greedy_policy_for(env), seed=1)
    assert log.mode_names == ("near", "far")
    assert set(np.unique(log.mode)) <= {-1, 0, 1}
    np.testing.assert_array_equal(log.mode >= 0, log.sensing)
    assert (log.mode == 0).any() and (log.mode == 1).any() and log.reward.sum() > 0
    write_csv(log, tmp_path)
    looks = (tmp_path / "looks.csv").read_text()
    assert looks.splitlines()[0].endswith(",mode")
    assert {line.rsplit(",", 1)[1] for line in looks.splitlines()[1:]} <= {"near", "far"}
