"""``sarsat.forecast``: the rest of an episode under a policy, and the diff of two forecasts."""

import jax
import numpy as np
import pytest
from test_live import SEED, T, _actions, _env

from sarsat.evaluate import run_episode
from sarsat.forecast import Forecast, diff, forecast, preserved


def _reset(env):
    state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(SEED))
    return state, timestep.observation


def test_forecast_from_reset_is_the_episode() -> None:
    """From the reset state a forecast records exactly what ``run_episode`` records."""
    env = _env()
    state, obs = _reset(env)
    action = _actions(env, jax.random.PRNGKey(1), sense=True)
    policy = lambda obs: action  # noqa: E731
    fc = forecast(env, state, obs, policy)
    log = run_episode(env, policy, seed=SEED)
    assert fc.start_step == 0 and fc.steps.tolist() == list(range(1, T + 1))
    np.testing.assert_array_equal(fc.captures, log.captures)
    np.testing.assert_array_equal(fc.sensing, log.sensing)
    np.testing.assert_allclose(fc.reward, log.reward, rtol=1e-5)
    np.testing.assert_allclose(fc.battery, log.battery[1:], rtol=1e-5)
    np.testing.assert_allclose(fc.battery_end, log.battery[-1], rtol=1e-5)
    assert len(fc.captures) > 0
    first = fc.first_capture()
    assert set(first) == set(fc.collected.tolist())
    for step, slot, _sat in fc.captures:
        assert first[int(slot)][0] <= step


def test_forecast_stops_at_the_horizon_or_the_time_limit() -> None:
    env = _env()
    state, obs = _reset(env)
    action = _actions(env, jax.random.PRNGKey(2), sense=False)
    short = forecast(env, state, obs, lambda obs: action, horizon=3)
    assert short.steps.tolist() == [1, 2, 3] and short.captures.shape == (0, 3)
    assert short.sensing.shape == (3, env.num_agents) and (short.mode == -1).all()
    full = forecast(env, state, obs, lambda obs: action, horizon=10_000)
    assert full.end_step == T and full.horizon == T


class _Counter:
    """A stateful policy whose state is how often it has been called."""

    def __init__(self, action):
        self.action, self.calls = action, 0

    def reset(self):
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        return self.action

    def snapshot(self):
        return self.calls

    def restore(self, calls):
        self.calls = calls


class _NoSnapshot:
    def reset(self):
        pass

    def __call__(self, obs):
        raise AssertionError("must not be called")


def test_forecast_leaves_a_stateful_policy_as_it_was() -> None:
    env = _env()
    state, obs = _reset(env)
    policy = _Counter(_actions(env, jax.random.PRNGKey(3), sense=True))
    policy.calls = 7
    fc = forecast(env, state, obs, policy, horizon=5)
    assert fc.horizon == 5 and policy.calls == 7
    with pytest.raises(TypeError):
        forecast(env, state, obs, _NoSnapshot(), horizon=1)
    with preserved(lambda obs: obs):  # stateless: nothing to do
        pass


def test_forecast_runs_a_reference_policy_from_the_current_state() -> None:
    env = _env()
    state, obs = _reset(env)
    fc = forecast(env, state, obs, "greedy_beam", horizon=4)
    assert fc.steps.tolist() == [1, 2, 3, 4] and fc.captures.shape[1] == 3
    assert fc.sensing.any()  # greedy_beam looks whenever a target is in reach


def _fc(first):
    """A forecast with the given slot -> (step, sat, mode) captures."""
    caps = np.array([[s, slot, sat] for slot, (s, sat, _) in first.items()], int).reshape(-1, 3)
    modes = np.array([m for (_, _, m) in first.values()], int)
    return Forecast(
        start_step=0,
        steps=np.arange(1, 4),
        sensing=np.zeros((3, 2), bool),
        mode=-np.ones((3, 2), int),
        captures=caps,
        capture_mode=modes,
        reward=np.zeros(3),
        battery=np.ones((3, 2)),
    )


def test_diff_sorts_slots_into_lost_gained_shifted_and_same() -> None:
    base = _fc({1: (1, 0, 0), 2: (2, 1, 0), 3: (3, 0, 0), 4: (3, 1, 1)})
    alt = _fc({1: (1, 0, 0), 2: (3, 1, 0), 4: (3, 0, 1), 5: (2, 1, 0)})
    d = diff(base, alt)
    assert d.lost == {3: (3, 0, 0)}
    assert d.gained == {5: (2, 1, 0)}
    assert d.shifted == {2: ((2, 1, 0), (3, 1, 0)), 4: ((3, 1, 1), (3, 0, 1))}
    assert d.same == {1: (1, 0, 0)}
    assert not d.same_set
    assert diff(base, base).same_set and diff(base, base).shifted == {}
