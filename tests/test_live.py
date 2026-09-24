"""``sarsat.live.LiveWorld``: rolling episodes over the episodic environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sarsat.live import LiveWorld, rebase_orbit
from sarsat.orbits import satellite_frames
from sarsat.targets import TargetSampler
from sarsat.windows import WindowedCoopSarSat, WindowedState

SEED = 3
T = 20


def _env(**overrides) -> WindowedCoopSarSat:
    kwargs = dict(
        num_satellites=20,
        max_targets=1000,
        hotspots=10,
        hotspot_targets=50,
        hotspot_weight=10.0,
        planes=2,
        window_steps=(5, 10),
        background_windows=False,
        recharge_rate=0.005,
        time_limit=T,
        num_slots=4,
        num_candidates=64,
        horizon=16,
        ranked_contention=True,
    )
    return WindowedCoopSarSat(**(kwargs | overrides))


def _actions(env, key, sense=True):
    n = env.action_spec.shape[0]
    k_point, k_side = jax.random.split(key)
    pointing = jax.random.uniform(k_point, (n, 2), minval=-1.0, maxval=1.0)
    side = jax.random.bernoulli(k_side, 0.5, (n, 1)).astype(jnp.float32)
    return jnp.concatenate([pointing, side, jnp.full((n, 1), float(sense))], axis=1)


def _assert_state_close(a, b, skip=()):
    for name, x, y in zip(a._fields, a, b, strict=True):
        if name in skip:
            continue
        for lx, ly in zip(jax.tree.leaves(x), jax.tree.leaves(y), strict=True):
            if jnp.issubdtype(lx.dtype, jnp.floating):
                np.testing.assert_allclose(lx, ly, rtol=1e-4, atol=1e-2, err_msg=name)
            else:
                np.testing.assert_array_equal(lx, ly, err_msg=name)


def test_episode_zero_is_the_plain_environment() -> None:
    """With no requests, episode 0 of the world is ``env.reset`` + ``env.step``."""
    env = _env()
    world = LiveWorld(env, seed=SEED)
    state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(SEED))
    step = jax.jit(env.step)
    _assert_state_close(world.state, state)
    for i in range(T):
        action = _actions(env, jax.random.PRNGKey(100 + i))
        state, timestep = step(state, action)
        result = world.step(action)
        assert result.reward == pytest.approx(float(timestep.reward[0]), abs=1e-7)
        assert result.step == i + 1 and result.new_episode == (i + 1 == T)
        if i + 1 < T:
            _assert_state_close(world.state, state)
            np.testing.assert_allclose(
                world.obs.agents_view, timestep.observation.agents_view, rtol=1e-4, atol=1e-4
            )
    assert world.episode == 1 and int(world.state.step) == 0
    assert len(world.episode_returns) == 1
    # Every event closed inside the episode was recorded once.
    assert len(world.events) == env.hotspots


def test_rebased_orbits_continue_the_old_ones() -> None:
    env = _env()
    orbit, _ = jax.jit(env.reset)(jax.random.PRNGKey(SEED))
    orbit = orbit.orbit
    shift = 180 * env.step_seconds  # a full-size episode, not just this test's T
    rebased = rebase_orbit(orbit, shift)
    for k in (0, 1, 17, 196):
        pos_new, frame_new = satellite_frames(rebased, k * env.step_seconds)
        pos_old, frame_old = satellite_frames(orbit, shift + k * env.step_seconds)
        np.testing.assert_allclose(pos_new, pos_old, atol=5e-2)  # km
        np.testing.assert_allclose(frame_new, frame_old, atol=1e-5)


def test_rollover_carries_the_world_over() -> None:
    env = _env()
    world = LiveWorld(env, seed=SEED)
    before = None
    for i in range(T):
        action = _actions(env, jax.random.PRNGKey(200 + i))
        if i + 1 == T:
            before = world.state
            pos_end, _ = satellite_frames(before.orbit, T * env.step_seconds)
        world.step(action)
    after = world.state
    assert int(after.step) == 0 and world.episode == 1
    # Orbits continue, batteries carry over.
    pos_start, _ = satellite_frames(after.orbit, 0.0)
    np.testing.assert_allclose(pos_start, pos_end, atol=5e-2)
    # The battery after the last transition of episode 0 is the battery at step 0 of 1.
    last = jax.jit(env.transition)(before, action)[0]
    np.testing.assert_allclose(after.battery, last.battery)
    # Background targets: the survivors keep their place, the captured ones respawn.
    n_hot = env.hotspots * env.hotspot_targets
    survived = np.asarray(last.target_active)[n_hot:]
    assert survived.any() and not survived.all()
    old, new = np.asarray(before.target_latlon)[n_hot:], np.asarray(after.target_latlon)[n_hot:]
    np.testing.assert_array_equal(old[survived], new[survived])
    assert not np.any(np.all(old[~survived] == new[~survived], axis=1))
    assert np.asarray(after.target_active).all()
    # Fresh events with fresh windows inside the episode, and a consistent schedule.
    windows = np.asarray(after.target_window[: env.hotspots])
    assert (windows[:, 0] >= 1).all() and (windows[:, 1] <= T).all()
    assert not np.array_equal(windows, np.asarray(before.target_window[: env.hotspots]))
    base = WindowedState(*after[: len(WindowedState._fields)])
    rebuilt = env.assemble(base, env.cluster_schedule(base, jnp.arange(env.schedule_length)))
    np.testing.assert_array_equal(rebuilt.cluster_access, after.cluster_access)
    np.testing.assert_array_equal(rebuilt.team_future, after.team_future)
    # The geometry cache was refreshed for step 1.
    fresh = env.refresh(after)
    np.testing.assert_array_equal(fresh.visible, after.visible)


def _feasible_point(world, rng, horizon):
    sampler = TargetSampler()
    for i in range(200):
        latlon = np.asarray(sampler(jax.random.PRNGKey(i), jnp.ones((1,), bool)))[0]
        feas = world.feasibility(*latlon, horizon)
        if feas["count"]:
            return latlon, feas
    raise AssertionError("no feasible point found")


def test_request_injection_is_visible_to_the_lookahead() -> None:
    env = _env()
    world = LiveWorld(env, seed=SEED)
    for i in range(2):
        world.step(_actions(env, jax.random.PRNGKey(300 + i), sense=False))
    t = int(world.state.step)
    horizon = 12
    latlon, feas = _feasible_point(world, None, horizon)
    request = world.inject_request(*latlon, "high", horizon, text="test")
    c = request.cluster
    state = world.state
    slots = c + env.hotspots * np.arange(env.hotspot_targets)
    np.testing.assert_allclose(np.asarray(state.target_latlon)[slots[0]], latlon, atol=1e-5)
    assert np.asarray(state.target_active)[slots].all()
    np.testing.assert_array_equal(
        np.asarray(state.target_window)[slots], [[t + 1, t + horizon]] * 50
    )
    unit = float(jnp.max(state.target_priority))
    np.testing.assert_allclose(np.asarray(state.target_priority)[slots], unit)
    # The cluster's schedule column follows the geometry, gated by its window.
    access = np.asarray(state.cluster_access)[:, :, c]
    assert not access[: t + 1].any() and not access[t + horizon + 1 :].any()
    sat, first = feas["satellites"][0]
    assert access[first - world.global_step + t, sat]
    # ... and the look-ahead value for that satellite is positive at that offset.
    ahead = np.asarray(world.obs.agents_view)[:, env.slot_dim * env.num_slots :]
    k = first - world.global_step - 1  # steps after t + 1
    assert ahead[sat, 2 * k] > 0.0
    # team_future is exactly what a full rebuild would give.
    base = WindowedState(*state[: len(WindowedState._fields)])
    rebuilt = env.assemble(base, state.cluster_access)
    np.testing.assert_array_equal(rebuilt.team_future, state.team_future)


def test_feasibility_matches_visibility() -> None:
    env = _env()
    world = LiveWorld(env, seed=SEED)
    horizon = 10
    latlon, feas = _feasible_point(world, None, horizon)
    request = world.inject_request(*latlon, "normal", horizon)
    c = request.cluster
    first_seen = {}
    for k in range(horizon):
        # visible is cached for step + 1 before the transition
        vis = np.asarray(world.state.visible)[:, c]
        for n in np.flatnonzero(vis):
            first_seen.setdefault(int(n), world.global_step + 1)
        world.step(_actions(env, jax.random.PRNGKey(400 + k), sense=False))
    expected = {n: g for n, g in feas["satellites"]}
    assert first_seen == expected


def test_request_spills_into_the_next_episode() -> None:
    env = _env()
    world = LiveWorld(env, seed=SEED)
    for i in range(T - 5):
        world.step(_actions(env, jax.random.PRNGKey(500 + i), sense=False))
    request = world.inject_request(10.0, 20.0, "normal", 30)
    c = request.cluster
    assert request.close_global == T - 5 + 30
    window = np.asarray(world.state.target_window)[c]
    np.testing.assert_array_equal(window, [T - 4, T])
    assert not np.asarray(world.state.cluster_access)[T + 1 :, :, c].any()
    for i in range(5):
        result = world.step(_actions(env, jax.random.PRNGKey(600 + i), sense=False))
    assert result.new_episode and request.status == "pending"
    assert [r.id for r in result.request_updates] == [request.id]
    window = np.asarray(world.state.target_window)[request.cluster]
    np.testing.assert_array_equal(window, [1, min(25, T)])
    assert world.cluster_windows()[request.cluster].tolist() == [1, min(25, T)]
    # It is missed once its deadline passes without a capture.
    while request.status == "pending":
        world.step(_actions(env, jax.random.PRNGKey(700 + world.global_step), sense=False))
    assert request.status == "missed"
    assert world.global_step == request.close_global
