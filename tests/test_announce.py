"""Announced requests (sarsat.announce): same dynamics, bounded notice, no leakage."""

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.announce import AnnouncedCoopSarSat, AnnouncedSarSat
from sarsat.windows import WindowedCoopSarSat, WindowedSarSat

SMALL = dict(
    num_satellites=20,
    max_targets=600,
    hotspots=8,
    hotspot_targets=25,
    planes=5,
    window_steps=(10, 30),
    background_windows=True,
)


def test_announcements_leave_the_episode_unchanged() -> None:
    windowed, announced = WindowedSarSat(**SMALL), AnnouncedSarSat(**SMALL)
    key = jax.random.PRNGKey(7)
    s0, _ = jax.jit(windowed.reset)(key)
    s1, _ = jax.jit(announced.reset)(key)
    leaves = jax.tree.leaves(s0)
    for a, b in zip(leaves, jax.tree.leaves(s1)[: len(leaves)], strict=True):
        assert jnp.array_equal(a, b)
    step0, step1 = jax.jit(windowed.step), jax.jit(announced.step)
    for t in range(40):
        action = jax.random.uniform(jax.random.PRNGKey(t), (windowed.num_agents, 4), minval=-1)
        action = action.at[:, 2:].set(action[:, 2:] > 0)
        s0, t0 = step0(s0, action)
        s1, t1 = step1(s1, action)
        assert jnp.array_equal(t0.reward, t1.reward)
        assert jnp.array_equal(t0.observation.agents_view, t1.observation.agents_view)


def test_announcement_notice_and_sharing() -> None:
    env = AnnouncedSarSat(**SMALL, time_limit=180)
    assert env.announce_lead == 30
    announce, window = [], []
    for seed in range(20):
        state, _ = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        announce.append(np.asarray(state.target_announce))
        window.append(np.asarray(state.target_window))
    announce, window = np.stack(announce), np.stack(window)
    first = window[..., 0]
    latest = np.maximum(first - 30, 0)
    assert (announce >= 0).all() and (announce <= latest).all()
    assert (announce[first <= 30] == 0).all()
    # Uniform on [0, latest]: mean announce / latest near one half.
    far = latest >= 30
    ratio = announce[far] / latest[far]
    assert 0.45 < ratio.mean() < 0.55 and ratio.min() < 0.05 and ratio.max() > 0.95
    # A cluster is one request: its targets share the announcement.
    num_hot = env.hotspots * env.hotspot_targets
    cluster = np.arange(num_hot) % env.hotspots
    for c in range(env.hotspots):
        assert (announce[:, :num_hot][:, cluster == c] == announce[:, [c]]).all()
    # An unannounced target is never visible.
    state, _ = jax.jit(env.reset)(jax.random.PRNGKey(0))
    for step in range(1, 181, 7):
        visible = np.asarray(env._target_geometry(state, jnp.int32(step))[1])
        assert not visible[:, np.asarray(state.target_announce) >= step].any()


def test_everything_announced_matches_windowed_observation() -> None:
    windowed, env = WindowedCoopSarSat(**SMALL), AnnouncedCoopSarSat(**SMALL)
    key = jax.random.PRNGKey(2)
    s0 = windowed.refresh(windowed.reset_state(key))
    s1 = env.reset_state(key)
    s1 = env.refresh(s1._replace(target_announce=jnp.zeros_like(s1.target_announce)))
    o0, o1 = windowed.observe(s0), env.observe(s1)
    np.testing.assert_allclose(o0.agents_view, o1.agents_view, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(o0.global_state, o1.global_state, rtol=1e-6, atol=1e-7)


def test_unannounced_requests_do_not_reach_the_observation() -> None:
    env = AnnouncedCoopSarSat(**SMALL)
    observe = jax.jit(env.observe)
    state = env.reset_state(jax.random.PRNGKey(4))
    state = env.refresh(state._replace(step=jnp.int32(20)))
    hidden = np.asarray(state.target_announce) > 20
    hidden_c = hidden[: env.num_clusters]
    assert hidden.any() and hidden_c.any() and not hidden_c.all()
    base = observe(state)
    # Rewrite everything about the hidden requests: windows, access, counts, status.
    rng = np.random.default_rng(0)
    access = np.asarray(state.cluster_access).copy()
    access[:, :, hidden_c] = rng.random(access[:, :, hidden_c].shape) < 0.5
    team = np.asarray(state.team_future).copy()
    team[:, hidden_c] = rng.integers(0, 50, team[:, hidden_c].shape)
    window = np.asarray(state.target_window).copy()
    window[hidden] = [60, 90]
    active = np.asarray(state.target_active).copy()
    active[hidden] = False
    changed = state._replace(
        cluster_access=jnp.asarray(access),
        team_future=jnp.asarray(team, jnp.float32),
        target_window=jnp.asarray(window),
        target_active=jnp.asarray(active),
    )
    other = observe(changed)
    assert jnp.array_equal(base.agents_view, other.agents_view)
    assert jnp.array_equal(base.global_state, other.global_state)
