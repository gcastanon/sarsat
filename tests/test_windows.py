"""Time-windowed targets (sarsat.windows) leave the plain environments untouched."""

import jax
import jax.numpy as jnp
import numpy as np

from sarsat import SarSat
from sarsat.coop import CoopSarSat
from sarsat.windows import WindowedCoopSarSat, WindowedSarSat

SMALL = dict(num_satellites=20, max_targets=600, hotspots=8, hotspot_targets=25, planes=5)


def test_no_windows_reproduces_sarsat() -> None:
    plain, windowed = SarSat(**SMALL), WindowedSarSat(**SMALL)
    key = jax.random.PRNGKey(3)
    s0, t0 = jax.jit(plain.reset)(key)
    s1, t1 = jax.jit(windowed.reset)(key)
    leaves = jax.tree.leaves(s0)
    for a, b in zip(leaves, jax.tree.leaves(s1)[: len(leaves)], strict=True):
        assert jnp.array_equal(a, b)
    assert jnp.array_equal(t0.observation.agents_view, t1.observation.agents_view)
    assert bool((s1.target_window[:, 0] == 0).all()) and bool(
        (s1.target_window[:, 1] == windowed.time_limit).all()
    )


def test_windows_gate_visibility_and_keep_orbits() -> None:
    plain = SarSat(**SMALL)
    env = WindowedSarSat(**SMALL, window_steps=(10, 30), background_windows=False)
    key = jax.random.PRNGKey(5)
    s0, _ = jax.jit(plain.reset)(key)
    s1, _ = jax.jit(env.reset)(key)
    assert jnp.array_equal(s0.target_pos, s1.target_pos)
    assert jnp.array_equal(s0.orbit.raan, s1.orbit.raan)
    window = np.asarray(s1.target_window)
    num_hot = env.hotspots * env.hotspot_targets
    length = window[:num_hot, 1] - window[:num_hot, 0] + 1
    assert length.min() >= 10 and length.max() <= 30 and window[:num_hot, 0].min() >= 1
    assert window[:num_hot, 1].max() <= env.time_limit
    # A cluster is one event: every target of cluster c carries cluster c's window.
    cluster = np.arange(num_hot) % env.hotspots
    for c in range(env.hotspots):
        assert (window[:num_hot][cluster == c] == window[c]).all()
    # Background stays open all episode.
    assert (window[num_hot:, 0] == 0).all() and (window[num_hot:, 1] == env.time_limit).all()
    # Outside the window a hotspot target is never visible; the plain env may see it.
    for step in (1, 60, 120, 180):
        visible = np.asarray(env._target_geometry(s1, jnp.int32(step))[1])
        closed = (step < window[:, 0]) | (step > window[:, 1])
        assert not visible[:, closed].any()
        base = np.asarray(plain._target_geometry(s0, jnp.int32(step))[1])
        assert (visible <= base).all()


def test_windowed_coop_observation() -> None:
    env = WindowedCoopSarSat(**SMALL, window_steps=(10, 30), background_windows=False)
    assert env.slot_dim == CoopSarSat.slot_dim + 1
    assert env.obs_dim == env.slot_dim * env.num_slots + 2 * env.horizon + 5
    state, ts = jax.jit(env.reset)(jax.random.PRNGKey(1))
    view = ts.observation.agents_view
    assert view.shape == (env.num_agents, env.obs_dim)
    assert bool(jnp.isfinite(view).all()) and bool(jnp.isfinite(ts.observation.global_state).all())
    left = view[:, : env.slot_dim * env.num_slots].reshape(env.num_agents, env.num_slots, -1)
    assert float(left[..., -1].min()) >= 0.0 and float(left[..., -1].max()) <= 1.0
    # Cluster look-ahead is closed outside the cluster's window.
    access = np.asarray(state.cluster_access)  # (T, N, C)
    window = np.asarray(state.target_window[: env.num_clusters])
    steps = np.arange(access.shape[0])
    closed = (steps[:, None] < window[None, :, 0]) | (steps[:, None] > window[None, :, 1])
    assert not access.transpose(0, 2, 1)[closed].any()
    action = jnp.concatenate([view[:, 1:4], view[:, 0:1]], axis=-1)
    state, ts = jax.jit(env.step)(state, action)
    assert ts.observation.agents_view.shape == (env.num_agents, env.obs_dim)


def test_lean_precompute_matches_precompute() -> None:
    """The memory-lean planner tables agree with the full-precision ones: identical access
    and counts, values equal to float32 round-off."""
    from sarsat.reference import lean_precompute, precompute

    env = WindowedSarSat(**SMALL, window_steps=(10, 30), background_windows=False)
    state, _ = jax.jit(env.reset)(jax.random.PRNGKey(3))
    full, lean = precompute(env, state), lean_precompute(env, state, pair_budget=50_000)
    assert np.array_equal(full["access"], lean["access"])
    assert np.array_equal(full["team"], lean["team"]) and np.array_equal(full["own"], lean["own"])
    assert lean["own"].dtype == np.uint8 and lean["own"].nbytes == full["access"].nbytes
    np.testing.assert_allclose(full["best_team"], lean["best_team"], rtol=1e-6)
    np.testing.assert_allclose(full["best_own"], lean["best_own"], rtol=1e-6)
