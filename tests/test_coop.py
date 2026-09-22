"""Behavioural tests for :mod:`sarsat.coop` (CoopSarSat: the beam-value / look-ahead
observation and compact global state used for MAPX training).

Uses a small scenario throughout so the whole file runs in well under two minutes on
CPU: 20 satellites, 600 max targets, 8 hotspots of 25 targets, 5 planes -- the same
configuration ``mapx_integration/tests/test_sarsat_coop_wrapper.py`` uses.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sarsat import SarSat, State
from sarsat.coop import SLOT_DIM, CoopSarSat

KEY = jax.random.PRNGKey(0)

SMALL = dict(num_satellites=20, max_targets=600, hotspots=8, hotspot_targets=25, planes=5)


def _env(**kwargs) -> CoopSarSat:
    return CoopSarSat(**SMALL, time_limit=40, **kwargs)


def _slot0_action(view: jax.Array) -> jax.Array:
    """The greedy rule: a slot's pointing features followed by its ``valid`` flag is
    exactly the action that images it (see the module docstring and DECISIONS.md
    issue 21)."""
    return jnp.concatenate([view[:, 1:4], view[:, 0:1]], axis=-1)


def _random_action(key: jax.Array, env: CoopSarSat) -> jax.Array:
    """A random legal action: pointing and side sampled freely, always requesting to
    sense (whether it is honoured depends on the battery, which is what exercises
    varied overlap patterns for the credit test)."""
    k_point, k_side = jax.random.split(key)
    pointing = jax.random.uniform(k_point, (env.num_agents, 2), minval=-1.0, maxval=1.0)
    side = jax.random.bernoulli(k_side, 0.5, (env.num_agents, 1)).astype(jnp.float32)
    sense = jnp.ones((env.num_agents, 1), jnp.float32)
    return jnp.concatenate([pointing, side, sense], axis=-1)


# --------------------------------------------------------------- observation / state


def test_observation_and_global_state_are_shaped_finite_and_bounded() -> None:
    env = _env()
    state, timestep = jax.jit(env.reset)(KEY)
    env.observation_spec.validate(timestep.observation)  # shapes and dtypes

    view = timestep.observation.agents_view
    global_state = timestep.observation.global_state
    assert view.shape == (env.num_agents, env.obs_dim)
    assert global_state.shape == (env.num_agents, env.global_dim)
    assert view.dtype == jnp.float32 and global_state.dtype == jnp.float32

    step = jax.jit(env.step)
    action = _slot0_action(view)
    for _ in range(5):
        state, timestep = step(state, action)
        action = _slot0_action(timestep.observation.agents_view)

    for name, array in [
        ("agents_view", timestep.observation.agents_view),
        ("global_state", timestep.observation.global_state),
    ]:
        assert bool(jnp.all(jnp.isfinite(array))), f"{name} has non-finite entries"
        # Pointing features are in [-1, 1]; log-squashed counts and team summaries add
        # a little headroom above that but everything stays a small, bounded number --
        # roughly [-1, 3] as documented, checked here with a safety margin.
        assert float(array.min()) >= -1.5, f"{name} min {float(array.min())}"
        assert float(array.max()) <= 5.0, f"{name} max {float(array.max())}"


def test_works_with_zero_hotspots() -> None:
    env = CoopSarSat(num_satellites=5, max_targets=50, hotspots=0, time_limit=10)
    state, timestep = jax.jit(env.reset)(KEY)
    env.observation_spec.validate(timestep.observation)
    step = jax.jit(env.step)
    for _ in range(10):
        action = _slot0_action(timestep.observation.agents_view)
        state, timestep = step(state, action)
        assert bool(jnp.all(jnp.isfinite(timestep.observation.agents_view)))
        assert bool(jnp.all(jnp.isfinite(timestep.observation.global_state)))


def test_reset_state_and_refresh_equals_reset() -> None:
    env = _env()
    combined = env.refresh(env.reset_state(KEY))
    direct, _ = jax.jit(env.reset)(KEY)
    for name, a, b in zip(combined._fields, combined, direct, strict=True):
        for leaf_a, leaf_b in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            if jnp.issubdtype(leaf_a.dtype, jnp.floating):
                # Position-like fields (km scale) pick up float32 rounding from XLA
                # fusing the two calls differently under jit; not a logic difference.
                np.testing.assert_allclose(leaf_a, leaf_b, rtol=1e-4, atol=1e-2, err_msg=name)
            else:
                np.testing.assert_array_equal(leaf_a, leaf_b, err_msg=name)


# -------------------------------------------------------------------------- slots


def test_slot_values_are_non_increasing_across_slots() -> None:
    """Slot 0 is the beam that captures the most priority, slot 1 the best over what
    slot 0 leaves, and so on (non-maximum suppression in ``CoopSarSat._slots``), so
    feature index 4 (``value``) of the 8-wide slot layout can only fall going down the
    slots, never rise."""
    env = _env()
    state, timestep = jax.jit(env.reset)(KEY)
    step = jax.jit(env.step)
    for _ in range(8):
        view = timestep.observation.agents_view
        slots = view[:, : SLOT_DIM * env.num_slots].reshape(env.num_agents, env.num_slots, SLOT_DIM)
        value = slots[..., 4]
        assert bool(jnp.all(jnp.diff(value, axis=1) <= 1e-6))
        state, timestep = step(state, _random_action(jax.random.fold_in(KEY, state.step), env))


# ------------------------------------------------------------------------ dynamics


def test_slot0_reward_matches_base_sarsat_dynamics() -> None:
    """CoopSarSat changes only the observation: for the same state and action, its
    per-step team reward must equal what the base :class:`SarSat` dynamics give. The
    base ``State`` is exactly the first 8 fields of ``CoopState`` (same names, same
    order)."""
    base_kwargs = {**SMALL, "time_limit": 40}
    coop_env = _env()
    base_env = SarSat(**base_kwargs)

    coop_state, timestep = jax.jit(coop_env.reset)(KEY)
    coop_step = jax.jit(coop_env.step)
    base_step = jax.jit(base_env.step)

    total_reward = 0.0
    for _ in range(10):
        action = _slot0_action(timestep.observation.agents_view)
        base_state = State(*coop_state[: len(State._fields)])
        _, base_timestep = base_step(base_state, action)
        coop_state, timestep = coop_step(coop_state, action)
        np.testing.assert_allclose(timestep.reward, base_timestep.reward, atol=1e-5)
        total_reward += float(timestep.reward[0])
        if bool(timestep.last()):
            break
    assert total_reward > 0.0, "the slot-0 policy never imaged anything over 10 steps"


def test_transition_with_credit_sums_to_team_reward() -> None:
    env = _env()
    state, _ = jax.jit(env.reset)(KEY)
    step = jax.jit(env.step)
    credit_fn = jax.jit(env.transition_with_credit)
    for i in range(8):
        action = _random_action(jax.random.fold_in(KEY, i), env)
        _, reward, _, credit = credit_fn(state, action)
        assert credit.shape == (env.num_agents,)
        np.testing.assert_allclose(jnp.sum(credit), reward, rtol=1e-5, atol=1e-6)
        state, _ = step(state, action)
