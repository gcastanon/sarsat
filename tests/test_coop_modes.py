"""Mode-aware CoopSarSat beam slots and reference policies (DECISIONS.md issue 33)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sarsat import SensorMode
from sarsat.coop import SLOT_DIM, CoopSarSat
from sarsat.quality import MODES, QualityCoopSarSat, QualitySarSat
from sarsat.reference import REFERENCE_POLICIES, ReferenceController

KEY = jax.random.PRNGKey(0)
# Every target fits in the candidate set (num_candidates = 64), so a slot's value is
# exactly what its look images.
SMALL = dict(
    num_satellites=10,
    max_targets=60,
    hotspots=2,
    hotspot_targets=20,
    hotspot_radius_km=30.0,
    planes=5,
    time_limit=40,
)
QUALITY = dict(hotspot_quality=(1.0, 3.0, 5.0, 7.0), background_quality=(1.0, 3.0, 5.0))


def _slots(env, view):
    return np.asarray(view[:, : env.slot_dim * env.num_slots]).reshape(
        env.num_agents, env.num_slots, env.slot_dim
    )


def _look(slot, num_modes):
    """A slot's pointing followed by its mode: the look that captures its value."""
    mode = np.argmax(slot[..., -num_modes:], axis=-1) + 1
    return np.concatenate([slot[..., 1:4], np.where(slot[..., 0] > 0.5, mode, 0)[..., None]], -1)


def test_one_explicit_mode_keeps_the_slots_and_adds_its_flag() -> None:
    plain = CoopSarSat(**SMALL)
    moded = CoopSarSat(**SMALL, modes=[SensorMode()])
    assert moded.slot_dim == SLOT_DIM + 1 and plain.slot_dim == SLOT_DIM
    assert moded.obs_dim == plain.obs_dim + plain.num_slots
    a, b = _slots(plain, plain.reset(KEY)[1].observation.agents_view), None
    b = _slots(moded, moded.reset(KEY)[1].observation.agents_view)
    np.testing.assert_array_equal(a, b[..., :SLOT_DIM])  # a full battery affords the mode
    np.testing.assert_array_equal(b[..., SLOT_DIM], b[..., 0])  # the one-hot is the valid flag


def test_each_slot_look_images_exactly_its_value() -> None:
    env = QualityCoopSarSat(**SMALL, **QUALITY, modes=MODES)
    state, timestep = env.reset(KEY)
    transition = jax.jit(env.transition)
    unit = float(jnp.max(state.target_priority))
    checked, used = 0, set()
    for t in range(30):
        slots = _slots(env, timestep.observation.agents_view)
        np.testing.assert_array_equal(slots[..., -env.num_modes :].sum(-1), slots[..., 0])
        for n, k in zip(*np.nonzero(slots[..., 0] > 0.5)):
            action = np.zeros((env.num_agents, env.action_dim), np.float32)
            action[n] = _look(slots[n, k], env.num_modes)
            _, reward, _ = transition(state, jnp.asarray(action))
            if k == 0:  # later slots are valued over what the earlier ones leave
                value = np.expm1(slots[n, k, 4] * np.log1p(50.0)) * unit
                assert float(reward) == pytest.approx(value, rel=1e-4)
                checked += 1
            else:
                assert float(reward) > 0.0
            used.add(int(action[n, -1]))
        greedy = _look(slots[:, 0], env.num_modes)
        state, timestep = env.step(state, jnp.asarray(greedy))
    assert checked >= 5 and len(used) >= 2


def test_a_slot_never_offers_a_mode_the_battery_cannot_pay_for() -> None:
    env = QualityCoopSarSat(**SMALL, **QUALITY, modes=MODES)
    state, _ = env.reset(KEY)
    state = env.refresh(state._replace(battery=jnp.full((env.num_agents,), 0.15)))
    slots = _slots(env, env.observe(state).agents_view)
    mode = np.argmax(slots[..., -env.num_modes :], axis=-1)
    affordable = np.asarray(env._mode_cost)[mode] <= 0.15
    assert np.all(affordable | (slots[..., 0] < 0.5))


@pytest.mark.parametrize("policy", REFERENCE_POLICIES)
def test_reference_policies_choose_affordable_modes_that_image(policy) -> None:
    env = QualitySarSat(**SMALL, **QUALITY, modes=MODES)
    state, _ = env.reset(KEY)
    controller = ReferenceController(env, state, policy)
    step = jax.jit(env.step)
    total, modes = 0.0, set()
    for _ in range(env.time_limit):
        action = controller(state)
        look = np.asarray(action[:, -1]).astype(int)
        cost = np.asarray(env._mode_cost)[np.maximum(look - 1, 0)]
        assert np.all((look == 0) | (np.asarray(state.battery) >= cost - 1e-6))
        modes |= set(look[look > 0].tolist())
        state, timestep = step(state, action)
        total += float(timestep.reward[0])
    assert total > 0.0 and len(modes) >= 2
