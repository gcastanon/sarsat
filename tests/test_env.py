"""Behavioural tests for the SarSat environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jumanji.types import StepType

from sarsat import SarSat, State
from sarsat.orbits import (
    EARTH_RADIUS_KM,
    EARTH_ROTATION_RAD_S,
    az_el,
    in_az_el_box,
    look_angle_band,
    sample_orbits,
    satellite_frames,
)
from sarsat.targets import TargetSampler

KEY = jax.random.PRNGKey(0)


def _ground_point(
    env: SarSat, state: State, look_deg: float, squint_deg: float, satellite: int = 0
) -> jax.Array:
    """Where the beam meets the Earth when pointed at ``(look, squint)`` at step 1."""
    position, frame = satellite_frames(state.orbit, env.step_seconds)
    look, squint = np.radians([look_deg, squint_deg])
    body = np.array([np.sin(squint), np.cos(squint) * np.sin(look), np.cos(squint) * np.cos(look)])
    origin = np.asarray(position[satellite], np.float64)
    ray = body @ np.asarray(frame[satellite], np.float64)  # body axes -> world
    b, c = ray @ origin, origin @ origin - EARTH_RADIUS_KM**2
    return jnp.asarray(origin + (-b - np.sqrt(b**2 - c)) * ray, jnp.float32)


def _single_target(
    env: SarSat, look_deg: float = 30.0, squint_deg: float = 0.0, satellite: int = 0
) -> State:
    """State whose only active target sits at that look / squint angle from a satellite."""
    state, _ = env.reset(KEY, num_targets=1)
    point = _ground_point(env, state, look_deg, squint_deg, satellite)
    return state._replace(target_pos=state.target_pos.at[0].set(point))


def _look_band_deg(env: SarSat, state: State, satellite: int = 0) -> tuple:
    band = look_angle_band(state.orbit.radius, env._incidence)
    return tuple(float(np.degrees(x[satellite])) for x in band)


def _action(
    env: SarSat, state: State, look_deg: float = 30.0, squint_deg: float = 0.0, sense: float = 1.0
) -> jax.Array:
    """Action pointing every satellite at ``(look, squint)`` in the env's own encoding."""
    n = env.num_agents
    look = jnp.full((n, 1), np.radians(look_deg), jnp.float32)
    squint = jnp.full((n, 1), np.radians(squint_deg), jnp.float32)
    pointing = env._encode_pointing(state.orbit, look, squint)[:, 0]
    return jnp.concatenate([pointing, jnp.full((n, 1), sense, jnp.float32)], axis=-1)


def _incidence_deg(state: State, step: int, env: SarSat) -> np.ndarray:
    """True incidence angle of every satellite-target pair, from the ground up."""
    position, _ = satellite_frames(state.orbit, step * env.step_seconds)
    position, target = np.asarray(position, np.float64), np.asarray(state.target_pos, np.float64)
    los = position[:, None, :] - target[None, :, :]
    vertical = target / np.linalg.norm(target, axis=-1, keepdims=True)
    cos = np.einsum("nmc,mc->nm", los, vertical) / np.linalg.norm(los, axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# ---------------------------------------------------------------------- orbits


def test_orbits_are_circular_orthonormal_and_periodic() -> None:
    orbit = sample_orbits(KEY, 50, (500.0, 600.0), (45.0, 98.0))
    position, frame = satellite_frames(orbit, 1234.0)
    np.testing.assert_allclose(jnp.linalg.norm(position, axis=-1), orbit.radius, rtol=1e-6)
    identity = jnp.broadcast_to(jnp.eye(3), (50, 3, 3))
    np.testing.assert_allclose(frame @ jnp.swapaxes(frame, 1, 2), identity, atol=1e-6)
    # Right-handed, with nadir pointing at the Earth's centre.
    np.testing.assert_allclose(jnp.cross(frame[:, 0], frame[:, 1]), frame[:, 2], atol=1e-6)
    np.testing.assert_allclose(frame[:, 2] * orbit.radius[:, None], -position, rtol=1e-5, atol=1e-2)
    assert 94.0 < float(2 * jnp.pi / orbit.mean_motion.min()) / 60.0 < 97.0  # ~95 min LEO period


def test_ground_track_drifts_west_by_earth_rotation() -> None:
    orbit = sample_orbits(KEY, 1, (550.0, 550.0), (60.0, 60.0))
    period = float(2 * jnp.pi / orbit.mean_motion[0])
    start, _ = satellite_frames(orbit, 0.0)
    end, _ = satellite_frames(orbit, period)
    drift = jnp.arctan2(end[0, 1], end[0, 0]) - jnp.arctan2(start[0, 1], start[0, 0])
    drift = (drift + jnp.pi) % (2 * jnp.pi) - jnp.pi
    np.testing.assert_allclose(drift, -EARTH_ROTATION_RAD_S * period, atol=1e-4)
    np.testing.assert_allclose(end[0, 2], start[0, 2], atol=0.5)  # same latitude [km]


def test_tangent_space_box_matches_explicit_angles() -> None:
    vectors = jax.random.normal(KEY, (20000, 3)) * jnp.asarray([1.0, 1.0, 0.5])
    lower, upper = jnp.deg2rad(jnp.asarray([[-30.0, 5.0], [20.0, 49.0]]))
    az, el = az_el(vectors)
    expected = (az >= lower[0]) & (az <= upper[0]) & (el >= lower[1]) & (el <= upper[1])
    inside = in_az_el_box(vectors, lower, upper)
    assert 100 < int(inside.sum()) < 19000
    np.testing.assert_array_equal(inside, expected)


# --------------------------------------------------------------------- targets


def test_target_split_priorities_and_padding() -> None:
    env = SarSat(num_satellites=2, max_targets=50)
    state, _ = jax.jit(env.reset)(KEY, 20)  # traced num_targets
    assert int(state.target_active.sum()) == 20
    np.testing.assert_allclose(state.target_priority.sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(state.target_priority[:20], 1 / 20)
    assert not state.target_priority[20:].any()
    np.testing.assert_allclose(
        jnp.linalg.norm(state.target_pos, axis=-1), EARTH_RADIUS_KM, rtol=1e-6
    )

    # 80 / 20 land / water, checked against the mask itself.
    is_land = np.unpackbits(np.load(TargetSampler.__init__.__globals__["_MASK_FILE"]))
    is_land = is_land.reshape(180, 360).astype(bool)
    lat, lon = np.asarray(state.target_latlon[:20]).T
    rows = np.clip(np.floor(90.0 - lat).astype(int), 0, 179)
    cols = np.clip(np.floor(lon + 180.0).astype(int), 0, 359)
    assert is_land[rows, cols].sum() == 16
    assert is_land[rows[:16], cols[:16]].all()


def test_reset_rejects_bad_target_count() -> None:
    env = SarSat(num_satellites=2, max_targets=10)
    with pytest.raises(ValueError):
        env.reset(KEY, num_targets=11)


# ---------------------------------------------------------- sensing and reward


def test_target_in_band_is_observed_imaged_once_and_shared() -> None:
    env = SarSat(num_satellites=3, max_targets=5)
    state = _single_target(env, look_deg=30.0)
    view = env._observe(state).agents_view[0]
    # Accessible, and the encoded slot is exactly what the action needs to point back:
    # incidence fraction rescaled to [-1, 1], zero squint, side = right.
    low, high = _look_band_deg(env, state)
    fraction = (30.0 - low) / (high - low)
    np.testing.assert_allclose(view[:4], [1.0, 2 * fraction - 1, 0.0, 1.0], atol=1e-3)

    state, timestep = jax.jit(env.step)(state, _action(env, state, 30.0))
    np.testing.assert_allclose(timestep.reward, jnp.ones(3))  # joint reward = priority = 1
    assert not state.target_active.any()
    assert timestep.step_type == StepType.LAST and not timestep.discount.any()  # terminated

    _, timestep = env.step(state, _action(env, state, 30.0))
    assert not timestep.reward.any()  # no credit twice


@pytest.mark.parametrize("side_switch", [True, False])
@pytest.mark.parametrize(
    ("right", "fraction", "squint"),
    [
        (True, 0.0, 0.0),
        (True, 1.0, 0.0),
        (False, 0.0, 0.0),
        (True, 0.75, 1.0),
        (False, 0.25, -1.0),
        (True, 0.6, 0.6),
        (False, 0.05, 0.3),
    ],
)
def test_observed_slot_copied_into_the_action_images_the_target(
    side_switch: bool, right: bool, fraction: float, squint: float
) -> None:
    """The observation encoding is the action encoding: copying a slot aims the beam.

    Points are given as (side, position in band, squint), so every case is inside the
    band by construction however the incidence band and the squint limit interact, and
    both action layouts are exercised on the same geometry.
    """
    env = SarSat(num_satellites=1, max_targets=1, side_switch=side_switch)
    if not side_switch and not right and fraction == 0.0:
        pytest.skip("the signed encoding cannot express 'left, inner edge': -0.0 == +0.0")
    if side_switch:
        command = jnp.asarray([[2 * fraction - 1, squint, float(right), 1.0]])
    else:
        command = jnp.asarray([[fraction if right else -fraction, squint, 1.0]])
    state, _ = env.reset(KEY, num_targets=1)
    pointing = command[:, : env._pointing_dim]
    look, squint_deg = np.degrees(np.asarray(env._decode_pointing(state.orbit, pointing))[0])
    assert (look >= 0) == right
    state = state._replace(
        target_pos=state.target_pos.at[0].set(_ground_point(env, state, look, squint_deg))
    )

    view = env._observe(state).agents_view[0]
    assert view[0] == 1.0, f"target at look {look:.1f} squint {squint_deg:.1f} is not accessible"
    np.testing.assert_allclose(view[1 : env.action_dim], command[0, :-1], atol=2e-3)

    copied = jnp.concatenate([view[1 : env.action_dim], view[:1]])[None]  # greedy rule
    _, timestep = env.step(state, copied)
    assert timestep.reward[0] > 0, "copying the observed slot into the action missed"


def test_both_action_layouts_decode_to_the_same_angles() -> None:
    """The side switch changes the interface, not the geometry it addresses."""
    signed = SarSat(num_satellites=5, max_targets=2, side_switch=False)
    switched = SarSat(num_satellites=5, max_targets=2, side_switch=True)
    state, _ = signed.reset(KEY)
    cross = jnp.asarray([-1.0, -0.4, 0.0, 0.4, 1.0])
    squint = jnp.asarray([0.3, -0.9, 0.0, 0.9, -0.3])
    a = signed._decode_pointing(state.orbit, jnp.stack([cross, squint], -1))
    b = switched._decode_pointing(
        state.orbit, jnp.stack([2 * jnp.abs(cross) - 1, squint, (cross >= 0) * 1.0], -1)
    )
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert signed.action_dim == 3 and switched.action_dim == 4
    assert switched.obs_dim - signed.obs_dim == signed.max_observed_targets * signed.lookahead_steps


def test_sensor_is_two_sided_but_has_a_nadir_hole() -> None:
    env = SarSat(num_satellites=1, max_targets=1)
    for look in (-30.0, 30.0):  # both sides of the ground track are reachable
        state = _single_target(env, look)
        assert env._target_geometry(state, state.step + 1)[1].all()
    for look in (0.0, 10.0, 17.0):  # inside the minimum incidence: no access
        state = _single_target(env, look)
        assert not env._target_geometry(state, state.step + 1)[1].any()
    for look in (45.0, 60.0):  # beyond the maximum incidence
        state = _single_target(env, look)
        assert not env._target_geometry(state, state.step + 1)[1].any()


def test_every_accessible_target_lies_inside_the_incidence_band() -> None:
    """Check the tangent-space access test against incidence computed at the ground."""
    env = SarSat(num_satellites=20, max_targets=4000)
    state, _ = env.reset(KEY)
    visible = np.asarray(env._target_geometry(state, state.step + 1)[1])
    incidence = _incidence_deg(state, 1, env)
    low, high = env.incidence_deg
    assert visible.sum() > 50, "expected a decent sample of accessible pairs"
    assert incidence[visible].min() >= low - 1e-3
    assert incidence[visible].max() <= high + 1e-3
    # And the band really is the binding constraint for some excluded pairs.
    assert (incidence[~visible] < low).any() and (incidence[~visible] > high).any()


def test_squint_limit_is_enforced() -> None:
    env = SarSat(num_satellites=1, max_targets=1)
    for squint in (-24.0, 0.0, 24.0):
        state = _single_target(env, 30.0, squint)
        assert env._target_geometry(state, state.step + 1)[1].all()
    for squint in (-26.0, 26.0):
        state = _single_target(env, 30.0, squint)
        assert not env._target_geometry(state, state.step + 1)[1].any()


@pytest.mark.parametrize(
    ("d_look_deg", "d_squint_deg", "hit"),
    [(3.9, 0.0, True), (4.1, 0.0, False), (0.0, -3.9, True), (0.0, 4.1, False), (3.9, 3.9, True)],
)
def test_rectangular_beam_edges(d_look_deg: float, d_squint_deg: float, hit: bool) -> None:
    """The beam is +-4 degrees about the boresight in look angle and in squint."""
    env = SarSat(num_satellites=1, max_targets=1)
    state = _single_target(env, 30.0 + d_look_deg, d_squint_deg)
    _, timestep = env.step(state, _action(env, state, 30.0, 0.0))
    assert bool(timestep.reward[0] > 0) == hit


def test_target_beyond_horizon_is_never_visible() -> None:
    env = SarSat(num_satellites=1, max_targets=1)
    state = _single_target(env)
    state = state._replace(target_pos=-state.target_pos)  # antipode
    assert not env._target_geometry(state, state.step + 1)[1].any()


# --------------------------------------------------------------------- battery


def test_battery_enforces_duty_cycle() -> None:
    env = SarSat(num_satellites=1, max_targets=5, sense_cost=0.25, recharge_rate=0.05)
    state, timestep = env.reset(KEY)
    step = jax.jit(env.step)
    sensed = []
    for _ in range(100):
        before = state.battery[0]
        sense_slot = env._pointing_dim
        assert bool(timestep.observation.action_mask[0, sense_slot]) == bool(before >= 0.25 - 1e-6)
        state, timestep = step(state, _action(env, state))
        sensed.append(bool(state.battery[0] < before))
        assert 0.0 <= float(state.battery[0]) <= 1.0
    # Always requesting: burn the initial charge, then settle at recharge / cost = 20%.
    assert all(sensed[:4])
    assert np.mean(sensed[20:]) == pytest.approx(0.2, abs=0.02)

    state = state._replace(battery=jnp.zeros(1))
    for _ in range(20):
        state, _ = step(state, _action(env, state, sense=0.0))
    np.testing.assert_allclose(state.battery, 1.0)  # recharges fully and saturates


# ----------------------------------------------------------------- observation


def test_neighbors_are_nearest_first_and_observation_is_bounded() -> None:
    env = SarSat(num_satellites=30, max_targets=200)
    state, timestep = env.reset(KEY)
    view = timestep.observation.agents_view
    assert view.shape == (30, env.obs_dim) and view.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(view))) and float(jnp.abs(view).max()) <= 1.0 + 1e-6

    start = (env._pointing_dim + 1) * env.max_observed_targets * env.lookahead_steps
    distance = view[:, start : start + 3 * env.num_neighbors].reshape(30, -1, 3)[..., 2]
    assert bool(jnp.all(jnp.diff(distance, axis=1) >= 0))

    position, _ = satellite_frames(state.orbit, env.step_seconds)
    pairwise = jnp.linalg.norm(position[:, None] - position[None], axis=-1)
    expected = jnp.sort(pairwise, axis=1)[:, 1 : 1 + env.num_neighbors] / env._max_distance
    np.testing.assert_allclose(distance, expected, rtol=1e-4)


def test_single_satellite_has_no_neighbor_features() -> None:
    env = SarSat(num_satellites=1, max_targets=3, max_observed_targets=10)
    assert env.obs_dim == 4 * 3 * 2 + 1  # (flag + 3 pointing) x 3 slots x 2 steps + battery
    env.observation_spec.validate(env.reset(KEY)[1].observation)


# ------------------------------------------------------------- episodes and scale


def test_episode_truncates_at_time_limit_with_live_discount() -> None:
    env = SarSat(num_satellites=2, max_targets=10, time_limit=5)
    state, _ = env.reset(KEY)
    for i in range(5):
        state, timestep = env.step(state, _action(env, state, sense=0.0))
        assert (timestep.step_type == StepType.LAST) == (i == 4)
    assert timestep.discount.all()


def test_vmapped_rollout_at_full_scale_is_finite() -> None:
    env = SarSat(num_satellites=1000, max_targets=1000)

    def rollout(key: jax.Array) -> jax.Array:
        state, _ = env.reset(key)

        def body(state: State, key: jax.Array) -> tuple:
            action = jax.random.uniform(key, (1000, 3), minval=-1.0)
            state, timestep = env.step(state, action)
            ok = jnp.all(jnp.isfinite(timestep.observation.agents_view))
            return state, (timestep.reward[0], ok)

        _, (reward, ok) = jax.lax.scan(body, state, jax.random.split(key, 5))
        return reward.sum(), ok.all()

    total, ok = jax.jit(jax.vmap(rollout))(jax.random.split(KEY, 2))
    assert bool(ok.all()) and bool(jnp.all((total >= 0) & (total <= 1 + 1e-6)))


# ------------------------------------------------------ hotspots and constellations


def test_hotspot_field_layout_and_priorities() -> None:
    env = SarSat(
        num_satellites=2, max_targets=500, hotspots=4, hotspot_targets=50, hotspot_weight=10.0
    )
    state, _ = jax.jit(env.reset)(KEY)
    prio = np.asarray(state.target_priority)
    np.testing.assert_allclose(prio.sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(prio[:200] / prio[200], 10.0, rtol=1e-5)  # hot vs background
    assert np.allclose(prio[200:], prio[200])
    latlon = np.asarray(state.target_latlon)
    # Slot i belongs to cluster i % hotspots: members scatter within a few sigma of each
    # other, far tighter than the globe.
    for c in range(4):
        members = latlon[c:200:4]
        spread = np.ptp(members[:, 0]) * 111.0
        assert spread < 8 * env.hotspot_radius_km, f"cluster {c} spans {spread:.0f} km"
    assert np.ptp(latlon[200:, 1]) > 180.0  # the background still covers the globe
    assert np.all(np.abs(latlon[:, 0]) <= 89.0) and np.all(np.abs(latlon[:, 1]) <= 180.0)

    # Trimming num_targets drops background first and keeps every hotspot target.
    state, _ = env.reset(KEY, num_targets=250)
    assert bool(state.target_active[:200].all()) and int(state.target_active.sum()) == 250
    np.testing.assert_allclose(state.target_priority.sum(), 1.0, rtol=1e-6)


def test_hotspots_must_fit() -> None:
    with pytest.raises(ValueError):
        SarSat(num_satellites=1, max_targets=100, hotspots=3, hotspot_targets=50)


def test_structured_constellations() -> None:
    walker = SarSat(num_satellites=12, max_targets=5, planes=3)
    orbit = walker.reset(KEY)[0].orbit
    plane = np.arange(12) // 4
    for c in range(3):
        assert np.allclose(
            np.asarray(orbit.raan)[plane == c], np.asarray(orbit.raan)[plane == c][0]
        )
        assert np.allclose(
            np.asarray(orbit.radius)[plane == c], np.asarray(orbit.radius)[plane == c][0]
        )
    raans = np.sort(np.unique(np.asarray(orbit.raan).round(5)))
    np.testing.assert_allclose(np.diff(raans), 2 * np.pi / 3, atol=1e-4)  # even plane spacing
    in_plane = np.sort(np.asarray(orbit.phase)[plane == 0])
    np.testing.assert_allclose(np.diff(in_plane) % (2 * np.pi), np.pi / 2, atol=1e-4)  # Walker

    train = SarSat(num_satellites=12, max_targets=5, planes=3, train_spacing_s=120.0)
    orbit = train.reset(KEY)[0].orbit
    gap = (np.asarray(orbit.phase)[0] - np.asarray(orbit.phase)[1]) % (2 * np.pi)
    np.testing.assert_allclose(gap / float(orbit.mean_motion[0]), 120.0, rtol=1e-4)
    position, _ = satellite_frames(orbit, 0.0)
    separation = np.linalg.norm(np.asarray(position)[0] - np.asarray(position)[1])
    assert 800 < separation < 1000  # two minutes of LEO flight, ~900 km

    with pytest.raises(ValueError):
        SarSat(num_satellites=10, max_targets=5, planes=3)
