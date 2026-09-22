"""SarSat: a cooperative multi-agent Earth-observation environment in JAX.

Each agent is a non-manoeuvring satellite on a circular low Earth orbit. Every 60 s
step an agent chooses where to point its rectangular beam (azimuth, elevation) and
whether to sense. Any active target inside a sensing beam pays its priority to the
whole team, once, and is removed. Sensing drains a battery that recharges slowly, so
satellites must respect a duty cycle.

The design rationale is recorded issue by issue in DECISIONS.md; everything that was
deliberately left out is listed in IMPROVEMENTS.md.
"""

from functools import cached_property
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import StepType, TimeStep, restart

from sarsat.orbits import (
    EARTH_RADIUS_KM,
    az_el,
    cross_track_band,
    in_az_el_box,
    in_incidence_band,
    look_angle_band,
    sample_orbits,
    sample_planes,
    satellite_frames,
)
from sarsat.targets import TargetSampler, latlon_to_unit_vector
from sarsat.types import Observation, Orbit, State

MAX_SATELLITES = 1000


class SarSat(Environment[State, specs.BoundedArray, Observation]):
    """Cooperative target-imaging with a constellation of LEO satellites.

    **Action** float32, one row per satellite. With the default ``side_switch=True`` it
    has four slots, ``[incidence, squint, side, sense]``:

    * ``[0]`` incidence command in ``[-1, 1]``, swept linearly from the inner to the
      outer edge of the accessible band (``incidence_deg``)
    * ``[1]`` squint command in ``[-1, 1]``, scaled to ``+-max_squint_deg``
    * ``[2]`` side: ``0`` looks left of the ground track, ``1`` looks right
    * ``[3]`` sense flag: ``> 0.5`` requests imaging during this step

    With ``side_switch=False`` the side is folded into the sign of the first slot
    (negative left, positive right, magnitude sweeping the band) and the action is
    ``[cross-track, squint, sense]``. DECISIONS.md issue 21 compares the two.

    The sensor is side-looking, as a SAR is: it cannot image at nadir. Incidence is
    measured at the target between the line of sight and the local vertical; the look
    angle is measured at the satellite from nadir, and is always the smaller of the two
    (:func:`sarsat.orbits.look_angle_band`). Squint is the along-track angle off
    broadside. The accessible region is therefore a pair of annular sectors, one on each
    side of the ground track. Pointing is instantaneous, so a satellite may look left on
    one step and right on the next.

    **Step order.** Time advances by ``step_seconds`` and the action is then applied at
    the *new* satellite positions. An observation therefore describes the geometry of
    the coming steps, and its first look-ahead slot is exactly the geometry under which
    the action chosen from it will be executed.

    **Observation** ``agents_view`` ``(N, obs_dim)`` float32, egocentric, roughly in
    ``[-1, 1]``:

    * ``max_observed_targets`` target slots x ``lookahead_steps`` x
      ``(accessible, <pointing features>)`` -- the accessible targets nearest the inner
      edge of the band, tracked over the next ``lookahead_steps`` steps (zeros where not
      accessible). The pointing features are exactly the action's pointing slots in the
      action's own encoding, so a slot's features followed by its ``accessible`` flag
      *is* the action that images that target;
    * ``num_neighbors`` x ``(az / pi, el / (pi / 2), distance / max_distance)`` -- the
      nearest satellites at the next step, nearest first;
    * own battery level.

    **Reward** ``(N,)``: the same team reward for every agent -- the summed priority of
    targets newly imaged this step. Priorities sum to one, so the episode return is the
    priority-weighted fraction of targets imaged.

    **Episode end.** Termination (discount 0) when no target is left; truncation
    (discount 1) after ``time_limit`` steps.
    """

    def __init__(
        self,
        num_satellites: int = 8,
        max_targets: int = 100,
        time_limit: int = 180,
        step_seconds: float = 60.0,
        altitude_km: Tuple[float, float] = (500.0, 600.0),
        inclination_deg: Tuple[float, float] = (45.0, 98.0),
        beam_half_width_deg: Tuple[float, float] = (4.0, 4.0),
        incidence_deg: Tuple[float, float] = (20.0, 45.0),
        max_squint_deg: float = 25.0,
        side_switch: bool = True,
        land_fraction: float = 0.8,
        hotspots: int = 0,
        hotspot_targets: int = 50,
        hotspot_radius_km: float = 100.0,
        hotspot_weight: float = 10.0,
        planes: int = 0,
        train_spacing_s: float = 0.0,
        sense_cost: float = 0.10,
        recharge_rate: float = 0.02,
        lookahead_steps: int = 2,
        max_observed_targets: int = 10,
        num_neighbors: int = 5,
    ) -> None:
        """Create the environment.

        Args:
            num_satellites: number of agents, 1 to 1000.
            max_targets: static capacity of the target arrays. ``reset`` may request any
                number of targets up to this value; the rest is inactive padding.
            time_limit: episode length in steps.
            step_seconds: simulated time per step.
            altitude_km: ``(min, max)`` of the uniform altitude distribution.
            inclination_deg: ``(min, max)`` of the uniform inclination distribution.
            beam_half_width_deg: rectangular beam half-widths ``(cross-track, squint)``.
            incidence_deg: ``(min, max)`` incidence angle at the target. The sensor is
                two-sided, so the accessible region is this band on both sides of the
                ground track, with a hole over nadir.
            max_squint_deg: largest along-track angle off broadside.
            side_switch: choose the look side with a categorical action slot (default)
                rather than the sign of the continuous cross-track slot.
            land_fraction: fraction of background targets on land; the rest are on water.
            hotspots: number of high-value clusters. Zero (the default) gives the plain
                uniform target field. The first ``hotspots * hotspot_targets`` target slots
                are hotspot targets, Gaussian-scattered around cluster centres on land;
                the remaining slots are the uniform background. See DECISIONS.md issue 22
                for why this is the recommended field for multi-agent training.
            hotspot_targets: targets per hotspot cluster.
            hotspot_radius_km: one-sigma radius of a hotspot cluster.
            hotspot_weight: priority of a hotspot target relative to a background target.
            planes: zero (the default) draws independent random orbits; otherwise the
                constellation is ``planes`` orbital planes of ``num_satellites / planes``
                satellites each, Walker-spread or flown as trains.
            train_spacing_s: with ``planes > 0``, the along-track spacing within a plane
                in seconds; zero spreads the plane evenly (Walker), a few seconds makes
                formation flyers, a few minutes makes trains.
            sense_cost: battery fraction consumed by one step of sensing.
            recharge_rate: battery fraction regained every step. The sustainable duty
                cycle is ``recharge_rate / sense_cost`` (20% by default).
            lookahead_steps: number of future steps described by the observation.
            max_observed_targets: target slots in the observation.
            num_neighbors: neighbouring satellites in the observation.
        """
        if not 1 <= num_satellites <= MAX_SATELLITES:
            raise ValueError(f"num_satellites must be in [1, {MAX_SATELLITES}]")
        if max_targets < 1 or lookahead_steps < 1:
            raise ValueError("max_targets and lookahead_steps must be at least 1")
        if not 0 < incidence_deg[0] < incidence_deg[1]:
            raise ValueError("incidence_deg must be an increasing (min, max) pair above 0")
        if incidence_deg[1] + beam_half_width_deg[0] >= 90:
            raise ValueError("max incidence plus beam half-width must stay below 90 degrees")
        if not 0 < max_squint_deg + beam_half_width_deg[1] < 90:
            raise ValueError("squint limit plus beam half-width must stay below 90 degrees")
        if not 0 < recharge_rate <= sense_cost <= 1:
            raise ValueError("need 0 < recharge_rate <= sense_cost <= 1")
        if hotspots < 0 or (hotspots and hotspots * hotspot_targets > max_targets):
            raise ValueError("hotspots * hotspot_targets must fit inside max_targets")
        if planes and num_satellites % planes:
            raise ValueError("num_satellites must be a multiple of planes")

        self.num_agents = num_satellites
        self.max_targets = max_targets
        self.time_limit = time_limit
        self.side_switch = side_switch
        # Pointing features per observation slot, which are also the pointing action slots.
        self._pointing_dim = 3 if side_switch else 2
        self.action_dim = self._pointing_dim + 1  # + sense
        self.step_seconds = step_seconds
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg
        self.land_fraction = land_fraction
        self.hotspots = hotspots
        self.hotspot_targets = hotspot_targets
        self.hotspot_radius_km = hotspot_radius_km
        self.hotspot_weight = hotspot_weight
        self.planes = planes
        self.train_spacing_s = train_spacing_s
        self.sense_cost = sense_cost
        self.recharge_rate = recharge_rate
        self.lookahead_steps = lookahead_steps
        self.max_observed_targets = min(max_observed_targets, max_targets)
        self.num_neighbors = min(num_neighbors, num_satellites - 1)

        self._beam_half_width = jnp.deg2rad(jnp.asarray(beam_half_width_deg, jnp.float32))
        self.incidence_deg = tuple(incidence_deg)
        self._incidence = jnp.deg2rad(jnp.asarray(incidence_deg, jnp.float32))
        self._max_squint = jnp.deg2rad(jnp.asarray(max_squint_deg, jnp.float32))
        self._max_distance = 2.0 * (EARTH_RADIUS_KM + altitude_km[1])
        self._sample_targets = TargetSampler()
        self.obs_dim = (
            (self._pointing_dim + 1) * self.max_observed_targets * lookahead_steps
            + 3 * self.num_neighbors
            + 1
        )
        super().__init__()

    # ------------------------------------------------------------------ #
    # Jumanji API
    # ------------------------------------------------------------------ #

    def reset(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> Tuple[State, TimeStep[Observation]]:
        """Start an episode with random orbits and ``num_targets`` random targets.

        ``num_targets`` defaults to ``max_targets`` and may be a traced integer, so the
        number of targets can change between episodes without recompiling.
        """
        state = self._initial_state(key, num_targets)
        return state, restart(self._observe(state), shape=(self.num_agents,))

    def _initial_state(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> State:
        """The state :meth:`reset` starts from; subclasses extend it here."""
        if num_targets is None:
            num_targets = self.max_targets
        elif isinstance(num_targets, int) and not 1 <= num_targets <= self.max_targets:
            raise ValueError(f"num_targets must be in [1, max_targets={self.max_targets}]")
        num_targets = jnp.clip(num_targets, 1, self.max_targets)

        key, k_orbit, k_target = jax.random.split(key, 3)
        target_active = jnp.arange(self.max_targets) < num_targets
        target_latlon, weight = self._sample_target_field(k_target, num_targets)
        weight = jnp.where(target_active, weight, 0.0)

        if self.planes:
            orbit = sample_planes(
                k_orbit,
                self.num_agents,
                self.planes,
                self.altitude_km,
                self.inclination_deg,
                self.train_spacing_s,
            )
        else:
            orbit = sample_orbits(k_orbit, self.num_agents, self.altitude_km, self.inclination_deg)

        return State(
            key=key,
            step=jnp.zeros((), jnp.int32),
            orbit=orbit,
            battery=jnp.ones((self.num_agents,), jnp.float32),
            target_latlon=target_latlon,
            target_pos=EARTH_RADIUS_KM * latlon_to_unit_vector(target_latlon),
            target_priority=weight / jnp.sum(weight),
            target_active=target_active,
        )

    def _sample_target_field(
        self, key: jax.Array, num_targets: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Lat/lon [deg] and unnormalised priority weight for every target slot.

        Slots ``[0, hotspots * hotspot_targets)`` are hotspot targets: slot ``i`` belongs
        to cluster ``i % hotspots``, scattered with a Gaussian of ``hotspot_radius_km``
        around a centre drawn on land, and weighted ``hotspot_weight``. The remaining
        slots are the uniform background with the usual land / water split and weight 1.
        With ``hotspots == 0`` this is the plain uniform field.
        """
        slot = jnp.arange(self.max_targets)
        num_hot = self.hotspots * self.hotspot_targets
        background = slot - num_hot
        on_land = background < jnp.round(self.land_fraction * (num_targets - num_hot))
        latlon = self._sample_targets(key, on_land)
        if not self.hotspots:
            return latlon, jnp.ones((self.max_targets,), jnp.float32)

        k_centre, k_scatter = jax.random.split(key)
        centres = self._sample_targets(k_centre, jnp.ones((self.hotspots,), bool))
        cluster = slot % self.hotspots
        sigma_deg = self.hotspot_radius_km / 111.0
        scatter = sigma_deg * jax.random.normal(k_scatter, (self.max_targets, 2))
        lat = jnp.clip(centres[cluster, 0] + scatter[:, 0], -89.0, 89.0)
        lon = centres[cluster, 1] + scatter[:, 1] / jnp.cos(jnp.deg2rad(lat))
        lon = (lon + 180.0) % 360.0 - 180.0
        is_hot = slot < num_hot
        latlon = jnp.where(is_hot[:, None], jnp.stack([lat, lon], axis=-1), latlon)
        weight = jnp.where(is_hot, self.hotspot_weight, 1.0).astype(jnp.float32)
        return latlon, weight

    def step(self, state: State, action: jax.Array) -> Tuple[State, TimeStep[Observation]]:
        """Advance one time step, then point, sense, score and recharge."""
        step = state.step + 1
        _, sensing, in_beam = self._apply_action(state, step, action)
        imaged = jnp.any(in_beam, axis=0)  # (M,); a target pays once however many beams hit
        reward = jnp.sum(jnp.where(imaged, state.target_priority, 0.0))

        state = state._replace(
            step=step,
            battery=jnp.clip(
                state.battery + self.recharge_rate - self.sense_cost * sensing, 0.0, 1.0
            ),
            target_active=state.target_active & ~imaged,
        )

        all_imaged = ~jnp.any(state.target_active)
        timestep = TimeStep(
            step_type=jnp.where(
                all_imaged | (step >= self.time_limit), StepType.LAST, StepType.MID
            ),
            reward=jnp.full((self.num_agents,), reward, jnp.float32),
            discount=jnp.full((self.num_agents,), jnp.where(all_imaged, 0.0, 1.0), jnp.float32),
            observation=self._observe(state),
            extras={},
        )
        return state, timestep

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        n = self.num_agents
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=specs.Array((n, self.obs_dim), jnp.float32, "agents_view"),
            action_mask=specs.BoundedArray((n, self.action_dim), bool, False, True, "action_mask"),
            step_count=specs.BoundedArray((n,), jnp.int32, 0, self.time_limit, "step_count"),
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        # Continuous pointing slots are bounded to [-1, 1]; categorical slots (side, sense)
        # carry a category index in {0, 1} stored as a float.
        categorical = self.action_dim - 2
        return specs.BoundedArray(
            shape=(self.num_agents, self.action_dim),
            dtype=jnp.float32,
            minimum=jnp.asarray([-1.0, -1.0] + [0.0] * categorical),
            maximum=jnp.asarray([1.0, 1.0] + [1.0] * categorical),
            name="action",
        )

    @cached_property
    def reward_spec(self) -> specs.Array:
        return specs.Array((self.num_agents,), jnp.float32, "reward")

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray((self.num_agents,), jnp.float32, 0.0, 1.0, "discount")

    # ------------------------------------------------------------------ #
    # Battery, geometry and observation
    # ------------------------------------------------------------------ #

    def _can_sense(self, battery: jax.Array) -> jax.Array:
        """Whether the battery holds one step of sensing (tolerant of float32 round-off)."""
        return battery >= self.sense_cost - 1e-6

    def _apply_action(
        self, state: State, step: jax.Array, action: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Decode an action at ``step``: pointing angles (N, 2) [rad], the (N,) sensing
        flags actually honoured, and the (N, M) targets caught in a sensing beam."""
        pointing = self._decode_pointing(state.orbit, action[:, : self._pointing_dim])
        sensing = (action[:, self._pointing_dim] > 0.5) & self._can_sense(state.battery)
        body, visible = self._target_geometry(state, step)
        beam_lower = (pointing - self._beam_half_width)[:, None, :]
        beam_upper = (pointing + self._beam_half_width)[:, None, :]
        in_beam = visible & sensing[:, None] & in_az_el_box(body, beam_lower, beam_upper)
        return pointing, sensing, in_beam

    def _decode_pointing(self, orbit: Orbit, pointing: jax.Array) -> jax.Array:
        """Map the pointing action slots ``(N, pointing_dim)`` to ``(N, 2)`` look and
        squint angles [rad].

        The incidence slot sweeps from the inner to the outer edge of the accessible band;
        the side comes from the categorical slot (``side_switch``) or from the sign of
        the incidence slot. The available cross-track range depends on the squint
        commanded alongside it, because the two combine into the off-nadir angle the
        incidence band constrains. Every action is therefore legal and lands somewhere
        inside the band; the encoding has no way to express "nadir", because a
        side-looking sensor has none.
        """
        off_nadir_min, off_nadir_max = look_angle_band(orbit.radius, self._incidence)
        first, squint = jnp.clip(pointing[:, 0], -1.0, 1.0), jnp.clip(pointing[:, 1], -1.0, 1.0)
        if self.side_switch:
            fraction, right = 0.5 * (first + 1.0), pointing[:, 2] > 0.5
        else:
            fraction, right = jnp.abs(first), first >= 0.0
        squint = squint * self._max_squint
        low, high = cross_track_band(off_nadir_min, off_nadir_max, squint)
        look = low + fraction * (high - low)
        return jnp.stack([jnp.where(right, look, -look), squint], axis=-1)

    def _encode_pointing(self, orbit: Orbit, look: jax.Array, squint: jax.Array) -> jax.Array:
        """Inverse of :meth:`_decode_pointing`: the pointing action slots that aim at
        ``(look, squint)``, as a trailing axis of ``pointing_dim``.

        ``look`` and ``squint`` are ``(..., N, slots)``: the satellite axis is second
        from the end. Feeding a target's own angles back in as an action points the beam
        straight at it, which is what makes the observation directly actionable.
        """
        band = (x[:, None] for x in look_angle_band(orbit.radius, self._incidence))
        low, high = cross_track_band(*band, squint)
        fraction = jnp.clip((jnp.abs(look) - low) / jnp.maximum(high - low, 1e-6), 0.0, 1.0)
        right = look >= 0.0
        squint = squint / self._max_squint
        if self.side_switch:
            return jnp.stack([2.0 * fraction - 1.0, squint, right.astype(jnp.float32)], axis=-1)
        return jnp.stack([jnp.where(right, fraction, -fraction), squint], axis=-1)

    def _target_geometry(self, state: State, step: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Line of sight to every target in every satellite's body axes at ``step``.

        Returns the ``(N, M, 3)`` line-of-sight vectors [km] and an ``(N, M)`` mask of the
        targets that can be imaged: active, inside the incidence band on either side,
        within the squint limit, and not hidden behind the horizon.
        """
        _, frame = satellite_frames(state.orbit, step * self.step_seconds)
        radius = state.orbit.radius[:, None]
        # Rotate the targets into body axes, then move the origin from the Earth's centre
        # to the satellite, which sits at (0, 0, -radius) in its own axes.
        body = jnp.einsum("mc,nbc->nmb", state.target_pos, frame).at[..., 2].add(radius)
        # A target on the horizon has a nadir component of (radius^2 - R^2) / radius;
        # nearer targets have less, targets hidden behind the limb have more.
        above_horizon = body[..., 2] < radius - EARTH_RADIUS_KM**2 / radius
        off_nadir_min, off_nadir_max = look_angle_band(state.orbit.radius, self._incidence)
        in_reach = in_incidence_band(
            body, off_nadir_min[:, None], off_nadir_max[:, None], self._max_squint
        )
        return body, state.target_active & above_horizon & in_reach

    def _observe(self, state: State) -> Observation:
        """Build the egocentric observation for the steps following ``state.step``."""
        n = self.num_agents

        # Targets: look-ahead geometry (L, N, M), reduced per satellite to the slots that
        # come closest to nadir. Angles are only evaluated for the selected slots.
        future_steps = state.step + 1 + jnp.arange(self.lookahead_steps)
        body, visible = jax.vmap(self._target_geometry, in_axes=(None, 0))(state, future_steps)
        cos_off_nadir = body[..., 2] / jnp.linalg.norm(body, axis=-1)
        score = jnp.max(jnp.where(visible, cos_off_nadir, -jnp.inf), axis=0)
        _, slots = jax.lax.top_k(score, self.max_observed_targets)  # (N, slots)
        visible = jnp.take_along_axis(visible, slots[None], axis=2)
        az, el = az_el(jnp.take_along_axis(body, slots[None, :, :, None], axis=2))
        pointing = self._encode_pointing(state.orbit, az, el)  # the action's own encoding
        target_features = jnp.concatenate(
            [visible[..., None].astype(jnp.float32), jnp.where(visible[..., None], pointing, 0.0)],
            axis=-1,
        )  # (L, N, slots, 1 + pointing_dim)
        target_features = jnp.moveaxis(target_features, 0, 2).reshape(n, -1)

        # Neighbours: nearest satellites at the next step, in the same body frame.
        position, frame = satellite_frames(state.orbit, (state.step + 1) * self.step_seconds)
        offset = position[None, :, :] - position[:, None, :]  # (N, N, 3)
        distance_sq = jnp.sum(offset**2, axis=-1) + jnp.diag(jnp.full((n,), jnp.inf))
        _, nearest = jax.lax.top_k(-distance_sq, self.num_neighbors)  # (N, K)
        offset = jnp.take_along_axis(offset, nearest[:, :, None], axis=1)  # (N, K, 3)
        neighbor_az, neighbor_el = az_el(jnp.einsum("nkc,nbc->nkb", offset, frame))
        distance = jnp.linalg.norm(offset, axis=-1)
        neighbor_features = jnp.stack(
            [neighbor_az / jnp.pi, neighbor_el / (jnp.pi / 2), distance / self._max_distance],
            axis=-1,
        ).reshape(n, -1)

        # Pointing and side are always available; sensing needs charge.
        action_mask = jnp.ones((n, self.action_dim), bool)
        action_mask = action_mask.at[:, self._pointing_dim].set(self._can_sense(state.battery))
        return Observation(
            agents_view=jnp.concatenate(
                [target_features, neighbor_features, state.battery[:, None]], axis=-1
            ),
            action_mask=action_mask,
            step_count=jnp.full((n,), state.step, jnp.int32),
        )
