"""CoopSarSat: SarSat with an observation built for multi-agent learning.

The dynamics, action and reward are exactly those of :class:`sarsat.env.SarSat`; only what
the agents see changes. Everything in the observation is something a satellite could hold
on board: the target catalogue with its imaged flags (the base observation already assumes
that), the public ephemeris of its teammates, and its own battery. Every feature is scaled
to roughly ``[0, 1]`` (pointing features to ``[-1, 1]``, the action's own encoding).

``agents_view`` layout (``obs_dim = 8 * slots + 2 * horizon + 5``):

* ``slots`` x ``(valid, incidence, squint, side, value, scarce_value, team_future,
  contention)`` -- the best beams available at the next step, found by non-maximum
  suppression over the accessible targets: slot 0 is the beam that captures the most
  priority (what ``greedy_beam`` aims at), slot 1 the best beam over what slot 0 leaves,
  and so on. ``value`` is the priority the beam captures, ``scarce_value`` the same with
  hotspot targets discounted by how often the team will still see their cluster,
  ``team_future`` that count itself, and ``contention`` how many charged teammates can see
  the slot's centre target at the same step (with ``ranked_contention`` only those with a
  lower index, a convention that lets exactly one contender see none).
* ``horizon`` x ``(value, scarce_value)`` -- what is left of the hotspot clusters this
  satellite will have access to at each of the next ``horizon`` steps.
* ``(battery, can_sense, time_remaining, visible_value, visible_count)``.

``global_state`` is the agent's own view followed by a fixed-size team summary (battery
statistics, remaining value, the sorted remaining cluster fractions, the team's summed
look-ahead), so a centralised critic stays small whatever the constellation size.
"""

from functools import cached_property
from typing import NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart

from sarsat.env import SarSat
from sarsat.orbits import (
    EARTH_RADIUS_KM,
    az_el,
    in_az_el_box,
    in_incidence_band,
    look_angle_band,
    satellite_frames,
)
from sarsat.types import Orbit, State

SLOT_DIM = 8
SELF_DIM = 5
_SCARCITY_KAPPA = 8.0  # accesses at which a cluster's scarcity discount is one half


class CoopState(NamedTuple):
    """:class:`sarsat.types.State` plus the constellation's access schedule to the clusters."""

    key: jax.Array
    step: jax.Array
    orbit: Orbit
    battery: jax.Array
    target_latlon: jax.Array
    target_pos: jax.Array
    target_priority: jax.Array
    target_active: jax.Array
    cluster_access: jax.Array  # (T + H + 1, N, C) bool, indexed by absolute step
    team_future: jax.Array  # (T + H + 1, C) accesses by any satellite from that step on
    # Geometry at ``step + 1``, shared by the observation and the transition that follows it
    # (the single most expensive computation; see ``refresh``).
    body: jax.Array  # (N, M, 3) line of sight in body axes [km]
    visible: jax.Array  # (N, M) bool


class CoopObservation(NamedTuple):
    agents_view: jax.Array  # (N, obs_dim)
    action_mask: jax.Array  # (N, action_dim)
    step_count: jax.Array  # (N,)
    global_state: jax.Array  # (N, global_dim)


def _squash(value: jax.Array, scale: float = 50.0) -> jax.Array:
    """Map a non-negative count-like quantity to ``[0, 1]`` on a log scale."""
    return jnp.log1p(value) / jnp.log1p(scale)


class CoopSarSat(SarSat):
    """SarSat with beam-value slots, cluster look-ahead and a compact global state."""

    def __init__(
        self,
        *args,
        num_slots: int = 4,
        num_candidates: int = 64,
        horizon: int = 16,
        ranked_contention: bool = False,
        **kwargs,
    ) -> None:
        self.ranked_contention = ranked_contention
        self.num_slots = num_slots
        self.num_candidates = num_candidates
        self.horizon = horizon
        super().__init__(*args, **kwargs)
        if not self.side_switch:
            raise ValueError("CoopSarSat assumes the side_switch action encoding")
        self.num_candidates = min(num_candidates, self.max_targets)
        self._num_hot = self.hotspots * self.hotspot_targets
        self._cluster_id = jnp.arange(self._num_hot) % self.num_clusters

    # The specs are built inside ``SarSat.__init__``, so the sizes are derived on demand.
    @property
    def num_clusters(self) -> int:
        return max(self.hotspots, 1)

    @property
    def obs_dim(self) -> int:
        return self.slot_dim * self.num_slots + 2 * self.horizon + SELF_DIM

    @obs_dim.setter
    def obs_dim(self, value: int) -> None:
        del value  # the base class assigns its own layout's size; ours is fixed above

    @property
    def global_extra_dim(self) -> int:
        return 8 + self.num_clusters + self.horizon

    @property
    def global_dim(self) -> int:
        return self.obs_dim + self.global_extra_dim

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #

    def reset_state(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> CoopState:
        """A fresh state whose geometry cache is still empty: call :meth:`refresh` next."""
        base = self.base_state(key, num_targets)
        steps = jnp.arange(self.schedule_length)
        return self.assemble(base, self.cluster_schedule(base, steps))

    @property
    def schedule_length(self) -> int:
        return self.time_limit + self.horizon + 1

    def base_state(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> State:
        """``SarSat.reset`` without its observation: orbits, targets and a full battery."""
        return self._initial_state(key, num_targets)

    # Extension points (sarsat.windows uses them); the defaults change nothing.
    state_cls = CoopState
    slot_dim = SLOT_DIM

    def _schedule_mask(self, base: State, steps: jax.Array) -> Optional[jax.Array]:
        """``(len(steps), C)`` mask on cluster access, or ``None`` for no mask."""
        return None

    def _extra_slot_features(self, state, slot_target: jax.Array, t1: jax.Array) -> list:
        """Further ``(N, K, 1)`` slot features appended after the standard eight."""
        return []

    def cluster_schedule(self, base: State, steps: jax.Array) -> jax.Array:
        """``(len(steps), N, C)`` access of every satellite to every cluster centre. This is
        the costly part of a reset (as much geometry as one pass over all targets), so it can
        be computed a few steps at a time (``mapx.wrappers.sarsat_coop`` does)."""
        if not self.hotspots:
            return jnp.zeros((steps.shape[0], self.num_agents, 1), bool)
        hot_pos = base.target_pos[: self._num_hot]
        centre = jax.ops.segment_sum(hot_pos, self._cluster_id, self.num_clusters)
        centre = EARTH_RADIUS_KM * centre / jnp.linalg.norm(centre, axis=-1, keepdims=True)
        access = jax.vmap(lambda step: self._point_access(base.orbit, centre, step))(steps)
        mask = self._schedule_mask(base, steps)
        return access if mask is None else access & mask[:, None, :]

    def assemble(self, base: State, access: jax.Array) -> CoopState:
        total = self.schedule_length
        # Accesses after the time limit are worth nothing to the team.
        counted = access & (jnp.arange(total) <= self.time_limit)[:, None, None]
        per_step = jnp.sum(counted, axis=1).astype(jnp.float32)  # (total, C)
        team_future = jnp.cumsum(per_step[::-1], axis=0)[::-1]
        n, m = self.num_agents, self.max_targets
        return self.state_cls(
            *base,
            cluster_access=access,
            team_future=team_future,
            body=jnp.zeros((n, m, 3), jnp.float32),
            visible=jnp.zeros((n, m), bool),
        )

    def refresh(self, state: CoopState) -> CoopState:
        """Fill the geometry cache for ``state.step + 1``. Must follow every change of
        ``step`` or ``target_active`` and precede :meth:`observe` / :meth:`transition`."""
        body, visible = self._target_geometry(state, state.step + 1)
        return state._replace(body=body, visible=visible)

    def reset(
        self, key: jax.Array, num_targets: Optional[Union[int, jax.Array]] = None
    ) -> Tuple[CoopState, TimeStep[CoopObservation]]:
        state = self.refresh(self.reset_state(key, num_targets))
        return state, restart(self.observe(state), shape=(self.num_agents,))

    def transition(
        self, state: CoopState, action: jax.Array
    ) -> Tuple[CoopState, jax.Array, jax.Array]:
        """The dynamics of :meth:`SarSat.step` without the observation: next state, team
        reward and whether every target has been imaged."""
        return self.transition_with_credit(state, action)[:3]

    def transition_with_credit(
        self, state: CoopState, action: jax.Array
    ) -> Tuple[CoopState, jax.Array, jax.Array, jax.Array]:
        """:meth:`transition` plus the ``(N,)`` share of the reward each satellite earned."""
        step = state.step + 1
        # SarSat._apply_action on the cached geometry.
        pointing = self._decode_pointing(state.orbit, action[:, : self._pointing_dim])
        sensing = (action[:, self._pointing_dim] > 0.5) & self._can_sense(state.battery)
        lower = (pointing - self._beam_half_width)[:, None, :]
        upper = (pointing + self._beam_half_width)[:, None, :]
        in_beam = state.visible & sensing[:, None] & in_az_el_box(state.body, lower, upper)
        imaged = jnp.any(in_beam, axis=0)
        reward = jnp.sum(jnp.where(imaged, state.target_priority, 0.0))
        # Who earned it: a target caught by several beams at once is split evenly. Sums to
        # ``reward``; exposed for credit-assignment experiments, the team reward is unchanged.
        share = state.target_priority / jnp.maximum(jnp.sum(in_beam, axis=0), 1)
        credit = in_beam.astype(jnp.float32) @ share  # (N,)
        state = state._replace(
            step=step,
            battery=jnp.clip(
                state.battery + self.recharge_rate - self.sense_cost * sensing, 0.0, 1.0
            ),
            target_active=state.target_active & ~imaged,
        )
        return state, reward, ~jnp.any(state.target_active), credit

    def make_timestep(
        self, state: CoopState, reward: jax.Array, all_imaged: jax.Array, last: jax.Array
    ) -> Tuple[CoopState, TimeStep[CoopObservation]]:
        """Refresh the geometry of a post-transition (or freshly reset) state and observe."""
        n = self.num_agents
        state = self.refresh(state)
        return state, TimeStep(
            step_type=jnp.where(last, StepType.LAST, StepType.MID),
            reward=jnp.full((n,), reward, jnp.float32),
            discount=jnp.full((n,), jnp.where(all_imaged, 0.0, 1.0), jnp.float32),
            observation=self.observe(state),
            extras={},
        )

    def step(
        self, state: CoopState, action: jax.Array
    ) -> Tuple[CoopState, TimeStep[CoopObservation]]:
        state, reward, all_imaged = self.transition(state, action)
        last = all_imaged | (state.step >= self.time_limit)
        return self.make_timestep(state, reward, all_imaged, last)

    @cached_property
    def observation_spec(self) -> specs.Spec[CoopObservation]:
        n = self.num_agents
        return specs.Spec(
            CoopObservation,
            "ObservationSpec",
            agents_view=specs.Array((n, self.obs_dim), jnp.float32, "agents_view"),
            action_mask=specs.BoundedArray((n, self.action_dim), bool, False, True, "action_mask"),
            step_count=specs.BoundedArray((n,), jnp.int32, 0, self.time_limit, "step_count"),
            global_state=specs.Array((n, self.global_dim), jnp.float32, "global_state"),
        )

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _point_access(self, orbit: Orbit, points: jax.Array, step: jax.Array) -> jax.Array:
        """``(N, P)`` whether each satellite can reach each Earth-fixed point at ``step``."""
        _, frame = satellite_frames(orbit, step * self.step_seconds)
        radius = orbit.radius[:, None]
        body = jnp.einsum("mc,nbc->nmb", points, frame).at[..., 2].add(radius)
        above_horizon = body[..., 2] < radius - EARTH_RADIUS_KM**2 / radius
        low, high = look_angle_band(orbit.radius, self._incidence)
        return above_horizon & in_incidence_band(
            body, low[:, None], high[:, None], self._max_squint
        )

    def _slots(self, hit: jax.Array, weight: jax.Array, scarce_weight: jax.Array):
        """Non-maximum suppression for one satellite: indices of the ``num_slots`` best
        beams and the raw / scarcity-weighted priority each adds over the earlier ones."""
        covered = jnp.zeros(weight.shape, bool)
        hit_f = hit.astype(jnp.float32)
        index, value, scarce = [], [], []
        for _ in range(self.num_slots):
            gain = hit_f @ jnp.where(covered, 0.0, weight)
            best = jnp.argmax(gain)
            index.append(best)
            value.append(gain[best])
            scarce.append(jnp.sum(jnp.where(hit[best] & ~covered, scarce_weight, 0.0)))
            covered = covered | hit[best]
        return jnp.stack(index), jnp.stack(value), jnp.stack(scarce)

    def observe(self, state: CoopState) -> CoopObservation:
        n, k = self.num_agents, self.num_slots
        t1 = state.step + 1
        priority = state.target_priority
        unit = jnp.max(priority)  # one hotspot target (or one target, without hotspots)
        can_sense = self._can_sense(state.battery)

        # Scarcity discount per target: hotspot targets by their cluster's future team
        # accesses, background by the mean over clusters.
        team_now = state.team_future[t1]  # (C,)
        discount_c = _SCARCITY_KAPPA / (_SCARCITY_KAPPA + team_now)
        background = jnp.mean(discount_c) if self.hotspots else jnp.ones(())
        discount = jnp.full((self.max_targets,), background)
        team_m = jnp.full((self.max_targets,), jnp.mean(team_now))
        if self.hotspots:
            discount = discount.at[: self._num_hot].set(discount_c[self._cluster_id])
            team_m = team_m.at[: self._num_hot].set(team_now[self._cluster_id])

        # Candidates: the most valuable accessible targets of every satellite.
        body, visible = state.body, state.visible  # (N, M, 3), (N, M)
        # Hotspot targets occupy the first slots, so "the first ``num_candidates`` visible
        # targets in slot order" is the most valuable set. Finding them by binary search on
        # the running count is far cheaper than a top-k over all M targets.
        count = jnp.cumsum(visible, axis=1, dtype=jnp.int32)  # (N, M)
        wanted = jnp.arange(1, self.num_candidates + 1, dtype=jnp.int32)
        cand = jax.vmap(lambda row: jnp.searchsorted(row, wanted, side="left"))(count)
        cand = jnp.minimum(cand, self.max_targets - 1)  # fewer visible: padded, masked below
        cand_visible = wanted[None, :] <= count[:, -1:]  # (N, C)
        az, el = az_el(jnp.take_along_axis(body, cand[:, :, None], axis=1))
        weight = jnp.where(cand_visible, priority[cand], 0.0) / unit
        hw = self._beam_half_width
        hit = (
            (jnp.abs(az[:, :, None] - az[:, None, :]) <= hw[0])
            & (jnp.abs(el[:, :, None] - el[:, None, :]) <= hw[1])
            & cand_visible[:, :, None]
            & cand_visible[:, None, :]
        )
        slot, value, scarce = jax.vmap(self._slots)(hit, weight, weight * discount[cand])
        valid = value > 0.0
        slot_az = jnp.take_along_axis(az, slot, axis=1)
        slot_el = jnp.take_along_axis(el, slot, axis=1)
        pointing = self._encode_pointing(state.orbit, slot_az, slot_el)  # (N, K, 3)
        slot_target = jnp.take_along_axis(cand, slot, axis=1)  # (N, K)
        charged = visible & can_sense[:, None]  # (N, M)
        if self.ranked_contention:
            # Only teammates with a lower index count: a shared convention ("lower index
            # goes first") under which exactly one of the contenders sees no competition.
            ahead = jnp.cumsum(charged, axis=0, dtype=jnp.int32) - charged
            others = jnp.take_along_axis(ahead, slot_target, axis=1)
        else:
            others = jnp.sum(charged, axis=0)[slot_target] - can_sense[:, None]
        slot_features = jnp.concatenate(
            [
                jnp.ones((n, k, 1)),
                pointing,
                _squash(value)[..., None],
                _squash(scarce)[..., None],
                _squash(team_m[slot_target], 200.0)[..., None],
                jnp.clip(others / 2.0, 0.0, 1.0)[..., None],
                *self._extra_slot_features(state, slot_target, t1),
            ],
            axis=-1,
        )
        slot_features = jnp.where(valid[..., None], slot_features, 0.0).reshape(n, -1)

        # Look-ahead over the hotspot clusters.
        if self.hotspots:
            active_hot = state.target_active[: self._num_hot].astype(jnp.float32)
            remaining = (
                jax.ops.segment_sum(active_hot, self._cluster_id, self.num_clusters)
                / self.hotspot_targets
            )
        else:
            remaining = jnp.zeros((self.num_clusters,))
        window = jax.lax.dynamic_slice_in_dim(state.cluster_access, t1, self.horizon, axis=0)
        team_window = jax.lax.dynamic_slice_in_dim(state.team_future, t1, self.horizon, axis=0)
        window = window.astype(jnp.float32)  # (H, N, C)
        ahead_value = jnp.einsum("hnc,c->nh", window, remaining)
        ahead_scarce = jnp.einsum(
            "hnc,hc->nh", window, remaining * _SCARCITY_KAPPA / (_SCARCITY_KAPPA + team_window)
        )
        ahead = jnp.stack([ahead_value, ahead_scarce], axis=-1).reshape(n, -1)

        time_remaining = 1.0 - state.step / self.time_limit
        own = jnp.stack(
            [
                state.battery,
                can_sense.astype(jnp.float32),
                jnp.full((n,), time_remaining),
                _squash(jnp.sum(weight, axis=1), 100.0),
                jnp.mean(cand_visible, axis=1),
            ],
            axis=-1,
        )
        agents_view = jnp.concatenate([slot_features, ahead, own], axis=-1).astype(jnp.float32)

        # Team summary for a centralised critic.
        left = jnp.where(state.target_active, priority, 0.0)
        summary = jnp.concatenate(
            [
                jnp.stack(
                    [
                        time_remaining,
                        jnp.mean(state.battery),
                        *jnp.quantile(state.battery, jnp.asarray([0.1, 0.5, 0.9])),
                        jnp.mean(can_sense),
                        jnp.sum(left[: self._num_hot]),
                        jnp.sum(left[self._num_hot :]),
                    ]
                ),
                -jnp.sort(-remaining),
                jnp.mean(ahead_value, axis=0),
            ]
        ).astype(jnp.float32)
        global_state = jnp.concatenate(
            [agents_view, jnp.broadcast_to(summary, (n, summary.shape[0]))], axis=-1
        )

        action_mask = jnp.ones((n, self.action_dim), bool)
        action_mask = action_mask.at[:, self._pointing_dim].set(can_sense)
        return CoopObservation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=jnp.full((n,), state.step, jnp.int32),
            global_state=global_state,
        )

    _observe = observe  # SarSat.reset / step call this name
