"""A world that never ends, built on the episodic :class:`sarsat.windows.WindowedCoopSarSat`.

The environment and the policies trained on it are fixed-horizon: the actor observes the
time remaining, the cluster look-ahead is precomputed for ``time_limit + horizon + 1`` steps
and every event window closes by the time limit. :class:`LiveWorld` keeps a *continuous*
world (orbits, batteries, background targets, live requests) and runs it as **rolling
episodes**: every ``time_limit`` steps the world is re-packed into a fresh state -- orbits
re-based so step 0 of the new episode is where step ``time_limit`` of the old one left off,
batteries carried over, captured background targets respawned, a fresh draw of events with
the training rules -- and the caller resets the policy's recurrent state. Inside an episode
the policy sees exactly what it saw in training.

Live requests (:meth:`LiveWorld.inject_request`) take over an event cluster whose window has
closed: all ``hotspot_targets`` of its slots are placed at the requested point (the first
exactly there, the rest scattered by a few km) so the cluster look-ahead values it like a
training event, with a window of at most ``deadline`` steps. A request whose window would
outlive the episode is clipped at the time limit and carried into the next episode, so the
policy never sees an access past the limit (it never did in training).

DECISIONS.md issue 28 has the reasoning; ``live/`` has the server that drives this.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.evaluate import _beam_directions, _ground_point, _latlon
from sarsat.orbits import EARTH_RADIUS_KM, EARTH_ROTATION_RAD_S, in_az_el_box, satellite_frames
from sarsat.targets import latlon_to_unit_vector
from sarsat.types import Orbit
from sarsat.windows import WindowedCoopSarSat, WindowedCoopState, WindowedState

TWO_PI = 2.0 * math.pi

# How a request's priority level maps onto the field: the weight relative to one event
# target (never above 1: ``max(priority)`` scales the whole observation) and the scatter of
# the cluster around the point (smaller = one look takes all of it).
PRIORITY_WEIGHT = {"low": 0.5, "normal": 1.0, "high": 1.0}
PRIORITY_SIGMA_KM = {"low": 20.0, "normal": 20.0, "high": 12.0}
MAX_DEADLINE_STEPS = 30


@dataclass
class Request:
    """A live tasking request and its fate."""

    id: int
    lat: float
    lon: float
    priority: str
    deadline_steps: int
    submitted_global: int  # global step (episode * time_limit + step) when it arrived
    close_global: int  # last global step at which it can be imaged
    cluster: int  # cluster slot it occupies in the current episode
    text: str = ""
    place: str = ""
    status: str = "pending"  # pending | collected | missed
    collected_global: Optional[int] = None
    collected_by: Optional[int] = None
    feasible_count: int = 0
    first_access_global: Optional[int] = None
    feasible: List[Tuple[int, int]] = field(default_factory=list)  # (satellite, global step)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventRecord:
    """An event cluster whose window has closed, with how much of it was imaged."""

    episode: int
    cluster: int
    open: int
    close: int
    imaged_fraction: float


@dataclass
class StepResult:
    """What one step produced, in numpy, for logging and streaming."""

    episode: int
    step: int  # in-episode step after the transition (1..time_limit)
    global_step: int
    reward: float
    battery: np.ndarray  # (N,)
    sensing: np.ndarray  # (N,) bool
    sat_latlon: np.ndarray  # (N, 2) deg
    captures: np.ndarray  # (K, 2) int: target slot, satellite
    footprints: Dict[int, np.ndarray]  # satellite -> (4, 2) lat/lon corners, sensing only
    boresight: Dict[int, np.ndarray]  # satellite -> (2,) lat/lon
    events_opened: List[int]
    events_closed: List[EventRecord]
    request_updates: List[Request]
    new_episode: bool  # the step ended an episode; the caller must reset its policy


def rebase_orbit(orbit: Orbit, time_s: float) -> Orbit:
    """The same orbits with their epoch moved forward by ``time_s`` seconds."""
    return orbit._replace(
        phase=(orbit.phase + orbit.mean_motion * time_s) % TWO_PI,
        raan=(orbit.raan - EARTH_ROTATION_RAD_S * time_s) % TWO_PI,
    )


class LiveWorld:
    """Rolling episodes of ``env`` with live requests. Not thread-safe: one driver."""

    def __init__(
        self,
        env: WindowedCoopSarSat,
        seed: int = 0,
        schedule_rows_per_step: int = 10,
        boundary_battery: str = "carry",
    ) -> None:
        """``boundary_battery``: ``carry`` keeps each satellite's charge across the episode
        boundary (the honest continuous world; the trained policy ends episodes nearly empty
        at a 5% duty cycle, which costs the next episode a few percent), ``full`` recharges
        every satellite at the boundary, as a training reset does."""
        if boundary_battery not in ("carry", "full"):
            raise ValueError("boundary_battery must be 'carry' or 'full'")
        self.boundary_battery = boundary_battery
        self.env = env
        self.T = int(env.time_limit)
        self.C = int(env.num_clusters)
        self.K = int(env.hotspot_targets)
        self.N = int(env.num_agents)
        self.M = int(env.max_targets)
        if not env.hotspots:
            raise ValueError("LiveWorld needs an environment with event clusters (hotspots > 0)")
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        self.rng = np.random.default_rng(seed)
        self.episode = 0
        self.global_step = 0
        self.requests: List[Request] = []
        self.events: List[EventRecord] = []
        self.episode_returns: List[float] = []
        self._episode_return = 0.0
        self._cluster_request: Dict[int, int] = {}  # cluster -> pending request id

        self._rows = int(schedule_rows_per_step)
        total = env.schedule_length
        self._prebuild_start = max(1, self.T - math.ceil(total / self._rows) - 1)
        self._next_base: Optional[WindowedState] = None
        self._next_rows: List[jax.Array] = []

        hw = np.asarray(env._beam_half_width) * 180 / np.pi
        self._corner_offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * hw

        self._jit_reset = jax.jit(env.reset)
        self._jit_advance = jax.jit(self._advance)
        self._jit_build_base = jax.jit(self._build_next_base)
        self._jit_rows = jax.jit(self._schedule_rows)
        self._jit_finish = jax.jit(self._finish_next)
        self._jit_inject = jax.jit(self._inject)
        self._jit_refresh_observe = jax.jit(self._refresh_observe)
        self._jit_observe = jax.jit(env.observe)
        self._jit_access = jax.jit(self._point_access_steps)

        # Episode 0 is exactly ``env.reset(PRNGKey(seed))``, as the offline evaluations are.
        self.state, timestep = self._jit_reset(self.key)
        self.obs = timestep.observation
        self._windows = np.array(self.state.target_window[: self.C])

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #

    def _refresh_observe(self, state):
        state = self.env.refresh(state)
        return state, self.env.observe(state)

    def _advance(self, state: WindowedCoopState, action: jax.Array):
        env = self.env
        pointing = env._decode_pointing(state.orbit, action[:, : env._pointing_dim])
        sensing = (action[:, env._pointing_dim] > 0.5) & env._can_sense(state.battery)
        lower = (pointing - env._beam_half_width)[:, None, :]
        upper = (pointing + env._beam_half_width)[:, None, :]
        in_beam = state.visible & sensing[:, None] & in_az_el_box(state.body, lower, upper)
        state, reward, _, _ = env.transition_with_credit(state, action)
        state = env.refresh(state)
        obs = env.observe(state)
        position, frame = satellite_frames(state.orbit, state.step * env.step_seconds)
        return state, obs, reward, sensing, in_beam, position, frame, pointing

    def step(self, action: jax.Array) -> StepResult:
        """Apply ``action`` ``(N, 4)``; rolls the episode over when it ends."""
        out = self._jit_advance(self.state, action)
        self.state, self.obs = out[0], out[1]
        reward = float(out[2])
        sensing, in_beam = np.asarray(out[3]), np.asarray(out[4])
        position, frame = np.asarray(out[5], np.float64), np.asarray(out[6], np.float64)
        pointing = np.degrees(np.asarray(out[7]))
        step = int(self.state.step)
        self.global_step += 1
        self._episode_return += reward

        # Footprints of the sensing satellites (as sarsat.evaluate.run_episode does).
        sats = np.flatnonzero(sensing)
        angles = np.concatenate(
            [pointing[sats, None, :], pointing[sats, None, :] + self._corner_offsets], axis=1
        )
        ground = _ground_point(position[sats, None, :], _beam_directions(frame[sats], angles))
        footprints = {int(s): ground[i, 1:] for i, s in enumerate(sats)}
        boresight = {int(s): ground[i, 0] for i, s in enumerate(sats)}

        hit_sat, hit_target = np.nonzero(in_beam)
        captures = np.stack([hit_target, hit_sat], axis=1)

        active = np.asarray(self.state.target_active)
        opened = [int(c) for c in np.flatnonzero(self._windows[:, 0] == step)]
        closed = []
        for c in np.flatnonzero(self._windows[:, 1] == step):
            slots = int(c) + self.C * np.arange(self.K)
            rec = EventRecord(
                self.episode,
                int(c),
                int(self._windows[c, 0]),
                int(self._windows[c, 1]),
                float(1.0 - active[slots].mean()),
            )
            closed.append(rec)
            self.events.append(rec)

        updates = self._update_requests(in_beam, active)

        if step >= self._prebuild_start:
            self._prebuild()

        new_episode = step >= self.T
        if new_episode:
            updates += self._rollover()

        return StepResult(
            episode=self.episode - int(new_episode),
            step=step,
            global_step=self.global_step,
            reward=reward,
            battery=np.asarray(self.state.battery),
            sensing=sensing,
            sat_latlon=_latlon(position),
            captures=captures,
            footprints=footprints,
            boresight=boresight,
            events_opened=opened,
            events_closed=closed,
            request_updates=updates,
            new_episode=new_episode,
        )

    def _update_requests(self, in_beam: np.ndarray, active: np.ndarray) -> List[Request]:
        updates = []
        for r in self.requests:
            if r.status != "pending":
                continue
            slots = r.cluster + self.C * np.arange(self.K)
            hit = in_beam[:, slots]
            centre_hit = np.flatnonzero(hit[:, 0])
            if centre_hit.size or (1.0 - active[slots].mean()) >= 0.5:
                r.status = "collected"
                r.collected_global = self.global_step
                by = centre_hit if centre_hit.size else np.flatnonzero(hit.any(axis=1))
                r.collected_by = int(by[0]) if by.size else None
            elif self.global_step >= r.close_global:
                r.status = "missed"
            else:
                continue
            self._cluster_request.pop(r.cluster, None)
            updates.append(r)
        return updates

    # ------------------------------------------------------------------ #
    # Rolling over
    # ------------------------------------------------------------------ #

    def _build_next_base(self, state: WindowedCoopState, key: jax.Array) -> WindowedState:
        """Next episode's base: fresh events and windows from the environment's own sampler
        (``fold_in`` keys, so the draw is reproducible), the orbits continued, batteries and
        background targets as they are (``_finish_next`` patches those at the boundary)."""
        env = self.env
        fresh = env.base_state(key)
        orbit = rebase_orbit(state.orbit, self.T * env.step_seconds)
        return WindowedState(
            key=fresh.key,
            step=jnp.zeros((), jnp.int32),
            orbit=orbit,
            battery=state.battery,
            target_latlon=fresh.target_latlon,
            target_pos=fresh.target_pos,
            target_priority=fresh.target_priority,
            target_active=fresh.target_active,
            target_window=fresh.target_window,
        )

    def _schedule_rows(self, base: WindowedState, start: jax.Array) -> jax.Array:
        steps = start + jnp.arange(self._rows)
        return self.env.cluster_schedule(base, steps)

    def _finish_next(
        self, base: WindowedState, access: jax.Array, state: WindowedCoopState
    ) -> WindowedCoopState:
        """Carry the batteries and the surviving background targets over, then assemble."""
        env = self.env
        is_background = jnp.arange(self.M) >= self.C * self.K
        keep = is_background & state.target_active
        latlon = jnp.where(keep[:, None], state.target_latlon, base.target_latlon)
        battery = (
            state.battery if self.boundary_battery == "carry" else jnp.ones_like(state.battery)
        )
        base = base._replace(
            battery=battery,
            target_latlon=latlon,
            target_pos=EARTH_RADIUS_KM * latlon_to_unit_vector(latlon),
        )
        return env.assemble(base, access[: env.schedule_length])

    def _prebuild(self) -> None:
        if self._next_base is None:
            self._next_base = self._jit_build_base(
                self.state, jax.random.fold_in(self.key, self.episode + 1)
            )
            self._next_rows = []
        done = len(self._next_rows) * self._rows
        if done < self.env.schedule_length:
            self._next_rows.append(self._jit_rows(self._next_base, jnp.int32(done)))

    def _rollover(self) -> List[Request]:
        total = self.env.schedule_length
        while self._next_base is None or len(self._next_rows) * self._rows < total:
            self._prebuild()
        access = jnp.concatenate(self._next_rows, axis=0)
        state = self._jit_finish(self._next_base, access, self.state)
        self._next_base, self._next_rows = None, []
        self.episode_returns.append(self._episode_return)
        self._episode_return = 0.0
        self.episode += 1
        self._windows = np.array(state.target_window[: self.C])
        self._cluster_request = {}

        # Requests still open at the boundary continue in the new episode.
        updates = []
        for r in self.requests:
            if r.status != "pending":
                continue
            remainder = min(r.close_global - self.episode * self.T, self.T)
            c = self._choose_cluster(0)
            latlon, weight = self._request_field(r, state)
            window = jnp.array([1, remainder], jnp.int32)
            state = self._jit_inject(state, jnp.int32(c), latlon, weight, window)
            self._windows[c] = (1, remainder)
            r.cluster = c
            self._cluster_request[c] = r.id
            updates.append(r)
        self.state, self.obs = self._jit_refresh_observe(state)
        return updates

    # ------------------------------------------------------------------ #
    # Requests
    # ------------------------------------------------------------------ #

    def _choose_cluster(self, step: int) -> int:
        """A cluster the policy can lose nothing on: closed longest ago, else opening last."""
        taken = set(self._cluster_request)
        closed = [c for c in range(self.C) if self._windows[c, 1] < step + 1 and c not in taken]
        if closed:
            return min(closed, key=lambda c: self._windows[c, 1])
        unopened = [c for c in range(self.C) if self._windows[c, 0] > step + 1 and c not in taken]
        if unopened:
            return max(unopened, key=lambda c: self._windows[c, 0])
        raise RuntimeError("every event cluster is open or holds a request")

    def _request_field(self, r: Request, state) -> Tuple[jax.Array, jax.Array]:
        """The ``(K, 2)`` points and ``(K,)`` priorities that stand in for a request."""
        sigma_deg = PRIORITY_SIGMA_KM[r.priority] / 111.0
        scatter = self.rng.normal(size=(self.K, 2)) * sigma_deg
        scatter[0] = 0.0
        lat = np.clip(r.lat + scatter[:, 0], -89.0, 89.0)
        lon = r.lon + scatter[:, 1] / np.cos(np.deg2rad(lat))
        lon = (lon + 180.0) % 360.0 - 180.0
        unit = float(jnp.max(state.target_priority))
        weight = np.full((self.K,), PRIORITY_WEIGHT[r.priority] * unit, np.float32)
        return jnp.asarray(np.stack([lat, lon], axis=1), jnp.float32), jnp.asarray(weight)

    def _inject(
        self,
        state: WindowedCoopState,
        c: jax.Array,
        latlon: jax.Array,
        priority: jax.Array,
        window: jax.Array,
    ) -> WindowedCoopState:
        """Cluster ``c`` becomes ``latlon`` with ``priority`` and ``window``; its column of
        the cluster schedule and the team's future-access counts follow (``coop.assemble``)."""
        env = self.env
        idx = c + self.C * jnp.arange(self.K)
        pos = EARTH_RADIUS_KM * latlon_to_unit_vector(latlon)
        centre = jnp.mean(pos, axis=0)
        centre = EARTH_RADIUS_KM * centre / jnp.linalg.norm(centre)
        steps = jnp.arange(env.schedule_length)
        access_c = jax.vmap(lambda s: env._point_access(state.orbit, centre[None], s))(steps)
        access_c = access_c[..., 0] & ((window[0] <= steps) & (steps <= window[1]))[:, None]
        access = state.cluster_access.at[:, :, c].set(access_c)
        counted = access & (steps <= self.T)[:, None, None]
        per_step = jnp.sum(counted, axis=1).astype(jnp.float32)
        team_future = jnp.cumsum(per_step[::-1], axis=0)[::-1]
        state = state._replace(
            target_latlon=state.target_latlon.at[idx].set(latlon),
            target_pos=state.target_pos.at[idx].set(pos),
            target_priority=state.target_priority.at[idx].set(priority),
            target_active=state.target_active.at[idx].set(True),
            target_window=state.target_window.at[idx].set(window),
            cluster_access=access,
            team_future=team_future,
        )
        return env.refresh(state)

    def inject_request(
        self,
        lat: float,
        lon: float,
        priority: str = "normal",
        deadline_steps: int = MAX_DEADLINE_STEPS,
        text: str = "",
        place: str = "",
    ) -> Request:
        """Add a request at ``(lat, lon)`` to be imaged within ``deadline_steps`` steps."""
        if priority not in PRIORITY_WEIGHT:
            raise ValueError(f"priority must be one of {sorted(PRIORITY_WEIGHT)}")
        deadline = int(max(1, min(int(deadline_steps), MAX_DEADLINE_STEPS)))
        step = int(self.state.step)
        c = self._choose_cluster(step)
        r = Request(
            id=len(self.requests) + 1,
            lat=float(lat),
            lon=float(lon),
            priority=priority,
            deadline_steps=deadline,
            submitted_global=self.global_step,
            close_global=self.global_step + deadline,
            cluster=c,
            text=text,
            place=place,
        )
        feas = self.feasibility(lat, lon, deadline)
        r.feasible = feas["satellites"]
        r.feasible_count = len(feas["satellites"])
        r.first_access_global = feas["first_global"]
        close = min(step + deadline, self.T)
        latlon, weight = self._request_field(r, self.state)
        window = jnp.array([step + 1, close], jnp.int32)
        state = self._jit_inject(self.state, jnp.int32(c), latlon, weight, window)
        self.state, self.obs = state, self._jit_observe(state)
        self._windows[c] = (step + 1, close)
        self._cluster_request[c] = r.id
        self.requests.append(r)
        return r

    def _point_access_steps(self, orbit: Orbit, point: jax.Array, steps: jax.Array) -> jax.Array:
        return jax.vmap(lambda s: self.env._point_access(orbit, point[None], s))(steps)[..., 0]

    def feasibility(
        self, lat: float, lon: float, horizon: int = MAX_DEADLINE_STEPS
    ) -> Dict[str, Any]:
        """Which satellites can image ``(lat, lon)`` in the next ``horizon`` steps, and when,
        on geometry and a battery forecast alone (whether the policy takes it is its call)."""
        env = self.env
        horizon = int(max(1, min(int(horizon), MAX_DEADLINE_STEPS)))
        step = int(self.state.step)
        steps = step + 1 + jnp.arange(horizon)
        point = EARTH_RADIUS_KM * latlon_to_unit_vector(jnp.asarray([lat, lon], jnp.float32))
        access = np.asarray(self._jit_access(self.state.orbit, point, steps))  # (H, N)
        battery = np.asarray(self.state.battery)
        k = np.arange(1, horizon + 1)[:, None]
        charged = battery[None, :] + env.recharge_rate * k >= env.sense_cost - 1e-6
        ok = access & charged
        sats = []
        for n in np.flatnonzero(ok.any(axis=0)):
            first = int(np.argmax(ok[:, n]))
            sats.append((int(n), self.global_step + 1 + first))
        sats.sort(key=lambda x: x[1])
        return {
            "satellites": sats,
            "count": len(sats),
            "first_global": sats[0][1] if sats else None,
            "first_in_steps": (sats[0][1] - self.global_step) if sats else None,
        }

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #

    def target_table(self) -> Dict[str, np.ndarray]:
        """The current field: ``latlon (M, 2)``, ``cluster (M,)`` (-1 background),
        ``weight (M,)`` in event-target units, ``window (M, 2)``, ``active (M,)``."""
        s = self.state
        priority = np.asarray(s.target_priority)
        unit = priority.max() if priority.size else 1.0
        slot = np.arange(self.M)
        cluster = np.where(slot < self.C * self.K, slot % self.C, -1)
        return {
            "latlon": np.asarray(s.target_latlon),
            "cluster": cluster,
            "weight": priority / unit,
            "window": np.asarray(s.target_window),
            "active": np.asarray(s.target_active),
        }

    def cluster_windows(self) -> np.ndarray:
        return self._windows.copy()

    @property
    def pending_requests(self) -> List[Request]:
        return [r for r in self.requests if r.status == "pending"]
