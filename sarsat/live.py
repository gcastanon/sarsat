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
closed: all ``hotspot_targets`` of its slots are placed at the requested point so the cluster
look-ahead values it like a training event, with a window of at most ``deadline`` steps. By
default every slot sits exactly on the point, so one look collects the request and the field
is a pure function of the request (no random draw: a forecast of the request and its
submission build the same field); ``request_scatter=1.0`` restores the former spread of the
other slots by a few km (DECISIONS.md issue 39). A request whose window would
outlive the episode is clipped at the time limit and carried into the next episode, so the
policy never sees an access past the limit (it never did in training).

DECISIONS.md issue 28 has the reasoning; ``live/`` has the server that drives this.

With collection modes and quality requirements (``sarsat.quality.WindowedQualityCoopSarSat``,
DECISIONS.md issues 31-36) the same world runs unchanged: each look's footprint is drawn with
its own mode's beam and :attr:`StepResult.modes` says which mode it was, a request can carry the
image quality it needs (``inject_request(..., quality=...)``), and background targets keep
their requirement across the episode boundary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.evaluate import beam_directions, ecef_to_latlon, ray_ground_point
from sarsat.forecast import Forecast, advance_fn, diff, forecast
from sarsat.orbits import EARTH_RADIUS_KM, EARTH_ROTATION_RAD_S, satellite_frames
from sarsat.targets import latlon_to_unit_vector
from sarsat.types import Orbit
from sarsat.windows import WindowedCoopSarSat, WindowedCoopState, WindowedState

TWO_PI = 2.0 * math.pi

# How a request's priority level maps onto the field: the weight relative to one event
# target (never above 1: ``max(priority)`` scales the whole observation) and, scaled by
# ``LiveWorld.request_scatter`` (0 by default: no spread), the one-sigma scatter of
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
    status: str = "pending"  # provisional | pending | collected | missed
    quality: Optional[float] = None  # image quality required (quality envs); None: any look
    collected_global: Optional[int] = None
    collected_by: Optional[int] = None
    feasible_count: int = 0
    first_access_global: Optional[int] = None
    feasible: List[Tuple[int, int]] = field(default_factory=list)  # (satellite, global step)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Preview:
    """A provisional request (``status == "provisional"``) with the world state it would
    produce: what a forecast of the request runs on, and what :meth:`LiveWorld.commit_request`
    puts into the world. Built by :meth:`LiveWorld.preview_request`."""

    request: Request
    state: Any
    obs: Any
    window: Tuple[int, int]  # in-episode window the request's cluster was given
    global_step: int  # the world's global step when it was built; stale once that moves


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
    modes: Dict[int, int] = field(default_factory=dict)  # satellite -> mode index, sensing only


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
        request_scatter: float = 0.0,
    ) -> None:
        """``boundary_battery``: ``carry`` keeps each satellite's charge across the episode
        boundary (the honest continuous world; the trained policy ends episodes nearly empty
        at a 5% duty cycle, which costs the next episode a few percent), ``full`` recharges
        every satellite at the boundary, as a training reset does.

        ``request_scatter`` scales ``PRIORITY_SIGMA_KM``, the spread of a request's slots
        around its point: 0 (the default) puts every slot on the point, so the field is
        deterministic and one look collects the request; 1 is the former 12-20 km spread."""
        if boundary_battery not in ("carry", "full"):
            raise ValueError("boundary_battery must be 'carry' or 'full'")
        if request_scatter < 0:
            raise ValueError("request_scatter must be >= 0")
        self.boundary_battery = boundary_battery
        self.request_scatter = float(request_scatter)
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
        # Start prebuilding early enough that the next episode's full schedule
        # (``total`` rows, ``_rows`` at a time) is ready by the boundary at step ``T``.
        self._prebuild_start = max(1, self.T - math.ceil(total / self._rows) - 1)
        self._next_base: Optional[WindowedState] = None
        self._next_rows: List[jax.Array] = []

        # Footprint corners per mode, (modes, 4, 2) degrees; one row without modes.
        hw = np.asarray(env._mode_beam) * 180 / np.pi
        self._corner_offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])[None] * hw[:, None]
        self._has_quality = "target_quality" in env.state_cls._fields

        self._jit_reset = jax.jit(env.reset)
        self._jit_advance = jax.jit(self._advance)
        self._jit_build_base = jax.jit(self._build_next_base)
        self._jit_rows = jax.jit(self._schedule_rows)
        self._jit_finish = jax.jit(self._finish_next)
        self._jit_inject = jax.jit(self._inject)
        self._jit_refresh_observe = jax.jit(self._refresh_observe)
        self._jit_observe = jax.jit(env.observe)
        self._jit_access = jax.jit(self._point_access_steps)
        self._jit_forecast_advance = None  # compiled on the first forecast

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
        """The transition plus the diagnostics ``step`` needs for logging: which beams
        fired and where, and the geometry to draw them (as ``sarsat.evaluate.run_episode``
        recomputes from ``_apply_action`` inside its own jitted step)."""
        env = self.env
        pointing, mode, sensing, in_beam = env.resolve_look(
            state, action, state.body, state.visible
        )
        state, reward, _, _ = env.transition_with_credit(state, action)
        state = env.refresh(state)
        obs = env.observe(state)
        position, frame = satellite_frames(state.orbit, state.step * env.step_seconds)
        return state, obs, reward, sensing, in_beam, position, frame, pointing, mode

    def step(self, action: jax.Array) -> StepResult:
        """Apply ``action`` ``(N, 4)``; rolls the episode over when it ends."""
        out = self._jit_advance(self.state, action)
        self.state, self.obs = out[0], out[1]
        reward = float(out[2])
        sensing, in_beam = np.asarray(out[3]), np.asarray(out[4])
        position, frame = np.asarray(out[5], np.float64), np.asarray(out[6], np.float64)
        pointing = np.degrees(np.asarray(out[7]))
        mode = np.asarray(out[8])
        step = int(self.state.step)
        self.global_step += 1
        self._episode_return += reward

        # Footprints of the sensing satellites (as sarsat.evaluate.run_episode does), each
        # with its own mode's beam.
        sats = np.flatnonzero(sensing)
        angles = np.concatenate(
            [pointing[sats, None, :], pointing[sats, None, :] + self._corner_offsets[mode[sats]]],
            axis=1,
        )
        ground = ray_ground_point(position[sats, None, :], beam_directions(frame[sats], angles))
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
            sat_latlon=ecef_to_latlon(position),
            captures=captures,
            footprints=footprints,
            boresight=boresight,
            events_opened=opened,
            events_closed=closed,
            request_updates=updates,
            new_episode=new_episode,
            modes={int(s): int(mode[s]) for s in sats},
        )

    def _update_requests(self, in_beam: np.ndarray, active: np.ndarray) -> List[Request]:
        """Resolve every pending request against this step's captures: collected if its
        centre target (or, failing that, most of its cluster) was imaged, missed once its
        deadline has passed, otherwise still pending."""
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
        fresh = env.base_state(key)  # WindowedState, or its quality variant
        orbit = rebase_orbit(state.orbit, self.T * env.step_seconds)
        return fresh._replace(step=jnp.zeros((), jnp.int32), orbit=orbit, battery=state.battery)

    def _schedule_rows(self, base: WindowedState, start: jax.Array) -> jax.Array:
        """One batch of ``cluster_schedule`` rows, ``self._rows`` steps starting at
        ``start``: the unit :meth:`_prebuild` accumulates across steps."""
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
        if self._has_quality:  # a surviving target keeps what it requires
            quality = jnp.where(keep, state.target_quality, base.target_quality)
            base = base._replace(target_quality=quality)
        return env.assemble(base, access[: env.schedule_length])

    def _prebuild(self) -> None:
        """Advance the next episode's schedule by one batch of rows, sampling its base
        state (fresh events and windows) on the first call. Spreads the cost of building
        a full episode's cluster-access schedule over the steps leading up to the
        boundary instead of paying for it in one step (see :meth:`_rollover`)."""
        if self._next_base is None:
            self._next_base = self._jit_build_base(
                self.state, jax.random.fold_in(self.key, self.episode + 1)
            )
            self._next_rows = []
        done = len(self._next_rows) * self._rows
        if done < self.env.schedule_length:
            self._next_rows.append(self._jit_rows(self._next_base, jnp.int32(done)))

    def _rollover(self) -> List[Request]:
        """End the episode and start the next one: finish the prebuilt schedule if it
        fell behind, assemble the new state, and carry any still-pending request into a
        cluster of the fresh field."""
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
            quality = jnp.float32(r.quality or 0.0)
            state = self._jit_inject(state, jnp.int32(c), latlon, weight, window, quality)
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
        sigma_deg = self.request_scatter * PRIORITY_SIGMA_KM[r.priority] / 111.0
        if sigma_deg > 0.0:
            scatter = self.rng.normal(size=(self.K, 2)) * sigma_deg
            scatter[0] = 0.0
        else:
            scatter = np.zeros((self.K, 2))  # no draw: the field is a function of the request
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
        quality: jax.Array = 0.0,
    ) -> WindowedCoopState:
        """Cluster ``c`` becomes ``latlon`` with ``priority``, ``window`` and, in a quality
        environment, the required ``quality``; its column of the cluster schedule and the
        team's future-access counts follow (``coop.assemble``)."""
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
        if self._has_quality:
            state = state._replace(target_quality=state.target_quality.at[idx].set(quality))
        return env.refresh(state)

    def preview_request(
        self,
        lat: float,
        lon: float,
        priority: str = "normal",
        deadline_steps: int = MAX_DEADLINE_STEPS,
        text: str = "",
        place: str = "",
        quality: Optional[float] = None,
    ) -> Preview:
        """Build the request and the world state it would produce, without changing the
        world: the provisional request a forecast can be run on (:meth:`forecast` with
        ``preview.state`` / ``preview.obs``). :meth:`commit_request` makes it real; the
        field is deterministic, so the committed world is exactly the previewed one."""
        if priority not in PRIORITY_WEIGHT:
            raise ValueError(f"priority must be one of {sorted(PRIORITY_WEIGHT)}")
        if quality is not None and not self._has_quality:
            raise ValueError("this environment has no quality requirements")
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
            status="provisional",
            quality=None if quality is None else float(quality),
        )
        feas = self.feasibility(lat, lon, deadline)
        r.feasible = feas["satellites"]
        r.feasible_count = len(feas["satellites"])
        r.first_access_global = feas["first_global"]
        close = min(step + deadline, self.T)
        latlon, weight = self._request_field(r, self.state)
        window = jnp.array([step + 1, close], jnp.int32)
        quality = jnp.float32(r.quality or 0.0)
        state = self._jit_inject(self.state, jnp.int32(c), latlon, weight, window, quality)
        return Preview(r, state, self._jit_observe(state), (step + 1, close), self.global_step)

    def commit_request(self, preview: Preview) -> Request:
        """Put a previewed request into the world. The world must not have moved since the
        preview (``global_step``), or the window and cluster it was built for are stale."""
        if preview.global_step != self.global_step:
            raise ValueError(
                f"preview built at global step {preview.global_step}, world is at "
                f"{self.global_step}: preview it again"
            )
        r = preview.request
        if r.status != "provisional":
            raise ValueError("this preview was already committed")
        r.status = "pending"
        self.state, self.obs = preview.state, preview.obs
        self._windows[r.cluster] = preview.window
        self._cluster_request[r.cluster] = r.id
        self.requests.append(r)
        return r

    def inject_request(
        self,
        lat: float,
        lon: float,
        priority: str = "normal",
        deadline_steps: int = MAX_DEADLINE_STEPS,
        text: str = "",
        place: str = "",
        quality: Optional[float] = None,
    ) -> Request:
        """Add a request at ``(lat, lon)`` to be imaged within ``deadline_steps`` steps, in a
        quality environment by a look of at least ``quality`` (``None``: any mode will do).
        :meth:`preview_request` followed by :meth:`commit_request`."""
        return self.commit_request(
            self.preview_request(lat, lon, priority, deadline_steps, text, place, quality)
        )

    # ------------------------------------------------------------------ #
    # Forecasting (sarsat.forecast; DECISIONS.md issue 40)
    # ------------------------------------------------------------------ #

    def forecast(
        self,
        policy: Any,
        preview: Optional[Preview] = None,
        horizon: Optional[int] = None,
    ) -> Forecast:
        """What ``policy`` collects from here to the end of the episode if nothing else
        changes; with ``preview``, from the world as it would be with that request in it.
        Neither the world nor the policy is changed (``sarsat.forecast.preserved``)."""
        if self._jit_forecast_advance is None:
            self._jit_forecast_advance = advance_fn(self.env)
        state, obs = (self.state, self.obs) if preview is None else (preview.state, preview.obs)
        return forecast(self.env, state, obs, policy, horizon, self._jit_forecast_advance)

    def schedule(self, fc: Forecast, preview: Optional[Preview] = None) -> Dict[str, Any]:
        """The forecast read as a schedule: every capture with what it is (a request, an
        event cluster, a background target), where and by whom; the fate of every pending
        request (and of ``preview``'s); and each event cluster's imaged fraction.

        Request fates follow :meth:`_update_requests`: collected at the first step its
        centre slot is imaged, or at which half of its cluster has been; missed when its
        window closes inside the forecast without that; ``open`` when the window outlives
        the episode (the next episode decides). Steps are global."""
        s = self.state if preview is None else preview.state
        latlon = np.asarray(s.target_latlon)
        active0 = np.asarray(s.target_active)
        priority = np.asarray(s.target_priority)
        unit = float(priority.max()) if priority.size else 1.0
        quality = np.asarray(s.target_quality) if self._has_quality else None
        offset = self.global_step - int(s.step)  # global = in-episode + offset
        requests = list(self.pending_requests)
        if preview is not None:
            requests.append(preview.request)
        by_cluster = {r.cluster: r for r in requests}
        mode_names = [m.name for m in self.env.modes] if self.env._mode_flags else None

        rows = []
        for (step, slot, sat), mode in zip(fc.captures, fc.capture_mode, strict=True):
            slot, step, sat, mode = int(slot), int(step), int(sat), int(mode)
            cluster = slot % self.C if slot < self.C * self.K else -1
            req = by_cluster.get(cluster)
            row = {
                "step": step,
                "global_step": step + offset,
                "slot": slot,
                "kind": "request" if req else ("event" if cluster >= 0 else "background"),
                "cluster": cluster if cluster >= 0 else None,
                "request": req.id if req else None,
                "lat": round(float(latlon[slot, 0]), 4),
                "lon": round(float(latlon[slot, 1]), 4),
                "weight": round(float(priority[slot] / unit), 4),
                "satellite": sat,
                "mode": mode if mode >= 0 else None,
                "mode_name": mode_names[mode] if mode_names and mode >= 0 else None,
            }
            if quality is not None:
                row["quality"] = round(float(quality[slot]), 2)
            rows.append(row)

        # Per-cluster capture history: slots imaged before the forecast, and in it by step.
        first = fc.first_capture()
        cluster_caps: Dict[
            int, List[Tuple[int, int, int, int]]
        ] = {}  # c -> (step, slot, sat, mode)
        for slot, (step, sat, mode) in first.items():
            if slot < self.C * self.K:
                cluster_caps.setdefault(slot % self.C, []).append((step, slot, sat, mode))
        for caps in cluster_caps.values():
            caps.sort()

        fates = []
        for r in requests:
            c = r.cluster
            slots = c + self.C * np.arange(self.K)
            inactive = int((~active0[slots]).sum())
            caps = cluster_caps.get(c, [])
            fate: Dict[str, Any] = {
                "request": r.id,
                "status": r.status,
                "cluster": c,
                "close_global": r.close_global,
                "collected": False,
                "step": None,
                "global_step": None,
                "by": None,
                "mode": None,
                "mode_name": None,
                "outcome": "open",
            }
            for step, slot, sat, mode in caps:
                inactive += 1
                if slot == c or inactive >= self.K / 2:
                    fate.update(
                        collected=True,
                        step=step,
                        global_step=step + offset,
                        by=sat,
                        mode=mode if mode >= 0 else None,
                        mode_name=mode_names[mode] if mode_names and mode >= 0 else None,
                        outcome="collected",
                    )
                    break
            if not fate["collected"]:
                close_step = r.close_global - offset
                if close_step <= fc.end_step:
                    fate["outcome"] = "missed"
            fates.append(fate)

        events = []
        for c in range(self.C):
            if c in by_cluster:
                continue
            slots = c + self.C * np.arange(self.K)
            before = int((~active0[slots]).sum())
            caps = cluster_caps.get(c, [])
            open_, close = (int(x) for x in self._windows[c])
            events.append(
                {
                    "cluster": c,
                    "window": [open_, close],
                    "window_global": [open_ + offset, close + offset],
                    "lat": round(float(latlon[slots[0]][0]), 4),
                    "lon": round(float(latlon[slots[0]][1]), 4),
                    "imaged_before": before / self.K,
                    "imaged_forecast": len(caps) / self.K,
                    "imaged": (before + len(caps)) / self.K,
                    "first_step": caps[0][0] + offset if caps else None,
                    "last_step": caps[-1][0] + offset if caps else None,
                    "satellites": sorted({sat for _, _, sat, _ in caps}),
                }
            )

        looks = fc.sensing.sum(axis=0)
        return {
            "start_global": self.global_step,
            "start_step": fc.start_step,
            "end_step": fc.end_step,
            "episode": self.episode,
            "horizon": fc.horizon,
            "reward": float(fc.reward.sum()),
            "captures": len(rows),
            "looks": int(looks.sum()),
            "looks_by_satellite": {int(n): int(k) for n, k in enumerate(looks) if k},
            "rows": rows,
            "requests": fates,
            "events": events,
            "seconds": round(fc.seconds, 2),
        }

    def whatif(
        self, policy: Any, preview: Preview, base: Optional[Forecast] = None
    ) -> Dict[str, Any]:
        """The schedule with ``preview``'s request in the world against the schedule without
        it (``base``, forecast here when not given). ``items`` lists every request, event
        cluster and background target whose fate differs, with ``change`` ``lost`` (collected
        without the request, not with it), ``gained``, or ``shifted`` (collected in both, at
        another step, by another satellite or in another mode); ``same_set`` says whether the
        set of collected targets is unchanged, which is the operational meaning of "no
        impact" (a shift is flagged, not counted against it)."""
        base = self.forecast(policy) if base is None else base
        alt = self.forecast(policy, preview)
        base_sched = self.schedule(base)
        alt_sched = self.schedule(alt, preview)
        d = diff(base, alt)
        new_slots = set((preview.request.cluster + self.C * np.arange(self.K)).tolist())

        def describe(slot, cap):
            if cap is None:
                return None
            step, sat, mode = cap
            return {
                "step": step,
                "global_step": step + (self.global_step - int(self.state.step)),
                "by": sat,
                "mode": mode if mode >= 0 else None,
            }

        items = []
        base_fates = {f["request"]: f for f in base_sched["requests"]}
        alt_fates = {f["request"]: f for f in alt_sched["requests"]}
        for rid, bf in base_fates.items():
            af = alt_fates[rid]
            if (bf["collected"], bf["global_step"], bf["by"], bf["mode"]) != (
                af["collected"],
                af["global_step"],
                af["by"],
                af["mode"],
            ):
                change = (
                    "lost"
                    if bf["collected"] and not af["collected"]
                    else ("gained" if af["collected"] and not bf["collected"] else "shifted")
                )
                items.append(
                    {"kind": "request", "id": rid, "change": change, "base": bf, "alt": af}
                )
        base_events = {e["cluster"]: e for e in base_sched["events"]}
        alt_events = {e["cluster"]: e for e in alt_sched["events"]}
        for c in sorted(set(base_events) | set(alt_events)):
            be, ae = base_events.get(c), alt_events.get(c)
            if be is None or ae is None:
                continue  # the cluster the request takes over (it had nothing left)
            slots = set((c + self.C * np.arange(self.K)).tolist())
            lost = sorted(slots & set(d.lost))
            gained = sorted(slots & set(d.gained))
            shifted = sorted(slots & set(d.shifted))
            if lost or gained or shifted:
                change = (
                    "lost"
                    if lost and not gained
                    else ("gained" if gained and not lost else ("mixed" if lost else "shifted"))
                )
                items.append(
                    {
                        "kind": "event",
                        "id": c,
                        "change": change,
                        "base": be,
                        "alt": ae,
                        "lost": len(lost),
                        "gained": len(gained),
                        "shifted": len(shifted),
                    }
                )
        for slot in sorted((set(d.lost) | set(d.gained) | set(d.shifted)) - new_slots):
            if slot < self.C * self.K:
                continue  # covered above
            change = "lost" if slot in d.lost else ("gained" if slot in d.gained else "shifted")
            bcap = base.first_capture().get(slot)
            acap = alt.first_capture().get(slot)
            latlon = np.asarray(self.state.target_latlon)[slot]
            items.append(
                {
                    "kind": "background",
                    "id": slot,
                    "change": change,
                    "lat": round(float(latlon[0]), 4),
                    "lon": round(float(latlon[1]), 4),
                    "base": describe(slot, bcap),
                    "alt": describe(slot, acap),
                }
            )

        lost_slots = set(d.lost) - new_slots
        gained_slots = set(d.gained) - new_slots
        request_fate = next(f for f in alt_sched["requests"] if f["request"] == preview.request.id)
        return {
            "request": preview.request.to_dict(),
            "request_fate": request_fate,
            "base": base_sched,
            "alt": alt_sched,
            "items": items,
            "same_set": not lost_slots and not gained_slots,
            "lost": len(lost_slots),
            "gained": len(gained_slots),
            "shifted": len(set(d.shifted) - new_slots),
            "reward_delta": round(float(alt.reward.sum() - base.reward.sum()), 4),
        }

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
        ``weight (M,)`` in event-target units, ``window (M, 2)``, ``active (M,)``, and in a
        quality environment ``quality (M,)``."""
        s = self.state
        priority = np.asarray(s.target_priority)
        unit = priority.max() if priority.size else 1.0
        slot = np.arange(self.M)
        cluster = np.where(slot < self.C * self.K, slot % self.C, -1)
        table = {
            "latlon": np.asarray(s.target_latlon),
            "cluster": cluster,
            "weight": priority / unit,
            "window": np.asarray(s.target_window),
            "active": np.asarray(s.target_active),
        }
        if self._has_quality:
            table["quality"] = np.asarray(s.target_quality)
        return table

    def cluster_windows(self) -> np.ndarray:
        return self._windows.copy()

    @property
    def pending_requests(self) -> List[Request]:
        return [r for r in self.requests if r.status == "pending"]
