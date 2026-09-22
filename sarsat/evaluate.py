"""Seeded evaluation of a policy with CSV logs and a self-contained HTML map viewer.

Command line::

    python -m sarsat.evaluate --satellites 8 --targets 100 --seed 0 --out runs/seed0

writes ``satellites.csv``, ``looks.csv``, ``targets.csv``, ``captures.csv`` and
``episode.html`` (open it in any browser; no server needed). ``run_episode`` is the
programmatic entry point and accepts any policy ``Observation -> action``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.env import SarSat
from sarsat.orbits import EARTH_RADIUS_KM, satellite_frames
from sarsat.reference import REFERENCE_POLICIES
from sarsat.types import Observation

PurePolicy = Callable[[Observation], jax.Array]


class StatefulPolicy(Protocol):
    """A policy that carries its own state across an episode (e.g. a recurrent actor).

    Detected by duck typing (``hasattr(policy, "reset")``), not by ``isinstance``:
    :meth:`reset` is called once before the episode and :meth:`__call__` once per step,
    both un-jitted (any jitting is the policy's own responsibility), which is what makes
    stepping a hidden state between calls possible. A plain function ``Observation ->
    action`` keeps taking the jitted path.
    """

    def reset(self) -> None:
        """Prepare for a fresh episode (e.g. zero a recurrent hidden state)."""

    def __call__(self, obs: Any) -> jax.Array:
        """The action for the step about to be taken, given the current observation."""


Policy = Union[PurePolicy, StatefulPolicy, str]

# ----------------------------------------------------------------------- policies


def greedy_policy(obs: Observation) -> jax.Array:
    """Point at target slot 0 of the first look-ahead step and sense if it is accessible.

    The observation stores each slot as ``(accessible, *pointing)`` in the action's own
    encoding, so the action is the slot's pointing features followed by its flag as the
    sense bit. Works for either action layout (``side_switch`` on or off).
    """
    pointing_dim = obs.action_mask.shape[-1] - 1
    slot = obs.agents_view[:, : 1 + pointing_dim]
    return jnp.concatenate([slot[:, 1:], slot[:, :1]], axis=-1)


def random_policy(obs: Observation) -> jax.Array:
    """Uniform random pointing, always requesting to sense (seeded by the step count)."""
    n, action_dim = obs.action_mask.shape
    k_point, k_side = jax.random.split(jax.random.PRNGKey(obs.step_count[0]))
    pointing = jax.random.uniform(k_point, (n, 2), minval=-1.0)
    # Any categorical slots before ``sense`` (the side, when enabled) are fair coin flips.
    side = jax.random.bernoulli(k_side, 0.5, (n, action_dim - 3)).astype(jnp.float32)
    return jnp.concatenate([pointing, side, jnp.ones((n, 1))], axis=-1)


POLICIES: Dict[str, Policy] = {"greedy": greedy_policy, "random": random_policy}

# ------------------------------------------------------------------------ episode


@dataclass
class EpisodeLog:
    """Everything recorded during one episode. ``T`` steps, ``N`` satellites, ``M`` targets."""

    step_seconds: float
    beam_half_width_deg: Tuple[float, float]
    sat_latlon: np.ndarray  # (T+1, N, 2) [deg]
    sat_alt_km: np.ndarray  # (N,)
    battery: np.ndarray  # (T+1, N)
    pointing_deg: np.ndarray  # (T, N, 2) commanded (az, el), executed at step t+1
    sensing: np.ndarray  # (T, N) bool, look actually made at step t+1
    boresight: np.ndarray  # (T, N, 2) ground lat/lon of the beam centre, NaN if off Earth
    footprint: np.ndarray  # (T, N, 4, 2) ground lat/lon of the beam corners
    target_latlon: np.ndarray  # (M, 2)
    target_on_land: np.ndarray  # (M,) bool
    target_priority: np.ndarray  # (M,)
    captures: np.ndarray  # (C, 3) rows of (step, target, satellite), step in 1..T
    reward: np.ndarray  # (T,)
    target_window: np.ndarray | None = None  # (M, 2) int first/last step a target can be imaged

    @property
    def num_steps(self) -> int:
        return self.reward.shape[0]


def _ground_point(position: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Lat/lon [deg] where rays from ``position`` (..., 3) along ``direction`` hit Earth."""
    b = np.sum(position * direction, axis=-1)
    c = np.sum(position**2, axis=-1) - EARTH_RADIUS_KM**2
    disc = b**2 - c
    with np.errstate(invalid="ignore"):
        s = -b - np.sqrt(disc)  # nearest intersection; NaN when the ray misses the Earth
    point = position + s[..., None] * direction
    return _latlon(point)


def _latlon(point: np.ndarray) -> np.ndarray:
    lat = np.degrees(np.arctan2(point[..., 2], np.hypot(point[..., 0], point[..., 1])))
    lon = np.degrees(np.arctan2(point[..., 1], point[..., 0]))
    return np.stack([lat, lon], axis=-1)


def _beam_directions(frame: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """World-frame unit vectors for body-frame (az, el) angles (N, K, 2); frame is (N, 3, 3)."""
    az, el = np.radians(angles_deg[..., 0]), np.radians(angles_deg[..., 1])
    body = np.stack([np.sin(el), np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)], axis=-1)
    return np.einsum("nkb,nbc->nkc", body, frame)


def run_episode(
    env: SarSat,
    policy: Policy,
    seed: int = 0,
    num_targets: int | None = None,
) -> EpisodeLog:
    """Roll out one episode under a fixed seed and record it.

    ``policy`` is the name of a :mod:`sarsat.reference` policy (acts from the full
    environment state and is built fresh for this episode), a plain function
    ``Observation -> action`` (jitted for the rollout), or a :class:`StatefulPolicy`
    (an object with ``reset()`` and ``__call__(obs) -> action``, e.g. a recurrent actor
    carrying a hidden state; called un-jitted every step, any jitting being its own
    business). ``env`` may be a plain :class:`SarSat`, a :class:`sarsat.windows.
    WindowedSarSat` (whose ``state.target_window`` is then also recorded), or a
    :class:`sarsat.coop.CoopSarSat` / ``WindowedCoopSarSat`` (whose observation is a
    ``CoopObservation``; only a policy that understands it should be used with one).
    """
    state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(seed), num_targets)
    num_targets = int(state.target_active.sum())
    target_window = (
        np.asarray(state.target_window)[:num_targets] if hasattr(state, "target_window") else None
    )

    if isinstance(policy, str):
        from sarsat.reference import ReferenceController

        controller = ReferenceController(env, state, policy)
        act = lambda state, obs: controller(state)  # noqa: E731
    elif hasattr(policy, "reset"):
        policy.reset()
        act = lambda state, obs: policy(obs)  # noqa: E731 (stateful: run un-jitted)
    else:
        act = jax.jit(lambda state, obs: policy(obs))

    @jax.jit
    def apply_and_step(state, action):
        pointing, sensing, in_beam = env._apply_action(state, state.step + 1, action)
        next_state, next_timestep = env.step(state, action)
        position, frame = satellite_frames(next_state.orbit, next_state.step * env.step_seconds)
        return next_state, next_timestep, (pointing, sensing, in_beam, position, frame)

    def act_and_step(state, obs):
        return apply_and_step(state, act(state, obs))

    position0, _ = satellite_frames(state.orbit, 0.0)
    sat_latlon = [_latlon(np.asarray(position0))]
    battery = [np.asarray(state.battery)]
    pointing_deg, sensing, boresight, footprint, captures, rewards = [], [], [], [], [], []
    hw = np.asarray(env._beam_half_width) * 180 / np.pi
    corner_offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * hw  # (4, 2)

    for _ in range(env.time_limit):
        state, timestep, (pointing, sense, in_beam, position, frame) = act_and_step(
            state, timestep.observation
        )
        pointing, sense, in_beam = (np.asarray(x) for x in (pointing, sense, in_beam))
        position, frame = np.asarray(position, np.float64), np.asarray(frame, np.float64)
        pointing = np.degrees(pointing)
        angles = np.concatenate([pointing[:, None, :], pointing[:, None, :] + corner_offsets], 1)
        ground = _ground_point(position[:, None, :], _beam_directions(frame, angles))

        sat_latlon.append(_latlon(position))
        battery.append(np.asarray(state.battery))
        pointing_deg.append(pointing)
        sensing.append(sense)
        boresight.append(ground[:, 0])
        footprint.append(ground[:, 1:])
        sats, targets = np.nonzero(in_beam)
        captures.append(np.stack([np.full_like(sats, int(state.step)), targets, sats], axis=1))
        rewards.append(float(timestep.reward[0]))
        if bool(timestep.last()):
            break

    return EpisodeLog(
        step_seconds=env.step_seconds,
        beam_half_width_deg=tuple(hw.tolist()),
        sat_latlon=np.stack(sat_latlon),
        sat_alt_km=np.asarray(state.orbit.radius) - EARTH_RADIUS_KM,
        battery=np.stack(battery),
        pointing_deg=np.stack(pointing_deg),
        sensing=np.stack(sensing),
        boresight=np.stack(boresight),
        footprint=np.stack(footprint),
        target_latlon=np.asarray(state.target_latlon)[:num_targets],
        target_on_land=np.arange(num_targets) < round(env.land_fraction * num_targets),
        target_priority=np.asarray(state.target_priority)[:num_targets],
        captures=np.concatenate(captures) if captures else np.zeros((0, 3), int),
        reward=np.asarray(rewards),
        target_window=target_window,
    )


# ---------------------------------------------------------------------------- CSV


def write_csv(log: EpisodeLog, out_dir: Path) -> List[Path]:
    """Write ``satellites.csv``, ``looks.csv``, ``targets.csv`` and ``captures.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, num_sats = log.sensing.shape
    f = "{:.6g}".format

    def write(name: str, header: List[str], rows) -> Path:
        path = out_dir / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    satellites = write(
        "satellites.csv",
        [
            "step",
            "time_s",
            "satellite",
            "lat_deg",
            "lon_deg",
            "alt_km",
            "battery",
            "sensing",
            "az_cmd_deg",
            "el_cmd_deg",
        ],
        (
            [
                t,
                f(t * log.step_seconds),
                n,
                f(log.sat_latlon[t, n, 0]),
                f(log.sat_latlon[t, n, 1]),
                f(log.sat_alt_km[n]),
                f(log.battery[t, n]),
                int(log.sensing[t - 1, n]) if t else 0,
                f(log.pointing_deg[t - 1, n, 0]) if t else "",
                f(log.pointing_deg[t - 1, n, 1]) if t else "",
            ]
            for t in range(steps + 1)
            for n in range(num_sats)
        ),
    )

    hits = np.zeros((steps, num_sats), int)
    for step, _, sat in log.captures:
        hits[step - 1, sat] += 1
    looks = write(
        "looks.csv",
        [
            "step",
            "time_s",
            "satellite",
            "az_deg",
            "el_deg",
            "boresight_lat_deg",
            "boresight_lon_deg",
        ]
        + [f"corner{i}_{c}_deg" for i in range(1, 5) for c in ("lat", "lon")]
        + ["targets_hit"],
        (
            [
                t + 1,
                f((t + 1) * log.step_seconds),
                n,
                f(log.pointing_deg[t, n, 0]),
                f(log.pointing_deg[t, n, 1]),
                f(log.boresight[t, n, 0]),
                f(log.boresight[t, n, 1]),
            ]
            + [f(v) for v in log.footprint[t, n].ravel()]
            + [hits[t, n]]
            for t, n in zip(*np.nonzero(log.sensing), strict=True)
        ),
    )

    captured_step = np.full(log.target_latlon.shape[0], -1)
    captured_by = np.full(log.target_latlon.shape[0], -1)
    for step, target, sat in log.captures[::-1]:  # earliest capture, lowest satellite wins
        captured_step[target], captured_by[target] = step, sat
    has_window = log.target_window is not None
    targets_header = [
        "target",
        "lat_deg",
        "lon_deg",
        "on_land",
        "priority",
        "captured_step",
        "captured_by",
    ]
    if has_window:
        targets_header = [*targets_header, "window_open", "window_close"]
    targets = write(
        "targets.csv",
        targets_header,
        (
            [
                m,
                f(log.target_latlon[m, 0]),
                f(log.target_latlon[m, 1]),
                int(log.target_on_land[m]),
                f(log.target_priority[m]),
                captured_step[m],
                captured_by[m],
            ]
            + ([int(log.target_window[m, 0]), int(log.target_window[m, 1])] if has_window else [])
            for m in range(log.target_latlon.shape[0])
        ),
    )

    cumulative = np.cumsum(log.reward)
    captures = write(
        "captures.csv",
        [
            "step",
            "time_s",
            "target",
            "satellite",
            "lat_deg",
            "lon_deg",
            "priority",
            "cumulative_return",
        ],
        (
            [
                step,
                f(step * log.step_seconds),
                target,
                sat,
                f(log.target_latlon[target, 0]),
                f(log.target_latlon[target, 1]),
                f(log.target_priority[target]),
                f(cumulative[step - 1]),
            ]
            for step, target, sat in log.captures
        ),
    )
    return [satellites, looks, targets, captures]


# --------------------------------------------------------------------------- HTML

_VIEWER_TEMPLATE = Path(__file__).with_name("viewer.html")


def write_html(log: EpisodeLog, path: Path, title: str = "SarSat episode") -> Path:
    """Write a single-file interactive map of the episode."""

    def rounded(array: np.ndarray, decimals: int = 3) -> list:
        array = np.round(np.asarray(array, np.float64), decimals)
        return np.where(np.isnan(array), None, array).tolist()

    data = {
        "title": title,
        "stepSeconds": log.step_seconds,
        "beamHalfWidthDeg": list(log.beam_half_width_deg),
        "satLatLon": rounded(log.sat_latlon, 2),
        "satAltKm": rounded(log.sat_alt_km, 0),
        "battery": rounded(log.battery, 3),
        "pointingDeg": rounded(log.pointing_deg, 1),
        "sensing": log.sensing.astype(int).tolist(),
        "boresight": rounded(log.boresight, 2),
        "footprint": rounded(log.footprint, 2),
        "targetLatLon": rounded(log.target_latlon, 2),
        "targetOnLand": log.target_on_land.astype(int).tolist(),
        "targetPriority": rounded(log.target_priority, 6),
        "captures": log.captures.tolist(),
        "reward": rounded(log.reward, 6),
    }
    if log.target_window is not None:
        window = log.target_window.astype(int)
        data["targetWindowOpen"] = window[:, 0].tolist()
        data["targetWindowClose"] = window[:, 1].tolist()
    html = _VIEWER_TEMPLATE.read_text().replace("/*EPISODE_DATA*/null", json.dumps(data))
    path = Path(path)
    path.write_text(html)
    return path


# ---------------------------------------------------------------------------- CLI


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--satellites", type=int, default=8)
    parser.add_argument("--targets", type=int, default=100)
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=sorted(POLICIES) + list(REFERENCE_POLICIES),
        default="greedy",
        help="an observation policy, or a sarsat.reference yardstick such as coop_plan",
    )
    parser.add_argument("--hotspots", type=int, default=0, help="high-value clusters (0: uniform)")
    parser.add_argument("--hotspot-targets", type=int, default=50)
    parser.add_argument("--hotspot-weight", type=float, default=10.0)
    parser.add_argument("--planes", type=int, default=0, help="orbital planes (0: random orbits)")
    parser.add_argument("--train-spacing-s", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = parser.parse_args(argv)

    env = SarSat(
        num_satellites=args.satellites,
        max_targets=args.targets,
        time_limit=args.time_limit,
        hotspots=args.hotspots,
        hotspot_targets=args.hotspot_targets,
        hotspot_weight=args.hotspot_weight,
        planes=args.planes,
        train_spacing_s=args.train_spacing_s,
    )
    log = run_episode(env, POLICIES.get(args.policy, args.policy), seed=args.seed)
    paths = write_csv(log, args.out)
    paths.append(
        write_html(
            log, args.out / "episode.html", f"SarSat: {args.policy} policy, seed {args.seed}"
        )
    )
    captured = len({int(target) for _, target, _ in log.captures})
    print(
        f"{log.num_steps} steps, {captured}/{log.target_latlon.shape[0]} targets imaged, "
        f"return {log.reward.sum():.3f}, {int(log.sensing.sum())} looks"
    )
    for path in paths:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
