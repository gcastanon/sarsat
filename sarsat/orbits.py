"""Closed-form circular-orbit kinematics and the egocentric az/el convention.

Everything here is a pure function of time, so there is no integrator and no
accumulated numerical error: a satellite's position at step ``k`` is evaluated
directly from its orbital elements (see DECISIONS.md, issues 2 and 3).

Units are kilometres, seconds and radians. All frames are Earth-fixed (ECEF).
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from sarsat.types import Orbit

EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 398600.4418  # gravitational parameter
EARTH_ROTATION_RAD_S = 7.2921159e-5
# Relative slack on the access bounds: about 1e-3 degrees, far below any real geometry,
# and enough that a float32 comparison on an exact edge is not a coin flip.
BAND_TOLERANCE = 1e-5


def sample_orbits(
    key: jax.Array,
    num_satellites: int,
    altitude_km: Tuple[float, float],
    inclination_deg: Tuple[float, float],
) -> Orbit:
    """Draw independent circular orbits, uniform in altitude, inclination, RAAN and phase."""
    k_alt, k_inc, k_raan, k_phase = jax.random.split(key, 4)
    shape = (num_satellites,)
    radius = EARTH_RADIUS_KM + jax.random.uniform(
        k_alt, shape, minval=altitude_km[0], maxval=altitude_km[1]
    )
    inclination = jnp.deg2rad(
        jax.random.uniform(k_inc, shape, minval=inclination_deg[0], maxval=inclination_deg[1])
    )
    return Orbit(
        radius=radius,
        mean_motion=jnp.sqrt(EARTH_MU_KM3_S2 / radius**3),
        inclination=inclination,
        raan=jax.random.uniform(k_raan, shape, maxval=2 * jnp.pi),
        phase=jax.random.uniform(k_phase, shape, maxval=2 * jnp.pi),
    )


def satellite_frames(orbit: Orbit, time_s: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Satellite positions and body frames at ``time_s``, in Earth-fixed coordinates.

    Earth rotation is folded into the node angle: seen from the rotating Earth, the
    orbital plane simply drifts westwards at the Earth's rotation rate, so no rotation
    matrix is needed.

    Args:
        orbit: elements of N satellites.
        time_s: scalar time since the start of the episode [s].

    Returns:
        position: (N, 3) [km].
        frame: (N, 3, 3) orthonormal body axes as rows: ``along`` (direction of flight),
            ``cross`` (right-handed completion) and ``nadir`` (towards Earth's centre).
    """
    u = orbit.phase + orbit.mean_motion * time_s
    node = orbit.raan - EARTH_ROTATION_RAD_S * time_s
    cos_u, sin_u = jnp.cos(u), jnp.sin(u)
    cos_n, sin_n = jnp.cos(node), jnp.sin(node)
    cos_i, sin_i = jnp.cos(orbit.inclination), jnp.sin(orbit.inclination)
    zeros = jnp.zeros_like(u)

    # Orthonormal basis of the orbital plane: p points at the ascending node, q is 90
    # degrees ahead of it along the orbit, and p x q is the orbit normal.
    p = jnp.stack([cos_n, sin_n, zeros], axis=-1)
    q = jnp.stack([-sin_n * cos_i, cos_n * cos_i, sin_i], axis=-1)
    normal = jnp.stack([sin_n * sin_i, -cos_n * sin_i, cos_i], axis=-1)

    radial = cos_u[:, None] * p + sin_u[:, None] * q
    along = -sin_u[:, None] * p + cos_u[:, None] * q
    nadir = -radial
    cross = -normal  # = nadir x along
    return orbit.radius[:, None] * radial, jnp.stack([along, cross, nadir], axis=-2)


def az_el(body_vector: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Azimuth and elevation of a vector given in body axes ``(along, cross, nadir)``.

    The boresight reference (az = el = 0) is nadir. Azimuth is the cross-track look
    angle and elevation the along-track look angle:

        direction = (sin el, cos el * sin az, cos el * cos az)

    The only singularities of this convention lie along the flight direction, which is
    above the horizon and therefore never a ground target. ``arctan2`` needs no
    normalisation or division, so the result is finite for any non-zero input.
    """
    along, cross, nadir = body_vector[..., 0], body_vector[..., 1], body_vector[..., 2]
    return jnp.arctan2(cross, nadir), jnp.arctan2(along, jnp.hypot(cross, nadir))


def in_az_el_box(body_vector: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
    """Whether ``lower <= (az, el) <= upper`` holds, without evaluating any angle.

    Equivalent to comparing :func:`az_el` against the bounds, but done in tangent space:
    ``tan`` is evaluated on the bounds (one pair per satellite) instead of ``arctan2`` on
    every satellite-target pair. Bounds have a trailing axis of 2 (az, el), must lie
    strictly inside +-90 degrees and broadcast against ``body_vector[..., 0]``.
    """
    along, cross, nadir = body_vector[..., 0], body_vector[..., 1], body_vector[..., 2]
    horizontal = jnp.hypot(cross, nadir)
    tan_lower, tan_upper = jnp.tan(lower), jnp.tan(upper)
    return (
        (nadir > 0.0)
        & (tan_lower[..., 0] * nadir <= cross)
        & (cross <= tan_upper[..., 0] * nadir)
        & (tan_lower[..., 1] * horizontal <= along)
        & (along <= tan_upper[..., 1] * horizontal)
    )


def look_angle_band(radius: jax.Array, incidence: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Look angles (at the satellite) that put a target inside an incidence band.

    Incidence is measured at the target, between the line of sight and the local
    vertical; the look angle is measured at the satellite, from nadir. The sine rule on
    the satellite-centre-target triangle gives ``sin(incidence) = (r / R) sin(look)``,
    so a band of incidence maps to a band of look angle that depends on orbit radius.
    Incidence always exceeds look angle, and the gap grows with range.

    Args:
        radius: ``(N,)`` orbit radii [km].
        incidence: ``(2,)`` ``(min, max)`` incidence angles [rad], both below pi / 2.

    Returns:
        ``(N,)`` minimum and maximum look angle [rad].
    """
    sin_look = EARTH_RADIUS_KM / radius[:, None] * jnp.sin(incidence)
    look = jnp.arcsin(jnp.clip(sin_look, 0.0, 1.0))
    return look[:, 0], look[:, 1]


def cross_track_band(
    off_nadir_min: jax.Array, off_nadir_max: jax.Array, squint: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Cross-track look angles that keep the *total* off-nadir angle inside a band.

    Squint and cross-track look combine: a line of sight at cross-track angle ``a`` and
    squint ``s`` has ``cos(off_nadir) = cos(s) cos(a)``, so squinting already moves the
    beam off nadir and the cross-track range available shrinks accordingly. At enough
    squint the lower edge is reached without any cross-track offset at all, and the
    clamp below returns zero.
    """
    cos_squint = jnp.cos(squint)
    low = jnp.arccos(jnp.clip(jnp.cos(off_nadir_min) / cos_squint, -1.0, 1.0))
    high = jnp.arccos(jnp.clip(jnp.cos(off_nadir_max) / cos_squint, -1.0, 1.0))
    return low, jnp.maximum(high, low)


def in_incidence_band(
    body_vector: jax.Array,
    off_nadir_min: jax.Array,
    off_nadir_max: jax.Array,
    max_squint: jax.Array,
) -> jax.Array:
    """Two-sided side-looking access test, evaluated without a single square root.

    True where the line of sight's *total* off-nadir angle lies in
    ``[off_nadir_min, off_nadir_max]`` -- on either side of the ground track, so the
    accessible region is a pair of annular sectors with a hole over nadir -- and the
    along-track squint is within ``+-max_squint``. Because ``cos(off_nadir)`` is
    ``nadir / |line of sight|``, every comparison can be squared, which keeps the test
    to multiplies and comparisons on the satellite-target pairs; the trigonometry is
    evaluated once per satellite on the bounds. Bounds broadcast against
    ``body_vector[..., 0]`` and must lie strictly inside +-90 degrees.

    The bounds are opened by :data:`BAND_TOLERANCE` so that a target lying exactly on an
    edge -- which is what a satellite aimed at the edge of its own band is looking at --
    is inside it, rather than left to float32 round-off to decide.
    """
    along, cross, nadir = body_vector[..., 0], body_vector[..., 1], body_vector[..., 2]
    in_plane_sq = cross * cross + nadir * nadir
    norm_sq = along * along + in_plane_sq
    nadir_sq = nadir * nadir
    cos_min_sq = jnp.cos(off_nadir_min) ** 2 * (1.0 + BAND_TOLERANCE)
    cos_max_sq = jnp.cos(off_nadir_max) ** 2 * (1.0 - BAND_TOLERANCE)
    tan_squint_sq = jnp.tan(max_squint) ** 2 * (1.0 + BAND_TOLERANCE)
    return (
        (nadir > 0.0)
        & (nadir_sq <= cos_min_sq * norm_sq)  # off-nadir angle at or past the inner edge
        & (nadir_sq >= cos_max_sq * norm_sq)  # and no further than the outer edge
        & (along * along <= tan_squint_sq * in_plane_sq)
    )


def sample_planes(
    key: jax.Array,
    num_satellites: int,
    planes: int,
    altitude_km: Tuple[float, float],
    inclination_deg: Tuple[float, float],
    train_spacing_s: float = 0.0,
) -> Orbit:
    """Draw a structured constellation: ``planes`` orbital planes sharing their satellites.

    Planes are evenly spaced in RAAN (with a random common rotation) and each draws its own
    altitude and inclination from the given ranges. Within a plane the satellites are
    either spread evenly around the orbit -- a Walker-style constellation
    (``train_spacing_s == 0``) -- or flown as a train, each ``train_spacing_s`` seconds
    behind the one before. Structured constellations overlap in coverage, which is what
    gives cooperation something to do (DECISIONS.md issue 22); a train spacing of a few
    seconds makes near-identical formation flyers.

    ``num_satellites`` must be a multiple of ``planes``.
    """
    if num_satellites % planes:
        raise ValueError("num_satellites must be a multiple of planes")
    per_plane = num_satellites // planes
    k_alt, k_inc, k_raan, k_phase = jax.random.split(key, 4)
    plane = jnp.arange(num_satellites) // per_plane
    slot = jnp.arange(num_satellites) % per_plane
    radius = EARTH_RADIUS_KM + jax.random.uniform(
        k_alt, (planes,), minval=altitude_km[0], maxval=altitude_km[1]
    )
    inclination = jnp.deg2rad(
        jax.random.uniform(k_inc, (planes,), minval=inclination_deg[0], maxval=inclination_deg[1])
    )
    mean_motion = jnp.sqrt(EARTH_MU_KM3_S2 / radius**3)
    raan = 2 * jnp.pi * plane / planes + jax.random.uniform(k_raan, maxval=2 * jnp.pi)
    plane_phase = jax.random.uniform(k_phase, (planes,), maxval=2 * jnp.pi)
    if train_spacing_s > 0:
        offset = -slot * mean_motion[plane] * train_spacing_s  # followers trail the leader
    else:
        # Walker delta: even spacing in-plane, staggered by one sub-slot between planes.
        offset = 2 * jnp.pi * (slot + plane / planes) / per_plane
    return Orbit(
        radius=radius[plane],
        mean_motion=mean_motion[plane],
        inclination=inclination[plane],
        raan=raan % (2 * jnp.pi),
        phase=(plane_phase[plane] + offset) % (2 * jnp.pi),
    )
