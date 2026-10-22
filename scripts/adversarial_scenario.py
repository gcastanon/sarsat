"""Smallest adversarial SarSat layout separating greedy_beam < coop_dedup < solo_plan < coop_plan.

Two satellites 3 s apart in one plane (same view), one look of charge each, four targets:
two junk targets J1, J2 (priority j) seen at an early step, two treasures T1, T2 (priority t)
seen later, each in its own beam. Predicted returns with j < t/2:
  greedy_beam: both look at J1 now          -> j
  coop_dedup:  one J1, one J2               -> 2j
  solo_plan:   both wait, both look at T1   -> t
  coop_plan:   both wait, one T1, one T2    -> 2t
"""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.env import SarSat
from sarsat.orbits import EARTH_RADIUS_KM, az_el, satellite_frames
from sarsat.reference import ReferenceController
from sarsat.targets import latlon_to_unit_vector

T_LIMIT, SENSE, RECHARGE, BATTERY0 = 10, 0.5, 0.001, 0.55
J_STEPS, T_STEPS = (2, 3), (6, 7)  # steps a target may be seen at (at least the first)
PRIO = np.array([3.0, 3.0, 1.0, 1.0])  # T1, T2, J1, J2 (the events first: hotspot slots)
NAMES = ["T1", "T2", "J1", "J2"]
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = sys.argv[2] if len(sys.argv) > 2 else "scenarios/adversarial_pair.json"

env = SarSat(num_satellites=2, max_targets=4, planes=1, train_spacing_s=3.0,
             sense_cost=SENSE, recharge_rate=RECHARGE, time_limit=T_LIMIT)
state, _ = env.reset(jax.random.PRNGKey(SEED))
hw = np.asarray(env._beam_half_width)


def latlon(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.stack([np.degrees(np.arcsin(v[..., 2])), np.degrees(np.arctan2(v[..., 1], v[..., 0]))], -1)


def with_targets(ll, active=None):
    ll = jnp.asarray(ll, jnp.float32)
    return state._replace(
        target_latlon=ll, target_pos=EARTH_RADIUS_KM * latlon_to_unit_vector(ll),
        target_active=jnp.ones(len(ll), bool) if active is None else active,
        target_priority=jnp.ones(len(ll), jnp.float32) / len(ll))


geometry = jax.jit(env._target_geometry)


def access(st):
    """(T, N, P) visibility and (T, N, P, 2) az/el for steps 1..T."""
    vis, ang = [], []
    for s in range(1, T_LIMIT + 1):
        body, v = geometry(st, jnp.int32(s))
        vis.append(np.asarray(v))
        ang.append(np.stack([np.asarray(a) for a in az_el(body)], -1))
    return np.stack(vis), np.stack(ang)


# Candidates: random points within ~900 km of the leader's ground track.
rng = np.random.default_rng(SEED)
cands = []
for s in range(1, T_LIMIT + 1):
    pos, _ = satellite_frames(state.orbit, s * env.step_seconds)
    sub = latlon(np.asarray(pos)[0])
    lat = sub[0] + rng.uniform(-8, 8, 4000)
    lon = sub[1] + rng.uniform(-8, 8, 4000) / np.cos(np.radians(np.clip(lat, -80, 80)))
    cands.append(np.stack([lat, (lon + 180) % 360 - 180], -1))
cands = np.concatenate(cands)
vis, ang = access(with_targets(cands))
steps = np.arange(1, T_LIMIT + 1)


def matching(window):
    """Candidates both satellites see at window[0], only within ``window``."""
    both = vis.all(1)  # (T, P)
    any_ = vis.any(1)
    inside = np.isin(steps, window)[:, None]
    return np.flatnonzero(both[window[0] - 1] & ~(any_ & ~inside).any(0)
                          & (vis[:, 0] == vis[:, 1]).all(0))


def separated_pair(idx, step):
    """Two candidates whose beams (aimed at either) never contain the other, for both satellites."""
    a = ang[step - 1][:, idx]  # (N, k, 2)
    for i in range(len(idx)):
        d = np.abs(a[:, i : i + 1] - a)  # (N, k, 2)
        far = ((d[..., 0] > 2 * hw[0]) | (d[..., 1] > 2 * hw[1])).all(0)
        j = np.flatnonzero(far)
        if len(j):
            return idx[i], idx[j[0]]
    return None


jc, tc = matching(J_STEPS), matching(T_STEPS)
print(f"candidates: junk {len(jc)}, treasure {len(tc)}")
jp, tp = separated_pair(jc, J_STEPS[0]), separated_pair(tc, T_STEPS[0])
if jp is None or tp is None:
    sys.exit("no separated pair found; try another seed")
ll = cands[[*tp, *jp]]

# The scenario: four targets, normalised priorities, one look of charge per satellite.
final = with_targets(ll)._replace(
    target_priority=jnp.asarray(PRIO / PRIO.sum(), jnp.float32),
    battery=jnp.full((2,), BATTERY0, jnp.float32))
fvis, fang = access(final)
step_fn = jax.jit(env.step)
results = {}
for policy in ("greedy_beam", "coop_dedup", "solo_plan", "coop_plan"):
    ctl, st, ret, log = ReferenceController(env, final, policy), final, 0.0, []
    for _ in range(T_LIMIT):
        before = np.asarray(st.target_active)
        act = ctl(st)
        look = np.asarray(act[:, -1])
        st, ts = step_fn(st, act)
        ret += float(ts.reward[0])
        got = np.flatnonzero(before & ~np.asarray(st.target_active))
        if look.any() or len(got):
            log.append({"step": int(st.step), "looking": [int(x) for x in np.flatnonzero(look > 0.5)],
                        "imaged": [NAMES[k] for k in got]})
        if not np.asarray(st.target_active).any():
            break
    results[policy] = {"return": round(ret, 6), "events": log}
    print(f"{policy:12s} {ret:.4f}  {log}")

# Export.
o = state.orbit
pos0, _ = satellite_frames(o, 0.0)
targets = []
for k, name in enumerate(NAMES):
    windows = []
    for n in range(2):
        s = steps[fvis[:, n, k]]
        if len(s):
            windows.append({"satellite": n, "steps": [int(x) for x in s],
                            "time_s": [float(x * env.step_seconds) for x in s],
                            "az_el_deg": [[round(float(np.degrees(v)), 3) for v in fang[x - 1, n, k]] for x in s]})
    targets.append({"name": name, "lat_deg": round(float(ll[k, 0]), 5), "lon_deg": round(float(ll[k, 1]), 5),
                    "priority": float(PRIO[k] / PRIO.sum()), "priority_raw": float(PRIO[k]),
                    "access_windows": windows})
sats = []
for n in range(2):
    lat, lon = latlon(np.asarray(pos0)[n])
    sats.append({"id": n, "altitude_km": round(float(o.radius[n] - EARTH_RADIUS_KM), 4),
                 "radius_km": float(o.radius[n]), "inclination_deg": float(np.degrees(o.inclination[n])),
                 "raan_deg": float(np.degrees(o.raan[n])), "arg_latitude_deg_t0": float(np.degrees(o.phase[n])),
                 "mean_motion_rad_s": float(o.mean_motion[n]),
                 "subsatellite_lat_lon_deg_t0": [round(float(lat), 5), round(float(lon), 5)],
                 "initial_battery": BATTERY0})
doc = {
    "name": "sarsat-adversarial-pair",
    "seed": SEED,
    "conventions": "Circular orbits; Earth-fixed frame at t=0 with Earth rotation folded into the node "
                   "(raan drifts west at 7.2921e-5 rad/s, sarsat.orbits.satellite_frames). Step s is "
                   "time s*60 s; a look chosen for step s is made at the step-s geometry.",
    "environment": {"step_seconds": env.step_seconds, "time_limit_steps": T_LIMIT,
                    "sense_cost": SENSE, "recharge_rate": RECHARGE, "initial_battery": BATTERY0,
                    "incidence_deg": [20.0, 45.0], "max_squint_deg": 25.0,
                    "beam_half_width_deg": [float(np.degrees(h)) for h in hw],
                    "reward": "sum of priorities of targets imaged (once each); priorities sum to 1"},
    "satellites": sats,
    "targets": targets,
    "reference_results": results,
}
with open(OUT, "w") as f:
    json.dump(doc, f, indent=1)
print("wrote", OUT)
