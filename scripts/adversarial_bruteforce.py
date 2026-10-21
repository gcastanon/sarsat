"""Exhaustive search over every joint look plan on scenarios/adversarial_pair.json.

A plan gives each satellite a set of (step, target) looks, the beam centred on the target.
That covers every imaged set when no beam can hold two targets, which is checked first.
Every plan is executed through the real ``SarSat.step``; unaffordable looks are dropped by
the environment, so battery feasibility is enforced by the dynamics, not assumed.
"""

import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np

from sarsat.env import SarSat
from sarsat.orbits import EARTH_RADIUS_KM, az_el
from sarsat.targets import latlon_to_unit_vector
from sarsat.types import Orbit, State

doc = json.load(open("scenarios/adversarial_pair.json"))
e, sats, tg = doc["environment"], doc["satellites"], doc["targets"]
n, m, T = len(sats), len(tg), e["time_limit_steps"]
env = SarSat(num_satellites=n, max_targets=m, sense_cost=e["sense_cost"],
             recharge_rate=e["recharge_rate"], time_limit=T)
f32 = lambda x: jnp.asarray(x, jnp.float32)
orbit = Orbit(radius=f32([s["radius_km"] for s in sats]),
              mean_motion=f32([s["mean_motion_rad_s"] for s in sats]),
              inclination=f32(np.radians([s["inclination_deg"] for s in sats])),
              raan=f32(np.radians([s["raan_deg"] for s in sats])),
              phase=f32(np.radians([s["arg_latitude_deg_t0"] for s in sats])))
ll = f32([[t["lat_deg"], t["lon_deg"]] for t in tg])
s0 = State(key=jax.random.PRNGKey(0), step=jnp.zeros((), jnp.int32), orbit=orbit,
           battery=f32([s["initial_battery"] for s in sats]), target_latlon=ll,
           target_pos=EARTH_RADIUS_KM * latlon_to_unit_vector(ll),
           target_priority=f32([t["priority"] for t in tg]), target_active=jnp.ones(m, bool))
geometry, step_fn = jax.jit(env._target_geometry), jax.jit(env.step)
hw = np.asarray(env._beam_half_width)

# Geometry at every step from the initial state (all targets active).
body, vis = zip(*[geometry(s0, jnp.int32(s)) for s in range(1, T + 1)])
body, vis = np.stack(body), np.stack(vis)  # (T, N, M, 3), (T, N, M)
az, el = (np.asarray(a) for a in az_el(jnp.asarray(body)))

# 1. Can any beam hold two targets? (a box of half-widths hw holds both iff the separations
#    are within 2 hw on both axes)
two = [(s + 1, i, a, b) for s in range(T) for i in range(n) for a, b in itertools.combinations(range(m), 2)
       if vis[s, i, a] and vis[s, i, b] and abs(az[s, i, a] - az[s, i, b]) <= 2 * hw[0]
       and abs(el[s, i, a] - el[s, i, b]) <= 2 * hw[1]]
print("beams that could hold two targets:", two or "none")

# 2. Every satellite's look options, and every plan of up to two looks each.
options = [[(s + 1, k) for s in range(T) for k in range(m) if vis[s, i, k]] for i in range(n)]
print("look options per satellite:", [len(o) for o in options])


def plans(opts):
    yield ()
    for r in (1, 2):
        for c in itertools.combinations(opts, r):
            if len({s for s, _ in c}) == r:
                yield c


def run(plan):
    st, ret, made = s0, 0.0, 0
    for s in range(1, T + 1):
        b, _ = geometry(st, jnp.int32(s))
        aim = np.zeros(n, int)
        look = np.zeros(n)
        for i in range(n):
            for ss, k in plan[i]:
                if ss == s:
                    aim[i], look[i] = k, 1.0
        a_az, a_el = az_el(b[jnp.arange(n), jnp.asarray(aim)])
        pointing = env._encode_pointing(st.orbit, a_az[:, None], a_el[:, None])[:, 0]
        before = st.battery
        st, ts = step_fn(st, jnp.concatenate([pointing, f32(look)[:, None]], -1))
        made += int(np.sum((look > 0) & (np.asarray(before) >= env.sense_cost - 1e-6)))
        ret += float(ts.reward[0])
    return ret, made


best, count, results = -1.0, 0, {}
for p in itertools.product(*[list(plans(o)) for o in options]):
    r, made = run(p)
    count += 1
    results[p] = r
    if r > best + 1e-9:
        best, argbest = r, [p]
    elif abs(r - best) < 1e-9:
        argbest.append(p)
names = [t["name"] for t in tg]
fmt = lambda p: " | ".join(f"sat{i}: " + (", ".join(f"{names[k]}@{s}" for s, k in p[i]) or "-") for i in range(n))
print(f"plans executed: {count}")
print(f"optimum: {best:.4f}, reached by {len(argbest)} plans, e.g. {fmt(argbest[0])}")
two_look = [p for p in results if any(len(x) == 2 for x in p)]
print("best return of any plan asking for a second look:", max(results[p] for p in two_look))
print("distinct returns:", sorted({round(v, 4) for v in results.values()}))
