"""Time the pieces of a CoopSarSat step under vmap (GPU)."""

import sys
import time

import jax
import jax.numpy as jnp

from sarsat.coop import CoopSarSat

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
env = CoopSarSat(num_satellites=100, max_targets=8000, hotspots=40, hotspot_targets=50, planes=10)
keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
states = jax.jit(jax.vmap(lambda k: env.refresh(env.reset_state(k))))(keys)
action = jnp.tile(jnp.asarray([0.0, 0.0, 1.0, 1.0]), (num_envs, env.num_agents, 1))


def timeit(name, fn, *args, reps=20):
    fn = jax.jit(jax.vmap(fn))
    jax.block_until_ready(fn(*args))
    t = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    print(f"{name:12s} {(time.perf_counter() - t) / reps * 1e3:8.2f} ms", flush=True)


def geometry(state):
    return env._target_geometry(state, state.step + 1)[1]


def topk(state):
    _, visible = env._target_geometry(state, state.step + 1)
    return jax.lax.top_k(jnp.where(visible, state.target_priority, -1.0), env.num_candidates)[1]


timeit("geometry", geometry, states)
timeit("geom+top_k", topk, states)
timeit("observe", env.observe, states)
timeit("transition", lambda s, a: env.transition(s, a)[1], states, action)
timeit("reset_state", env.reset_state, keys)
timeit("base_state", lambda k: env.base_state(k).target_pos, keys)
timeit("refresh", lambda s: env.refresh(s).visible, states)
timeit("full step", lambda s, a: env.step(s, a)[1].observation.agents_view, states, action)
