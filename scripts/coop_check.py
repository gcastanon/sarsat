"""Sanity check and benchmark for CoopSarSat: the slot-0 policy should match greedy_beam."""

import sys
import time

import jax
import jax.numpy as jnp

from sarsat.coop import CoopSarSat

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
env = CoopSarSat(num_satellites=100, max_targets=8000, hotspots=40, hotspot_targets=50, planes=10)


def slot0_policy(obs):
    view = obs.agents_view
    return jnp.concatenate([view[..., 1:4], view[..., 0:1]], axis=-1)  # pointing + sense if valid


def episode(key):
    state, ts = env.reset(key)

    def body(carry, _):
        state, ts = carry
        state, ts = env.step(state, slot0_policy(ts.observation))
        return (state, ts), (ts.reward[0], ts.observation.agents_view)

    _, (reward, views) = jax.lax.scan(body, (state, ts), None, env.time_limit)
    return reward.sum(), views


run = jax.jit(jax.vmap(episode))
keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(1000, 1000 + num_envs))
t = time.perf_counter()
ret, views = run(keys)
ret.block_until_ready()
print(f"compile+run {time.perf_counter() - t:.1f}s")
t = time.perf_counter()
ret, views = run(keys)
ret.block_until_ready()
dt = time.perf_counter() - t
print(f"{num_envs * env.time_limit / dt:.0f} env steps/s ({num_envs} envs)")
print("slot0 returns:", [round(float(r), 4) for r in ret], "mean", float(ret.mean()))
flat = views.reshape(-1, views.shape[-1])
print("obs min/max per feature:")
print(jnp.round(flat.min(0), 2))
print(jnp.round(flat.max(0), 2))
print("obs mean per feature:")
print(jnp.round(flat.mean(0), 3))
print("finite:", bool(jnp.isfinite(flat).all()))
