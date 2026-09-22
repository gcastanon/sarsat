"""The fused auto-reset must hand over exactly the state a plain reset would build."""

import time

import jax
import jax.numpy as jnp
from mapx.wrappers.sarsat_coop import SarSatCoopWrapper

from sarsat.coop import CoopSarSat


def _env(**kwargs) -> CoopSarSat:
    return CoopSarSat(
        num_satellites=20, max_targets=600, hotspots=8, hotspot_targets=25, planes=5, **kwargs
    )


def test_fused_reset_matches_plain_reset() -> None:
    env = _env(time_limit=70)
    wrapper = SarSatCoopWrapper(env, fused_auto_reset=True)
    state, _ = jax.jit(wrapper.reset)(jax.random.PRNGKey(0))
    next_key = state.next_key
    action = jnp.tile(jnp.asarray([0.0, 0.0, 1.0, 1.0]), (env.num_agents, 1))
    step = jax.jit(wrapper.step)
    for _ in range(env.time_limit):
        state, timestep = step(state, action)
    assert bool(timestep.last())
    expected = jax.jit(lambda k: env.refresh(env.reset_state(k)))(next_key)
    assert int(state.env_state.step) == 0
    for name, got, want in zip(state.env_state._fields, state.env_state, expected, strict=True):
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            if jnp.issubdtype(a.dtype, jnp.floating):
                assert jnp.allclose(a, b, rtol=1e-5, atol=1e-3), name
            else:
                assert jnp.array_equal(a, b), name
    assert not jnp.array_equal(state.next_key, next_key)


if __name__ == "__main__":
    test_fused_reset_matches_plain_reset()
    print("fused reset OK")
    env = CoopSarSat(
        num_satellites=100, max_targets=8000, hotspots=40, hotspot_targets=50, planes=10
    )
    wrapper = SarSatCoopWrapper(env, True, 10.0, fused_auto_reset=True)
    n = 32
    states, _ = jax.jit(jax.vmap(wrapper.reset))(jax.random.split(jax.random.PRNGKey(0), n))
    action = jnp.tile(jnp.asarray([0.0, 0.0, 1.0, 1.0]), (n, env.num_agents, 1))

    @jax.jit
    def rollout(states):
        def body(s, _):
            s, ts = jax.vmap(wrapper.step)(s, action)
            return s, ts.reward[:, 0]

        return jax.lax.scan(body, states, None, 60)

    jax.block_until_ready(rollout(states))
    t = time.perf_counter()
    jax.block_until_ready(rollout(states))
    print(f"{n * 60 / (time.perf_counter() - t):.0f} env steps/s, fused wrapper, {n} envs")
