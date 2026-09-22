"""Throughput benchmark. Run: ``python scripts/benchmark.py``"""

import time

import jax

from sarsat import SarSat


def benchmark(num_satellites: int, max_targets: int, num_envs: int, num_steps: int = 50) -> None:
    env = SarSat(num_satellites=num_satellites, max_targets=max_targets)

    def rollout(key: jax.Array) -> jax.Array:
        state, _ = env.reset(key)

        def body(state, key):
            action = jax.random.uniform(key, (num_satellites, 3), minval=-1.0)
            state, timestep = env.step(state, action)
            return state, timestep.reward[0]

        return jax.lax.scan(body, state, jax.random.split(key, num_steps))[1].sum()

    run = jax.jit(jax.vmap(rollout))
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    run(keys).block_until_ready()  # compile
    start = time.perf_counter()
    run(keys).block_until_ready()
    rate = num_envs * num_steps / (time.perf_counter() - start)
    print(
        f"{num_satellites:5d} satellites {max_targets:5d} targets {num_envs:5d} envs: "
        f"{rate:12,.0f} env steps/s  {rate * num_satellites:14,.0f} agent steps/s"
    )


if __name__ == "__main__":
    print(f"device: {jax.devices()[0]}")
    benchmark(8, 100, 1024)
    benchmark(64, 1000, 64)
    benchmark(1000, 1000, 1)
