"""Greedy baseline: image the first visible target slot whenever the battery allows.

Doubles as a usage example. Run: ``python scripts/greedy_baseline.py``
"""

import jax

from sarsat import SarSat
from sarsat.evaluate import greedy_policy


def episode_return(env: SarSat, key: jax.Array) -> jax.Array:
    state, timestep = env.reset(key)

    def body(carry, _):
        state, timestep = carry
        state, timestep = env.step(state, greedy_policy(timestep.observation))
        return (state, timestep), timestep.reward[0]

    _, rewards = jax.lax.scan(body, (state, timestep), None, env.time_limit)
    return rewards.sum()


if __name__ == "__main__":
    for num_satellites in (8, 64):
        env = SarSat(num_satellites=num_satellites, max_targets=100)
        keys = jax.random.split(jax.random.PRNGKey(0), 16)
        returns = jax.jit(jax.vmap(lambda k, env=env: episode_return(env, k)))(keys)
        print(f"{num_satellites:4d} satellites: mean fraction imaged = {returns.mean():.3f}")
