"""Run the fused-auto-reset wrapper for several back-to-back episodes with the slot-0
policy and compare every episode's return and observation statistics with the first."""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if not os.path.abspath(p or ".").startswith(_here[:-6])]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from mapx.wrappers.sarsat_coop import SarSatCoopWrapper  # noqa: E402

from sarsat.coop import CoopSarSat  # noqa: E402

env = CoopSarSat(num_satellites=100, max_targets=8000, hotspots=40, hotspot_targets=50, planes=10)
wrapper = SarSatCoopWrapper(env, True, 1.0, fused_auto_reset=True)
n, episodes = 8, 3
states, ts = jax.jit(jax.vmap(wrapper.reset))(jax.random.split(jax.random.PRNGKey(3), n))


def policy(obs):
    view = obs.agents_view
    return jnp.concatenate([view[..., 1:4], view[..., 0:1]], axis=-1)


@jax.jit
def rollout(states, ts):
    def body(carry, _):
        s, t = carry
        s, t = jax.vmap(wrapper.step)(s, policy(t.observation))
        o = t.observation
        return (s, t), (t.reward[:, 0], o.agents_view.mean((0, 1)), o.global_state.mean((0, 1)))

    return jax.lax.scan(body, (states, ts), None, episodes * env.time_limit)


_, (reward, view, glob) = rollout(states, ts)
reward = reward.reshape(episodes, env.time_limit, n)
view = view.reshape(episodes, env.time_limit, -1).mean(1)
glob = glob.reshape(episodes, env.time_limit, -1).mean(1)
for e in range(episodes):
    print(
        f"episode {e}: returns {jnp.round(reward[e].sum(0), 3)} mean {reward[e].sum(0).mean():.4f}"
    )
for e in range(1, episodes):
    print(
        f"episode {e} vs 0: max |d view| {jnp.abs(view[e] - view[0]).max():.4f} at "
        f"{int(jnp.abs(view[e] - view[0]).argmax())}, max |d global| "
        f"{jnp.abs(glob[e] - glob[0]).max():.4f} at {int(jnp.abs(glob[e] - glob[0]).argmax())}"
    )
print("view ep0:", jnp.round(view[0], 3))
print("view ep1:", jnp.round(view[1], 3))
