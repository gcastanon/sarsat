"""Shape / consistency smoke test of the slot-mixture head (CPU is fine)."""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if not os.path.abspath(p or ".").startswith(_here[:-6])]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from mapx.networks.sarsat import SlotMixtureHead  # noqa: E402
from mapx.types import ActionBlock, ActionLayout  # noqa: E402

layout = ActionLayout((ActionBlock(size=2), *([ActionBlock(size=1, num_categories=2)] * 2)))
head = SlotMixtureHead(action_dim=4, layout=layout)
key = jax.random.PRNGKey(0)
emb = jax.random.normal(key, (3, 5, 7, 16))
slots = jax.random.uniform(key, (3, 5, 7, 4, 8), minval=-0.9, maxval=0.9)
slots = slots.at[..., 0].set((slots[..., 0] > 0).astype(jnp.float32))
slots = slots.at[..., 3].set((slots[..., 3] > 0).astype(jnp.float32)).reshape(3, 5, 7, 32)
mask = jnp.ones((3, 5, 7, 4), bool).at[..., 0, 3].set(False)
params = head.init(key, emb, mask, slots)
dist = head.apply(params, emb, mask, slots)
a = dist.sample(seed=key)
lp = dist.log_prob(a)
print("sample", a.shape, "log_prob", lp.shape, "finite", bool(jnp.isfinite(lp).all()))
print("mode", dist.mode().shape, "entropy", dist.entropy(seed=key).shape)
assert not bool(a[..., 0, 3].any()), "masked sense must stay off"
# Monte-Carlo check that the density integrates the discrete parts: E[exp(-lp)] is large but
# the side/sense marginals must match their logits.
many = jax.vmap(lambda k: dist.sample(seed=k))(jax.random.split(key, 2000))
print("P(sense) sampled", float(many[..., 1:, 3].mean()), "(logits near 0 -> ~0.5)")
g = jax.grad(lambda p: head.apply(p, emb, mask, slots).log_prob(a).sum())(params)
print("grad finite", all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(g)))

# Collection modes (DECISIONS.md issue 33): slots end with a one-hot of their mode and the
# look slot holds the chosen slot's mode, so every sampled look must be some slot's mode.
modes = 4
layout_m = ActionLayout(
    (ActionBlock(size=2), ActionBlock(size=1, num_categories=2),
     ActionBlock(size=1, num_categories=modes + 1))
)
head_m = SlotMixtureHead(action_dim=4, layout=layout_m, slot_dim=8 + modes)
slot_mode = jax.random.randint(key, (3, 5, 7, 4), 0, modes)
slots_m = jnp.concatenate(
    [slots.reshape(3, 5, 7, 4, 8), jax.nn.one_hot(slot_mode, modes)], axis=-1
).reshape(3, 5, 7, 4 * (8 + modes))
params_m = head_m.init(key, emb, mask, slots_m)
dist_m = head_m.apply(params_m, emb, mask, slots_m)
many_m = jax.vmap(lambda k: dist_m.sample(seed=k))(jax.random.split(key, 500))
look = many_m[..., 3].astype(jnp.int32)
in_slots = (look[..., None] == slot_mode[None] + 1).any(-1) | (look == 0)
print("modes: looks in a slot's mode", bool(in_slots.all()), "values", sorted(set(look.ravel().tolist())))
assert bool(in_slots.all()) and not bool(look[:, 0, 0, 0].any())
lp_m = jax.vmap(dist_m.log_prob)(many_m)
print("modes: log_prob finite", bool(jnp.isfinite(lp_m).all()), "mode", dist_m.mode().shape)
g_m = jax.grad(lambda p: head_m.apply(p, emb, mask, slots_m).log_prob(many_m[0]).sum())(params_m)
print("modes: grad finite", all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(g_m)))
