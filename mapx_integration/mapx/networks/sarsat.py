# Copyright 2026 MAPX contributors.
# SPDX-License-Identifier: Apache-2.0

"""Networks for the CoopSarSat observation.

* :class:`SarSatTorso` reads the structured observation: one shared encoder over the
  ranked beam slots, one over the look-ahead sequence, and the remaining flat features
  (own status, plus the team summary when the input is the critic's global state).
* :class:`AnchoredHybridHead` is ``HybridActionHead`` with the pointing of slot 0 -- the
  beam worth the most right now -- added to the Gaussian mean (in pre-tanh space) and to
  the side logit. The network therefore learns a *correction* to a sensible aim rather
  than having to discover pointing from a sparse reward; whether to sense at all, and any
  departure from slot 0, is still entirely learned.
* :class:`AnchoredRecurrentActor` is ``RecurrentActor`` with a skip connection around the
  GRU and the anchor passed to the head.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal
from mapx.networks.base import ScannedRNN
from mapx.networks.distributions import HybridDistribution, TanhTransformedDistribution
from mapx.types import ActionLayout, DistributionLike, RNNMapxObservation

SLOT_DIM = 8


class SarSatTorso(nn.Module):
    num_slots: int = 4
    slot_dim: int = SLOT_DIM
    horizon: int = 16
    slot_features: int = 32
    ahead_features: int = 64
    layer_sizes: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        n_slot, n_ahead = self.slot_dim * self.num_slots, 2 * self.horizon
        lead = observation.shape[:-1]
        slots = observation[..., :n_slot].reshape(*lead, self.num_slots, self.slot_dim)
        ahead = observation[..., n_slot : n_slot + n_ahead]
        rest = observation[..., n_slot + n_ahead :]

        slots = nn.relu(nn.Dense(self.slot_features, kernel_init=orthogonal(jnp.sqrt(2)))(slots))
        ahead = nn.relu(nn.Dense(self.ahead_features, kernel_init=orthogonal(jnp.sqrt(2)))(ahead))
        x = jnp.concatenate([slots.reshape(*lead, -1), ahead, rest], axis=-1)
        for size in self.layer_sizes:
            x = nn.Dense(size, kernel_init=orthogonal(jnp.sqrt(2)))(x)
            x = nn.relu(nn.LayerNorm(use_scale=False)(x))
        return x


class AnchoredHybridHead(nn.Module):
    action_dim: int
    layout: ActionLayout
    slot_dim: int = SLOT_DIM  # unused: the anchor is the first four features of slot 0
    min_scale: float = 1e-3
    init_scale: float = 0.15
    side_anchor_logit: float = 3.0

    @nn.compact
    def __call__(
        self, embedding: jax.Array, action_mask: jax.Array, anchor: jax.Array
    ) -> HybridDistribution:
        """``anchor`` is slot 0 of the observation: ``(valid, incidence, squint, side)``."""
        valid = anchor[..., 0:1]
        aim = jnp.arctanh(jnp.clip(anchor[..., 1:3], -0.97, 0.97)) * valid
        loc = aim + nn.Dense(2, kernel_init=orthogonal(0.01))(embedding)
        raw = jnp.log(jnp.expm1(self.init_scale))  # softplus(raw) == init_scale
        log_std = self.param("log_std", nn.initializers.constant(raw), (2,))
        scale = jax.nn.softplus(log_std) * jnp.ones_like(loc) + self.min_scale
        continuous = tfd.Independent(
            TanhTransformedDistribution(tfd.Normal(loc=loc, scale=scale)),
            reinterpreted_batch_ndims=1,
        )

        categoricals = []
        for block, (slots, num_categories) in enumerate(self.layout.categorical_blocks):
            logits = nn.Dense(len(slots) * num_categories, kernel_init=orthogonal(0.01))(embedding)
            logits = logits.reshape(*embedding.shape[:-1], len(slots), num_categories)
            if block == 0:  # the look side: lean towards the anchor's side
                lean = self.side_anchor_logit * (2.0 * anchor[..., 3:4] - 1.0) * valid
                logits = logits.at[..., 1].add(lean)
            available = action_mask[..., jnp.asarray(slots)].astype(bool)[..., None]
            keep = available | (jnp.arange(num_categories) == 0)
            logits = jnp.where(keep, logits, jnp.finfo(jnp.float32).min)
            categoricals.append(
                (
                    tfd.Independent(tfd.Categorical(logits=logits), reinterpreted_batch_ndims=1),
                    len(slots),
                    num_categories,
                )
            )
        return HybridDistribution(continuous, categoricals, self.layout.part_slots)


class AnchoredRecurrentActor(nn.Module):
    pre_torso: nn.Module
    post_torso: nn.Module
    action_head: nn.Module
    hidden_state_dim: int = 128

    @nn.compact
    def __call__(
        self, policy_hidden_state: jax.Array, observation_done: RNNMapxObservation
    ) -> Tuple[jax.Array, DistributionLike]:
        observation, done = observation_done
        embedding = self.pre_torso(observation.agents_view)
        policy_hidden_state, memory = ScannedRNN(self.hidden_state_dim)(
            policy_hidden_state, (embedding, done)
        )
        embedding = self.post_torso(jnp.concatenate([embedding, memory], axis=-1))
        policy = self.action_head(
            embedding, observation.action_mask, observation.agents_view[..., :4]
        )
        return policy_hidden_state, policy


class SlotMixtureDistribution:
    """Policy over ``[incidence, squint, side, sense]`` that first picks one of the ranked
    beam slots (a latent choice, marginalised in ``log_prob``), then aims near it.

    Component ``k`` is a tanh-Gaussian around slot ``k``'s pointing times a Bernoulli that
    leans to slot ``k``'s side; sensing is an independent Bernoulli. The action vector and
    the environment are untouched: the mixture only gives the policy a way to say "take the
    second-best beam" (e.g. because a teammate will take the best) without having to learn
    a large pointing offset.
    """

    def __init__(
        self,
        mix_logits: jax.Array,  # (..., K)
        loc: jax.Array,  # (..., K, 2) pre-tanh means
        scale: jax.Array,  # (2,) or broadcastable to loc
        side_logits: jax.Array,  # (..., K) logit of looking right
        sense_logits: jax.Array,  # (..., 2), already masked
    ) -> None:
        self._mix = tfd.Categorical(logits=mix_logits)
        self._mix_logp = jax.nn.log_softmax(mix_logits, axis=-1)
        self._loc, self._scale = loc, scale
        self._pointing = tfd.Independent(
            TanhTransformedDistribution(tfd.Normal(loc=loc, scale=scale * jnp.ones_like(loc))),
            reinterpreted_batch_ndims=1,
        )
        self._side_logits = side_logits
        self._sense = tfd.Categorical(logits=sense_logits)

    def _pick(self, values: jax.Array, k: jax.Array) -> jax.Array:
        index = k.reshape(*k.shape, *([1] * (values.ndim - k.ndim)))
        return jnp.take_along_axis(values, index, axis=k.ndim).squeeze(k.ndim)

    def sample(self, seed: jax.Array) -> jax.Array:
        k_mix, k_point, k_side, k_sense = jax.random.split(seed, 4)
        k = self._mix.sample(seed=k_mix)
        loc = self._pick(self._loc, k)
        noise = jax.random.normal(k_point, loc.shape) * self._scale
        pointing = jnp.clip(jnp.tanh(loc + noise), -0.999999, 0.999999)
        side = jax.random.bernoulli(k_side, jax.nn.sigmoid(self._pick(self._side_logits, k)))
        sense = self._sense.sample(seed=k_sense)
        return jnp.concatenate(
            [pointing, side[..., None].astype(loc.dtype), sense[..., None].astype(loc.dtype)], -1
        )

    def log_prob(self, event: jax.Array) -> jax.Array:
        pointing = self._pointing.log_prob(event[..., None, 0:2])  # (..., K)
        right = event[..., 2:3] > 0.5
        side = -jax.nn.softplus(jnp.where(right, -self._side_logits, self._side_logits))
        mixture = jax.nn.logsumexp(self._mix_logp + pointing + side, axis=-1)
        sense = jnp.clip(jnp.round(event[..., 3]), 0, 1).astype(jnp.int32)
        return mixture + self._sense.log_prob(sense)

    def entropy(self, seed: jax.Array = None) -> jax.Array:
        """Single-sample estimate (the mixture has no closed form); logging only."""
        seed = jax.random.PRNGKey(0) if seed is None else seed
        return -self.log_prob(jax.lax.stop_gradient(self.sample(seed)))

    def mode(self) -> jax.Array:
        k = jnp.argmax(self._mix_logp, axis=-1)
        loc = self._pick(self._loc, k)
        side = self._pick(self._side_logits, k) > 0.0
        sense = self._sense.mode()
        return jnp.concatenate(
            [jnp.tanh(loc), side[..., None].astype(loc.dtype), sense[..., None].astype(loc.dtype)],
            -1,
        )


class SlotMixtureHead(nn.Module):
    """Head producing a :class:`SlotMixtureDistribution`; starts out as "aim at slot 0"."""

    action_dim: int
    layout: ActionLayout
    num_slots: int = 4
    slot_dim: int = SLOT_DIM
    min_scale: float = 1e-3
    init_scale: float = 0.15
    side_anchor_logit: float = 3.0
    slot0_bias: float = 3.0

    @nn.compact
    def __call__(
        self, embedding: jax.Array, action_mask: jax.Array, slots: jax.Array
    ) -> SlotMixtureDistribution:
        lead = embedding.shape[:-1]
        slots = slots.reshape(*lead, self.num_slots, self.slot_dim)
        valid = slots[..., 0]  # (..., K)
        aim = jnp.arctanh(jnp.clip(slots[..., 1:3], -0.97, 0.97)) * valid[..., None]
        correction = nn.Dense(2, kernel_init=orthogonal(0.01))(embedding)
        loc = aim + correction[..., None, :]
        raw = jnp.log(jnp.expm1(self.init_scale))
        scale = jax.nn.softplus(self.param("log_std", nn.initializers.constant(raw), (2,)))
        scale = scale + self.min_scale

        bias = jnp.zeros((self.num_slots,)).at[0].set(self.slot0_bias)
        mix = nn.Dense(self.num_slots, kernel_init=orthogonal(0.01))(embedding) + bias
        usable = (valid > 0.5).at[..., 0].set(True)  # slot 0 stands in when nothing is visible
        mix = jnp.where(usable, mix, jnp.finfo(jnp.float32).min)

        side = nn.Dense(1, kernel_init=orthogonal(0.01))(embedding)
        side = side + self.side_anchor_logit * (2.0 * slots[..., 3] - 1.0) * valid

        sense = nn.Dense(2, kernel_init=orthogonal(0.01))(embedding)
        keep = action_mask[..., 3:4].astype(bool) | (jnp.arange(2) == 0)
        sense = jnp.where(keep, sense, jnp.finfo(jnp.float32).min)
        return SlotMixtureDistribution(mix, loc, scale, side, sense)


class SlotMixtureActor(nn.Module):
    """:class:`AnchoredRecurrentActor` handing every beam slot to the head."""

    pre_torso: nn.Module
    post_torso: nn.Module
    action_head: nn.Module
    hidden_state_dim: int = 128

    @nn.compact
    def __call__(
        self, policy_hidden_state: jax.Array, observation_done: RNNMapxObservation
    ) -> Tuple[jax.Array, DistributionLike]:
        observation, done = observation_done
        embedding = self.pre_torso(observation.agents_view)
        policy_hidden_state, memory = ScannedRNN(self.hidden_state_dim)(
            policy_hidden_state, (embedding, done)
        )
        embedding = self.post_torso(jnp.concatenate([embedding, memory], axis=-1))
        num = self.action_head.slot_dim * self.action_head.num_slots
        policy = self.action_head(
            embedding, observation.action_mask, observation.agents_view[..., :num]
        )
        return policy_hidden_state, policy
