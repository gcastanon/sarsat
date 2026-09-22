"""Random ground targets with a fixed land / water split.

Land is defined by a 1-degree majority-vote mask (8 kB, bit-packed) derived from the
NOAA GLOBE data set; see ``scripts/build_land_mask.py`` and DECISIONS.md issue 9.
"""

from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

_MASK_FILE = Path(__file__).with_name("land_mask_1deg.npy")


class TargetSampler:
    """Samples points uniformly by area over the land cells or over the water cells."""

    def __init__(self) -> None:
        # Row 0 is the cell [89N, 90N]; column 0 is the cell [180W, 179W].
        is_land = np.unpackbits(np.load(_MASK_FILE)).reshape(180, 360).astype(bool)
        cell_area = np.cos(np.deg2rad(89.5 - np.arange(180)))[:, None] * np.ones((1, 360))

        def cells_and_cdf(mask: np.ndarray) -> Tuple[jax.Array, jax.Array]:
            cells = np.flatnonzero(mask)
            cdf = np.cumsum(cell_area.ravel()[cells], dtype=np.float64)
            return jnp.asarray(cells, jnp.int32), jnp.asarray(cdf / cdf[-1], jnp.float32)

        self._land_cells, self._land_cdf = cells_and_cdf(is_land)
        self._water_cells, self._water_cdf = cells_and_cdf(~is_land)

    def __call__(self, key: jax.Array, on_land: jax.Array) -> jax.Array:
        """Return ``(M, 2)`` latitude / longitude in degrees; ``on_land`` is an ``(M,)`` bool."""
        k_cell, k_jitter = jax.random.split(key)
        u = jax.random.uniform(k_cell, on_land.shape)
        # method="sort" gives the same indices as the default binary search but runs in
        # parallel on an accelerator, where the sequential scan dominated the cost of reset.
        cell = jnp.where(
            on_land,
            self._land_cells[jnp.searchsorted(self._land_cdf, u, method="sort")],
            self._water_cells[jnp.searchsorted(self._water_cdf, u, method="sort")],
        )
        jitter = jax.random.uniform(k_jitter, (*on_land.shape, 2))
        lat = 90.0 - (cell // 360 + jitter[:, 0])
        lon = -180.0 + (cell % 360 + jitter[:, 1])
        return jnp.stack([lat, lon], axis=-1)


def latlon_to_unit_vector(latlon_deg: jax.Array) -> jax.Array:
    """Convert ``(..., 2)`` latitude / longitude [deg] to Earth-fixed unit vectors."""
    lat, lon = jnp.deg2rad(latlon_deg[..., 0]), jnp.deg2rad(latlon_deg[..., 1])
    return jnp.stack(
        [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)], axis=-1
    )
