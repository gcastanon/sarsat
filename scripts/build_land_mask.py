"""Regenerate ``sarsat/land_mask_1deg.npy`` (one-off; not needed at runtime).

Downsamples the 30-arc-second NOAA GLOBE land/ocean mask shipped with the
``global-land-mask`` pip package to a 1-degree majority-vote grid and stores it
bit-packed (8.1 kB).  Row 0 is the cell [89N, 90N]; column 0 is [180W, 179W].
"""

import os

import global_land_mask
import numpy as np

src = os.path.join(os.path.dirname(global_land_mask.__file__), "globe_combined_mask_compressed.npz")
is_ocean = np.load(src)["mask"]  # (21600, 43200) bool, True = ocean
land_fraction = 1.0 - is_ocean.reshape(180, 120, 360, 120).mean(axis=(1, 3))
is_land = land_fraction > 0.5  # (180, 360)
out = os.path.join(os.path.dirname(__file__), "..", "sarsat", "land_mask_1deg.npy")
np.save(out, np.packbits(is_land))
print(f"land cells: {is_land.sum()} / {is_land.size}")
