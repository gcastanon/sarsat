"""Probe how much GPU memory JAX can get under WSL2: total in 1 GB chunks, then one block."""

import jax.numpy as jnp

held = []
try:
    for _ in range(16):
        x = jnp.zeros((1024, 1024, 256), jnp.float32)
        x.block_until_ready()
        held.append(x)
except Exception:
    pass
print("total 1GB chunks:", len(held), flush=True)
del held, x

for gb in [3, 4, 5, 6, 8, 10, 12]:
    try:
        x = jnp.zeros((gb * 1024, 1024, 256), jnp.float32)
        x.block_until_ready()
        print(gb, "GB ok", flush=True)
        del x
    except Exception:
        print(gb, "GB FAIL", flush=True)
        break
