"""Probe how much GPU memory JAX can get under WSL2: total in 1 GB chunks, then one block.

One-off diagnostic: it is what established the ~4 GB single-allocation cap under WSL2's
default BFC allocator recorded in DECISIONS.md issue 23 (worked around there with
``XLA_PYTHON_CLIENT_ALLOCATOR=platform``). Reads and writes nothing; prints pass/fail per
chunk size to stdout.

Run: ``python scripts/gpu_alloc_probe.py``. Seconds; GPU only, meaningless on CPU.
"""

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
