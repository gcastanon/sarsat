"""Write the actor params of a run's latest (best) checkpoint as a small msgpack snapshot.

MAPX's orbax checkpoints hold the whole learner state, environment states included (about
280 MB at 100 satellites, 1.7 GB at 500); the actor params alone are a few MB. This writes
``<run-dir>/sync/params_<step>.msgpack`` with the tree ``{params: {actor_params: ...}}``,
which ``sarsat_live.policy.load_actor_params`` reads.

Usage: ``python params_snapshot.py <run-dir> [<run-dir> ...]`` (CPU is fine).
"""

import os
import sys

# The ``mapx/`` overlay next to this script would shadow the installed package.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from flax import serialization  # noqa: E402

sys.path.append(_here)
from eval_checkpoint import find_checkpoint_dir, restore_actor_params  # noqa: E402


def snapshot(run_dir: str, model_name: str = "rec_mappo") -> str:
    params, step = restore_actor_params(find_checkpoint_dir(run_dir, model_name))
    out_dir = os.path.join(run_dir, "sync")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"params_{step}.msgpack")
    with open(path, "wb") as f:
        f.write(serialization.msgpack_serialize({"params": {"actor_params": params}}))
    return path


if __name__ == "__main__":
    for run in sys.argv[1:]:
        path = snapshot(run)
        print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB)")
