#!/usr/bin/env bash
# Sync the integration files into the MAPX checkout and launch a training run (WSL).
#   ./run.sh <ippo|mappo> <run-name> [hydra overrides...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mapx_root="${MAPX_ROOT:-$HOME/marl/mapx}"
cp -r "$here/mapx/." "$mapx_root/mapx/"
source "${VENV:-$HOME/marl/.venv}/bin/activate"
system="$1"; name="$2"; shift 2
out="${RUNS:-$HOME/marl/runs}/$name"
mkdir -p "$out"
cd "$out"
# Under WSL2 the BFC allocator cannot reserve more than about half the VRAM (neither up
# front nor by growing); the platform allocator reaches all of it, and a fully jitted
# learner makes few enough allocations that its cost does not matter.
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_CPP_MIN_LOG_LEVEL=2
exec python "$here/train_sarsat.py" "$system" system.seed=42 "$@" 2>&1 | tee "$out/train.log"
