#!/usr/bin/env bash
# Paired-seed evaluation of several runs' best checkpoints, one after another.
#   ./eval_runs.sh <ippo|mappo> <run> [runs...]      (JAX_PLATFORMS / SARSAT_HEAD from the env)
system="$1"; shift
here="$(cd "$(dirname "$0")" && pwd)"
runs="${RUNS:-$HOME/marl/runs}"
export SARSAT_HEAD="${SARSAT_HEAD:-mixture}"
for r in "$@"; do
  python "$here/eval_checkpoint.py" --run-dir "$runs/$r" --system "$system" \
    --seeds-start "${SEED_START:-1000}" --seeds-end "${SEED_END:-1015}" \
    --out "$runs/$r/eval_seeds${SEED_TAG:-}.json" 2>&1 |
    grep -E 'Restored|means|beats|Error|Traceback' | sed "s/^/$r /"
done
