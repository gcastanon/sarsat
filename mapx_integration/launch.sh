#!/usr/bin/env bash
# Launch a training run with the issue-23 recipe:  ./launch.sh <ippo|mappo> <run> <scenario> [overrides]
#   scenario: sarsat-100sat-hotspots | sarsat-200sat-events | sarsat-500sat-events |
#             sarsat-500sat-announced | ...
# Defaults: slot-mixture head, credit_mix 0.5, ranked contention, LR decay, 1500 updates,
# 32 envs (16 at 200 satellites, 4 at 500 for the WSL2 cap; 16 needs ~35 GB). Any hydra override may follow.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
system="$1"; run="$2"; scenario="$3"; shift 3
envs=32; episodes=32
case "$scenario" in *200sat*) envs=16; episodes=16;; *500sat*) envs=4; episodes=8;; esac
export SARSAT_HEAD="${SARSAT_HEAD:-mixture}"
exec bash "$here/run.sh" "$system" "$run" "env/scenario=$scenario" \
  "arch.num_envs=$envs" system.rollout_length=180 system.recurrent_chunk_size=30 \
  system.update_batch_size=1 system.num_updates=1500 arch.num_evaluation=50 \
  "arch.num_eval_episodes=$episodes" arch.evaluation_greedy=True arch.absolute_metric=False \
  system.gamma=0.995 system.policy_clip_eps=0.1 system.value_clip_eps=10.0 \
  system.actor_lr=3e-4 system.critic_lr=5e-4 system.decay_learning_rates=True \
  logger.checkpointing.save_model=True +env.credit_mix=0.5 +env.kwargs.ranked_contention=True "$@"
