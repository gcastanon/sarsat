# SarSat

Start with STATUS.md; design rationale is in DECISIONS.md, training in mapx_integration/README.md.

## Runpod
- Framework: JAX (GPU; the satellite-target geometry runs on the GPU, so runs are GPU-bound, not CPU-bound). Python 3.12.
- Setup: ship this repo and a MAPX checkout (git@github.com:Chulabhaya/mapx.git is private: tar `~/marl/mapx` from WSL, excluding `.git`) to `/workspace/{sarsat,mapx}`, then
  `uv venv /workspace/venv --python 3.12 && uv pip install "jax[cuda12]==0.11.2" flax==0.12.9 optax==0.2.8 chex==0.1.92 jumanji==1.1.2 hydra-core==1.4.0.dev10 omegaconf==2.4.0.dev15 orbax-checkpoint==0.12.4 tfp-nightly==0.26.0.dev20260921 numpy==2.5.3 absl-py==2.5.0 colorama==0.4.6 && uv pip install -e /workspace/mapx -e /workspace/sarsat` (about 2 minutes). Unpack with `tar --no-same-owner`.
- Train: `cd /workspace/sarsat/mapx_integration && MAPX_ROOT=/workspace/mapx VENV=/workspace/venv RUNS=/workspace/sarsat/runs XLA_PYTHON_CLIENT_ALLOCATOR=bfc ./launch.sh <ippo|mappo> <run> <scenario> [hydra overrides]`. Seed: `system.seed=<n>` (run.sh passes 42). Later overrides win, so `arch.num_envs=16` after the scenario replaces launch.sh's default.
- Launch detached: wrap the whole command in `nohup bash -c '...' > file 2>&1 < /dev/null &`, or the SSH session stays attached.
- Checkpoints: `runs/<run>/checkpoints/rec_<system>/<timestamp>/` (MAPX keeps the best by its own evaluation); log `runs/<run>/train.log`. Restoring needs the same `arch.num_envs`.
- Parallel envs: 32 at 100 satellites, 16 at 200 and 500 (Runpod has no WSL2 4 GB allocation cap); 180-step rollouts, 1500 updates.
- Measured: `sarsat-500sat-events`, 16 envs, 1x A40 48GB: 259 env-steps/s training, 423 evaluating (steady; the first segment includes ~3 min of compilation), ~6.4 min per 30 updates, plateau by update ~600 (~2.2 h, $1.19 all-in), GPU 100%, 34.5 GB VRAM, CPU ~2% (GPU-bound). Full checkpoints are 1.7 GB (they hold env states); `params_snapshot.py`-style extraction of params + opt_states is 5.4 MB.
- Early-stop criteria: stop a saturated run (user, 2026-09-23). Applied as: after 20 evaluations, the last 10 neither beat the earlier best by > 0.005 nor average > 0.003 above the previous 10. Snapshot and run the paired-seed evaluation before teardown.
- Notes: the pod's own `RUNPOD_API_KEY` returned Unauthorized for `runpodctl get pod`, so the on-pod self-destruct is unverified; enforce caps from the PC too.
