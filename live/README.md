# SarSat live

A web UI in which the trained MAPPO actor runs **forever** over the 500-satellite event
scenario at 60x real time (one 60 s environment step per wall-clock second), and where a
typed request such as *"image the port of Rotterdam, high priority, within 20 minutes"*
becomes a located, prioritised, time-windowed target the agents try to collect. Nothing
calls a cloud API: the language model (optional) and the gazetteer run locally.

## Run it

Windows, from the repo root (a Python 3.12 venv with CPU JAX, flax and the MAPX checkout;
`.claude/launch.json` starts the same thing from the Claude desktop app):

```bash
live/.venv/Scripts/python -m live.server --port 8000
```

then open <http://localhost:8000>. Flags: `--speed` (sim seconds per wall second, 60 by
default, also changeable in the page), `--seed`, `--params` (a `params_<step>.msgpack`
snapshot, default `runs/ev500_mappo_a/sync/params_1728000.msgpack`), `--boundary-battery
carry|full`, the difficulty knobs `--hotspots`, `--window MIN MAX`, `--hotspot-targets`,
`--hotspot-radius-km`, `--recharge-rate` (defaults are the training scenario), and for
natural language `--gazetteer DIR` and `--model FILE.gguf` (below).

Setting up the venv (once):

```bash
pip install uv && uv venv live/.venv --python 3.12
uv pip install --python live/.venv/Scripts/python.exe jax==0.11.2 flax==0.12.9 optax chex jumanji==1.1.2 \
  hydra-core==1.4.0.dev10 omegaconf==2.4.0.dev15 tfp-nightly==0.26.0.dev20260921 numpy fastapi "uvicorn[standard]" rapidfuzz pytest ruff
uv pip install --python live/.venv/Scripts/python.exe --no-deps -e <path to the MAPX checkout> -e .
```

## How it works

* **`sarsat/live.py` `LiveWorld`** keeps a continuous world over the episodic
  `WindowedCoopSarSat` as *rolling episodes*: every 180 steps the world is re-packed into a
  fresh state (orbits re-based, batteries carried over, captured background targets
  respawned, a fresh draw of 100 events with the training rules) and the policy's GRU carry
  is reset, so inside an episode the policy sees exactly what it saw in training. The next
  episode's cluster schedule is built a few rows per step, so the boundary costs one
  geometry refresh. Requests take over an event cluster whose window has closed: all 50 of
  its slots go to the requested point (the first exactly there, the rest scattered by 12-20
  km), with a window of at most 30 steps, clipped at the episode end and carried into the
  next episode. DECISIONS.md issue 28.
* **`live/policy.py`** rebuilds the MAPX actor from constants (no hydra, no orbax) and loads
  the 5.4 MB msgpack snapshot; `python -m live.policy` replays seed 1000 and checks the
  return against `eval_seeds.json` (0.8551 here vs 0.8550 on the A40).
* **`live/server.py`** steps the world in a thread on a monotonic clock and streams one
  ~15 KB frame per step over a WebSocket (`/stream`); the full target table goes out once
  per episode. `POST /parse` previews a request, `POST /request` injects it, `/snapshot`
  serves late joiners. Feasibility (which satellites can reach the point in the window, on
  geometry and a battery forecast) is reported at submission; the policy decides whether it
  is actually taken.
* **`live/ui/index.html`** is the map viewer from `sarsat/viewer.html` made live, plus the
  request box, a ledger and a stats strip. Clicking the map fills in coordinates.
* **`live/nl/`** parses requests in three optional stages: rules (coordinates, priority
  words, deadlines: these two fields come from the rules alone), an offline GeoNames
  gazetteer that picks the place out of the sentence (longest exact name, cities before
  countries, a strict spelling-tolerant match only on a short residue), and a small local
  instruct model under a JSON-schema grammar (llama.cpp) as the fallback that names a place
  the sentence does not spell out ("the biggest Dutch port" -> Rotterdam). The model's own
  coordinates are used only when the gazetteer cannot geocode its answer, and are flagged.

## Natural language: data and model (optional, both local)

* **Gazetteer:** download from <https://download.geonames.org/export/dump/> (CC BY 4.0)
  into `data/geonames/`: `cities15000.zip` (unzip to `cities15000.txt`, ~10 MB),
  `countryInfo.txt`, `admin1CodesASCII.txt`. Then `--gazetteer data/geonames`.
* **Model:** any GGUF chat model; `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~1 GB, from the
  Qwen Hugging Face repo) answers in a few seconds on CPU. Install
  `llama-cpp-python` (`uv pip install --python live/.venv/Scripts/python.exe llama-cpp-python
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`) and pass
  `--model models/<file>.gguf`.
* **Evaluate:** `python -m live.nl.evaluate --gazetteer data/geonames [--model ...]
  [--model-first]` scores a synthetic set (templates x gazetteer cities, 15% with "N km
  east of" offsets) and 15 hand-written phrases: location within 50 km, priority and
  deadline exact. Measured 2026-09-24 (GeoNames cities15000, Qwen2.5-1.5B-Instruct Q4_K_M):

  | path | hand-written (15) | synthetic (n) |
  |---|---|---|
  | rules + gazetteer (the default) | 100% | 96.7% (300); every miss is a homonym, e.g. Birmingham UK vs Alabama |
  | model first, then gazetteer (`--model-first`) | 100% | 95% (100), 2.2 s per request on 8 CPU threads |

  A first version that let the model set priority and deadline scored 67%: it invents a
  priority when none is stated and drops offsets. Hence rules own those fields and the
  model only names places. No fine-tuning was needed at this accuracy; `live/nl/llm.py`
  takes any GGUF if a larger model is wanted.

## Soak results (2026-09-24, `python -m live.soak --episodes 3 --requests 10`)

Windows CPU, 500 satellites: 83 ms per step (the budget at 60x is 1 s). All 30 random-land
requests were collected, most on the first feasible pass. Episode returns 0.930 (episode 0,
seed 1000, includes the requests' priority), then 0.822 and 0.814: with `--boundary-battery
carry` the policy ends each episode with the fleet at ~8% charge (it spends everything at a
5% duty cycle), so later episodes start nearly empty and lose ~4% against the offline 0.868;
`--boundary-battery full` restores the training start.
