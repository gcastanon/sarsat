"""The live SarSat server: the world and the trained actor stepping in a thread at 60x real
time, streamed to browsers over a WebSocket, with natural-language tasking requests.

    PYTHONPATH=. live/.venv/Scripts/python -m live.server --port 8000

Endpoints:

* ``GET /``            the live viewer (``live/ui/index.html``)
* ``WS /stream``       a snapshot on connect, then one frame per step
* ``GET /snapshot``    the same snapshot as JSON (targets, recent frames, requests, stats)
* ``POST /parse``      ``{"text"}`` -> how the request would be understood, without submitting
* ``POST /request``    ``{"text"}`` or ``{"lat", "lon", "priority", "deadline_minutes"}`` ->
                       the request as injected, with its feasibility
* ``GET /requests``    the ledger
* ``POST /speed``      ``{"speed"}`` sim seconds per wall second (60 = one step per second)
"""

from __future__ import annotations

import os
import sys

# The sim runs on CPU JAX (a GPU may be busy training), from the repo root.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
import collections
import json
import math
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from live import nl
from live.policy import DEFAULT_PARAMS, load_policy, make_env
from sarsat.live import MAX_DEADLINE_STEPS, LiveWorld, Request, StepResult

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "ui", "index.html")
RECENT_FRAMES = 30
ROLLING_EVENTS = 100


def _round(a, nd=2):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _clean(points: np.ndarray) -> Optional[List[List[float]]]:
    """Lat/lon rows with NaN (a beam corner that misses the Earth) -> null."""
    if not np.isfinite(points).all():
        return None
    return _round(points)


class Simulation(threading.Thread):
    """Steps the world on a monotonic clock and hands frames to the event loop."""

    def __init__(self, world: LiveWorld, policy, speed: float, loop: asyncio.AbstractEventLoop):
        super().__init__(name="sarsat-sim", daemon=True)
        self.world, self.policy, self.loop = world, policy, loop
        self.speed = float(speed)
        self.stop = threading.Event()
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.clients: Dict[int, asyncio.Queue] = {}
        self.recent: collections.deque = collections.deque(maxlen=RECENT_FRAMES)
        self.episode_payload: Dict[str, Any] = {}
        self.cluster_payloads: Dict[int, Dict[str, Any]] = {}  # injected this episode
        self.episode_return = 0.0
        self.steps_per_s = 0.0
        self.started = time.time()
        self.step_seconds = float(world.env.step_seconds)
        self.stats_cache: Dict[str, Any] = {}
        self._set_episode_payload()

    # ---------------------------------------------------------------- commands
    def submit(self, fn) -> Future:
        fut: Future = Future()
        self.commands.put((fn, fut))
        return fut

    def _drain(self) -> None:
        while True:
            try:
                fn, fut = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn())
            except Exception as err:  # reported to the caller
                fut.set_exception(err)

    # ---------------------------------------------------------------- main loop
    def run(self) -> None:
        self.policy.reset()
        period = self.step_seconds / self.speed
        next_tick = time.monotonic()
        last = time.monotonic()
        while not self.stop.is_set():
            self._drain()
            t0 = time.monotonic()
            action = self.policy(self.world.obs)
            result = self.world.step(action)
            if result.new_episode:
                self.policy.reset()
            self._publish_step(result, time.monotonic() - t0)
            if result.new_episode:
                with self.lock:
                    self.cluster_payloads = {}
                    self._set_episode_payload()
                self._broadcast(self.episode_payload)
            now = time.monotonic()
            self.steps_per_s = (
                0.9 * self.steps_per_s + 0.1 / max(now - last, 1e-6)
                if self.steps_per_s
                else 1 / max(now - last, 1e-6)
            )
            last = now
            period = self.step_seconds / self.speed
            next_tick += period
            delay = next_tick - now
            if delay > 0:
                self.stop.wait(delay)
            elif delay < -3 * period:  # fell far behind (compilation, a pause): resync
                next_tick = now

    # ---------------------------------------------------------------- payloads
    def _set_episode_payload(self) -> None:
        w = self.world
        table = w.target_table()
        t0 = w.episode * w.T
        window = table["window"].astype(int)
        self.episode_payload = {
            "type": "episode",
            "episode": w.episode,
            "t0": t0,
            "T": w.T,
            "N": w.N,
            "M": w.M,
            "C": w.C,
            "K": w.K,
            "step_seconds": self.step_seconds,
            "lat": _round(table["latlon"][:, 0]),
            "lon": _round(table["latlon"][:, 1]),
            "cluster": table["cluster"].astype(int).tolist(),
            "weight": _round(table["weight"], 2),
            "open": (window[:, 0] + t0).tolist(),
            "close": (window[:, 1] + t0).tolist(),
            "active": table["active"].astype(int).tolist(),
        }
        self.episode_return = 0.0

    def _cluster_payload(self, r: Request) -> Dict[str, Any]:
        w = self.world
        table = w.target_table()
        slots = r.cluster + w.C * np.arange(w.K)
        t0 = w.episode * w.T
        window = table["window"][slots[0]].astype(int)
        return {
            "type": "cluster",
            "episode": w.episode,
            "cluster": int(r.cluster),
            "slots": slots.tolist(),
            "lat": _round(table["latlon"][slots, 0]),
            "lon": _round(table["latlon"][slots, 1]),
            "weight": float(np.round(table["weight"][slots[0]], 2)),
            "open": int(window[0] + t0),
            "close": int(window[1] + t0),
            "request": r.id,
        }

    def stats(self) -> Dict[str, Any]:
        w = self.world
        closed = w.events[-ROLLING_EVENTS:]
        done = [r for r in w.requests if r.status != "pending"]
        collected = [r for r in done if r.status == "collected"]
        waits = [r.collected_global - r.submitted_global for r in collected]
        return {
            "episode": w.episode,
            "global_step": w.global_step,
            "sim_s": w.global_step * self.step_seconds,
            "speed": self.speed,
            "steps_per_s": round(self.steps_per_s, 2),
            "episode_return": round(self.episode_return, 4),
            "episode_returns": [round(x, 4) for x in w.episode_returns[-10:]],
            "events_closed": len(w.events),
            "events_imaged_mean": round(float(np.mean([e.imaged_fraction for e in closed])), 3)
            if closed
            else None,
            "requests_total": len(w.requests),
            "requests_pending": len(w.requests) - len(done),
            "requests_collected": len(collected),
            "requests_missed": len(done) - len(collected),
            "request_wait_mean_steps": round(float(np.mean(waits)), 1) if waits else None,
            "battery_mean": round(float(np.mean(np.asarray(w.state.battery))), 3),
            "uptime_s": round(time.time() - self.started),
        }

    def _publish_step(self, res: StepResult, compute_s: float) -> None:
        self.episode_return += res.reward
        looks = []
        for sat, corners in res.footprints.items():
            looks.append(
                {
                    "sat": sat,
                    "c": _clean(corners),
                    "b": _clean(res.boresight[sat]),
                    "hit": int(bool((res.captures[:, 1] == sat).any())),
                }
            )
        frame = {
            "type": "step",
            "episode": res.episode,
            "step": res.step,
            "global_step": res.global_step,
            "sim_s": res.global_step * self.step_seconds,
            "reward": round(res.reward, 5),
            "compute_ms": round(compute_s * 1e3),
            "sats": _round(res.sat_latlon),
            "battery": _round(res.battery),
            "looks": looks,
            "captures": res.captures.astype(int).tolist(),
            "opened": res.events_opened,
            "closed": [
                {"cluster": e.cluster, "imaged": round(e.imaged_fraction, 3)}
                for e in res.events_closed
            ],
            "requests": [r.to_dict() for r in res.request_updates],
            "new_episode": res.new_episode,
        }
        frame["stats"] = self.stats()
        with self.lock:
            self.recent.append(frame)
        self._broadcast(frame)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "type": "snapshot",
                "episode": self.episode_payload,
                "clusters": list(self.cluster_payloads.values()),
                "frames": list(self.recent),
                "requests": [r.to_dict() for r in self.world.requests],
                "stats": self.stats(),
                "config": {
                    "speed": self.speed,
                    "max_deadline_minutes": MAX_DEADLINE_STEPS * self.step_seconds / 60,
                },
            }

    # ---------------------------------------------------------------- clients
    def _broadcast(self, message: Dict[str, Any]) -> None:
        text = json.dumps(message)
        self.loop.call_soon_threadsafe(self._fanout, text)

    def _fanout(self, text: str) -> None:
        for q in list(self.clients.values()):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(text)

    # ---------------------------------------------------------------- requests
    def inject(self, parsed: nl.Parsed) -> Dict[str, Any]:
        """Runs in the sim thread between steps."""
        deadline_steps = round(parsed.deadline_minutes * 60 / self.step_seconds)
        r = self.world.inject_request(
            parsed.lat,
            parsed.lon,
            parsed.priority,
            deadline_steps,
            text=parsed.text,
            place=parsed.place,
        )
        payload = self._cluster_payload(r)
        with self.lock:
            self.cluster_payloads[r.cluster] = payload
        self._broadcast(payload)
        self._broadcast({"type": "request", "request": r.to_dict()})
        return {"request": r.to_dict(), "parsed": parsed.to_dict(), "cluster": payload}

    def feasibility(self, lat: float, lon: float, deadline_minutes: float) -> Dict[str, Any]:
        steps = round(deadline_minutes * 60 / self.step_seconds)
        return self.world.feasibility(lat, lon, steps)


def build_app(sim: Simulation) -> FastAPI:
    app = FastAPI(title="SarSat live")

    @app.get("/")
    async def index():
        return FileResponse(UI)

    @app.get("/snapshot")
    async def snapshot():
        return JSONResponse(sim.snapshot())

    @app.get("/requests")
    async def requests():
        return JSONResponse([r.to_dict() for r in sim.world.requests])

    async def _parse(body: Dict[str, Any]) -> nl.Parsed:
        if body.get("lat") is not None and body.get("lon") is not None:
            p = nl.Parsed(
                text=str(body.get("text", "")),
                lat=float(body["lat"]),
                lon=float(body["lon"]),
                place=str(body.get("place", "")),
                priority=nl.schema.normalise_priority(body.get("priority")),
                deadline_minutes=nl.schema.clamp_deadline(body.get("deadline_minutes")),
                source="coordinates",
                confidence=1.0,
            )
            return p
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "give 'text' or 'lat' and 'lon'")
        loop = asyncio.get_running_loop()
        p = await loop.run_in_executor(None, nl.parse, text)
        # Explicit fields beside the text override what was read from it.
        if body.get("priority"):
            p.priority = nl.schema.normalise_priority(body["priority"])
        if body.get("deadline_minutes") is not None:
            p.deadline_minutes = nl.schema.clamp_deadline(body["deadline_minutes"])
        return p

    @app.post("/parse")
    async def parse(body: Dict[str, Any]):
        p = await _parse(body)
        out = p.to_dict()
        if p.located:
            fut = sim.submit(lambda: sim.feasibility(p.lat, p.lon, p.deadline_minutes))
            out["feasibility"] = await asyncio.wrap_future(fut)
        return JSONResponse(out)

    @app.post("/request")
    async def request(body: Dict[str, Any]):
        p = await _parse(body)
        if not p.located:
            raise HTTPException(422, {"error": "no location", "parsed": p.to_dict()})
        if not (-90 <= p.lat <= 90 and -180 <= p.lon <= 180):
            raise HTTPException(422, "coordinates out of range")
        fut = sim.submit(lambda: sim.inject(p))
        try:
            return JSONResponse(await asyncio.wrap_future(fut))
        except RuntimeError as err:
            raise HTTPException(503, str(err)) from err

    @app.post("/speed")
    async def speed(body: Dict[str, Any]):
        value = float(body.get("speed", 60))
        if not (1 <= value <= 3600) or math.isnan(value):
            raise HTTPException(400, "speed must be in [1, 3600]")
        sim.speed = value
        return JSONResponse({"speed": sim.speed})

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        key = id(q)
        sim.clients[key] = q
        try:
            await ws.send_text(json.dumps(sim.snapshot()))
            while True:
                await ws.send_text(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sim.clients.pop(key, None)

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--speed", type=float, default=60.0, help="sim seconds per wall second")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--boundary-battery", choices=("carry", "full"), default="carry")
    ap.add_argument("--hotspots", type=int, help="event clusters per episode (training: 100)")
    ap.add_argument(
        "--window", type=int, nargs=2, metavar=("MIN", "MAX"), help="event window steps"
    )
    ap.add_argument("--hotspot-targets", type=int, help="targets per event (training: 50)")
    ap.add_argument("--hotspot-radius-km", type=float, help="event scatter (training: 100)")
    ap.add_argument("--recharge-rate", type=float, help="battery per step (training: 0.005)")
    ap.add_argument("--model", default=os.environ.get("SARSAT_NL_MODEL"), help="GGUF path")
    ap.add_argument("--gazetteer", default=os.environ.get("SARSAT_GAZETTEER"), help="GeoNames dir")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    overrides: Dict[str, Any] = {}
    if args.hotspots is not None:
        overrides["hotspots"] = args.hotspots
    if args.window is not None:
        overrides["window_steps"] = tuple(args.window)
    if args.hotspot_targets is not None:
        overrides["hotspot_targets"] = args.hotspot_targets
    if args.hotspot_radius_km is not None:
        overrides["hotspot_radius_km"] = args.hotspot_radius_km
    if args.recharge_rate is not None:
        overrides["recharge_rate"] = args.recharge_rate
    env = make_env(**overrides)
    policy = load_policy(args.params, env)
    world = LiveWorld(env, seed=args.seed, boundary_battery=args.boundary_battery)

    geocoder = model = None
    if args.gazetteer:
        from live.nl.geocode import Gazetteer

        geocoder = Gazetteer(args.gazetteer)
    if args.model:
        from live.nl.llm import LocalModel

        model = LocalModel(args.model, n_threads=args.threads)
    nl.configure(geocoder=geocoder, model=model)
    print(
        f"world: {world.N} satellites, {world.M} targets, {world.C} events/episode; "
        f"NL: rules{' + gazetteer' if geocoder else ''}{' + model' if model else ''}"
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sim = Simulation(world, policy, args.speed, loop)
    app = build_app(sim)
    sim.start()
    config = uvicorn.Config(app, host=args.host, port=args.port, loop="asyncio", log_level="info")
    server = uvicorn.Server(config)
    try:
        loop.run_until_complete(server.serve())
    finally:
        sim.stop.set()


if __name__ == "__main__":
    main()
