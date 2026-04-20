"""Servidor de observación read-only.

Se engancha al mismo SharedState que usa la terminal app, exponiendo
métricas, frames y trayectorias vía WebSockets.  No tiene endpoints de
control (start/stop/pause) — solo lectura.

Uso desde app.py:
    from ui.observer import start_in_background
    start_in_background(state, cfg, port=8000)
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from configs.schema import Config
from env.level_metrics import X_MAX_LEVEL_1_2
from training.shared_state import SharedState
from utils.frame_encode import ndarray_to_jpeg_b64
from utils.logging_setup import get_logger

logger = get_logger("observer")

STATIC_DIR = Path(__file__).parent / "static"


def create_app(state: SharedState, cfg: Config) -> FastAPI:
    app = FastAPI(title="Mario RL Observer", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index():
        return HTMLResponse(
            (STATIC_DIR / "observer.html").read_text(encoding="utf-8")
        )

    @app.get("/status")
    async def status():
        snap = state.snapshot()
        snap["x_max_level"] = X_MAX_LEVEL_1_2
        snap["total_timesteps"] = cfg.training.total_timesteps
        snap["n_envs"] = cfg.training.n_envs
        snap["device"] = cfg.training.device
        snap["env_id"] = cfg.env.env_id
        snap["learning_rate"] = cfg.ppo.learning_rate
        return snap

    @app.websocket("/stream/metrics")
    async def ws_metrics(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                snap = state.snapshot()
                snap["x_max_level"] = X_MAX_LEVEL_1_2
                snap["total_timesteps"] = cfg.training.total_timesteps
                snap["n_envs"] = cfg.training.n_envs
                await ws.send_json(snap)
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, Exception):
            pass

    @app.websocket("/stream/frame")
    async def ws_frame(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                with state.lock:
                    frame = (
                        state.last_frame.copy()
                        if state.last_frame is not None
                        else None
                    )
                if frame is not None:
                    try:
                        b64 = ndarray_to_jpeg_b64(frame, quality=55)
                        await ws.send_text(b64)
                    except Exception:
                        pass
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, Exception):
            pass

    @app.websocket("/stream/trajectories")
    async def ws_trajectories(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                with state.lock:
                    trajs = [list(t) for t in state.trajectories]
                    current = list(state.current_trajectory)
                await ws.send_json(
                    {
                        "trajectories": trajs,
                        "current": current,
                        "x_max": X_MAX_LEVEL_1_2,
                    }
                )
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, Exception):
            pass

    return app


def start_in_background(
    state: SharedState, cfg: Config, port: int = 8000
) -> None:
    """Inicia el servidor de observación en un hilo daemon."""
    app = create_app(state, cfg)

    def _run():
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )

    thread = threading.Thread(target=_run, daemon=True, name="observer-server")
    thread.start()
    logger.info("dashboard en http://127.0.0.1:%d", port)
