"""Endpoints WebSocket del dashboard."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ui.controller import DashboardController
from utils.frame_encode import ndarray_to_jpeg_b64

router = APIRouter()


def register_ws_routes(app, controller: DashboardController) -> None:
    @app.websocket("/stream/metrics")
    async def metrics_ws(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                snap = controller.state.snapshot()
                await ws.send_text(json.dumps(snap))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    @app.websocket("/stream/frame")
    async def frame_ws(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                frame = None
                with controller.state.lock:
                    if controller.state.last_frame is not None:
                        frame = controller.state.last_frame.copy()
                if frame is not None:
                    try:
                        b64 = ndarray_to_jpeg_b64(frame, quality=55)
                        await ws.send_text(b64)
                    except Exception:
                        pass
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    @app.websocket("/stream/trajectories")
    async def trajectories_ws(ws: WebSocket) -> None:
        await ws.accept()
        last_count = 0
        try:
            # envío inicial completo
            with controller.state.lock:
                initial = [list(t) for t in controller.state.trajectories]
                last_count = len(initial)
            await ws.send_text(json.dumps({"replace": True, "trajectories": initial}))

            while True:
                await asyncio.sleep(1.0)
                with controller.state.lock:
                    total = len(controller.state.trajectories)
                    new = [list(t) for t in controller.state.trajectories[last_count:]]
                if total < last_count:
                    with controller.state.lock:
                        all_t = [list(t) for t in controller.state.trajectories]
                    last_count = total
                    await ws.send_text(json.dumps({"replace": True, "trajectories": all_t}))
                elif new:
                    last_count = total
                    await ws.send_text(json.dumps({"append": True, "trajectories": new}))
        except WebSocketDisconnect:
            return
