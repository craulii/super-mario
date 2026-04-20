"""FastAPI app para el dashboard local.

Levantar con:
    uvicorn ui.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from env.level_metrics import X_MAX_LEVEL_1_2
from ui.controller import DashboardController
from ui.schemas import (
    ActionResponse,
    ConfigPatch,
    ConfigPatchResponse,
    SavePathRequest,
    StartRequest,
    StatusResponse,
)
from ui.websockets import register_ws_routes

controller = DashboardController()

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    if controller.trainer.is_running():
        controller.trainer.stop(timeout=10)


app = FastAPI(title="Mario RL Dashboard", lifespan=_lifespan)
register_ws_routes(app, controller)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    snap = controller.state.snapshot()
    return StatusResponse(**snap)


@app.get("/config")
async def get_config():
    return {
        "config": controller.config_snapshot(),
        "x_max_level": X_MAX_LEVEL_1_2,
    }


@app.put("/config", response_model=ConfigPatchResponse)
async def put_config(patch: ConfigPatch) -> ConfigPatchResponse:
    applied: list[str] = []
    ignored: list[str] = []
    if patch.reward is not None:
        a, i = controller.apply_reward_patch(patch.reward.model_dump(exclude_none=True))
        applied.extend(a)
        ignored.extend(i)
    return ConfigPatchResponse(applied=applied, ignored=ignored, requires_restart=[])


@app.post("/train/start", response_model=ActionResponse)
async def train_start(req: StartRequest) -> ActionResponse:
    if controller.trainer.is_running():
        raise HTTPException(status_code=409, detail="ya hay un training activo")
    controller.start(config_path=req.config_path, resume_path=req.resume_path)
    return ActionResponse(ok=True, message="entrenamiento iniciado")


@app.post("/train/pause", response_model=ActionResponse)
async def train_pause() -> ActionResponse:
    controller.pause()
    return ActionResponse(ok=True, message="pausado")


@app.post("/train/resume", response_model=ActionResponse)
async def train_resume() -> ActionResponse:
    controller.resume()
    return ActionResponse(ok=True, message="reanudado")


@app.post("/train/stop", response_model=ActionResponse)
async def train_stop() -> ActionResponse:
    controller.stop()
    return ActionResponse(ok=True, message="detenido")


@app.post("/model/save", response_model=ActionResponse)
async def model_save(req: SavePathRequest) -> ActionResponse:
    try:
        final = controller.save(req.path)
        return ActionResponse(ok=True, message=str(final), data={"path": str(final)})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/model/load", response_model=ActionResponse)
async def model_load(req: SavePathRequest) -> ActionResponse:
    try:
        meta = controller.load(req.path)
        return ActionResponse(ok=True, message="cargado", data={"metadata": meta})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
