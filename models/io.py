"""Guardado y carga de modelos PPO con metadata JSON acompañante."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO


def _meta_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def save_model(model: PPO, path: Path | str, metadata: dict[str, Any] | None = None) -> Path:
    """Guarda el modelo en `.zip` + un `.meta.json` hermano con metadatos."""
    path = Path(path)
    if path.suffix == ".zip":
        base = path.with_suffix("")
    else:
        base = path
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(base))

    meta = {
        "timestamp": time.time(),
        "timesteps": int(getattr(model, "num_timesteps", 0)),
    }
    if metadata:
        meta.update(metadata)
    _meta_path(base).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return base.with_suffix(".zip")


def load_model(path: Path | str, env=None, device: str = "auto") -> tuple[PPO, dict[str, Any]]:
    """Carga el modelo. Retorna (PPO, metadata_dict)."""
    path = Path(path)
    if path.suffix == ".zip":
        base = path.with_suffix("")
    else:
        base = path

    model = PPO.load(str(base), env=env, device=device)
    meta: dict[str, Any] = {}
    mp = _meta_path(base)
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    return model, meta
