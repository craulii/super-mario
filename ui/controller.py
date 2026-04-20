"""Controller: único punto de acceso desde el server al Trainer y al SharedState."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from configs.schema import Config, RewardConfig, load_config
from training.shared_state import SharedState
from training.trainer import Trainer


class DashboardController:
    def __init__(self, default_config_path: Path | str = "configs/default.yaml"):
        self._cfg: Config = load_config(default_config_path)
        self._cfg.ensure_dirs()
        self._state = SharedState(reward_config=self._cfg.reward)
        self._trainer = Trainer(self._cfg, self._state)

    @property
    def state(self) -> SharedState:
        return self._state

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    @property
    def config(self) -> Config:
        return self._cfg

    # ---------- control de training ----------
    def start(self, config_path: str | None = None, resume_path: str | None = None) -> None:
        if config_path:
            new_cfg = load_config(config_path)
            new_cfg.ensure_dirs()
            # preservar la referencia compartida del RewardConfig
            for f in fields(RewardConfig):
                setattr(self._cfg.reward, f.name, getattr(new_cfg.reward, f.name))
            self._cfg.env = new_cfg.env
            self._cfg.ppo = new_cfg.ppo
            self._cfg.training = new_cfg.training
            self._cfg.paths = new_cfg.paths
        self._trainer.start(resume_path=Path(resume_path) if resume_path else None)

    def pause(self) -> None:
        self._trainer.pause()

    def resume(self) -> None:
        self._trainer.resume()

    def stop(self) -> None:
        self._trainer.stop(timeout=30)

    # ---------- save/load ----------
    def save(self, path: str | None) -> Path:
        target = Path(path) if path else self._cfg.paths.model_save_path
        return self._trainer.save(target)

    def load(self, path: str) -> dict[str, Any]:
        return self._trainer.load(Path(path))

    # ---------- config ----------
    def apply_reward_patch(self, patch: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Aplica cambios in-place sobre RewardConfig. Devuelve (applied, ignored).

        Propaga cambios a subprocesos via env_method si el training está activo.
        """
        applied: list[str] = []
        ignored: list[str] = []
        valid = {f.name for f in fields(RewardConfig)}
        filtered_patch: dict[str, Any] = {}
        with self._state.lock:
            for k, v in patch.items():
                if v is None:
                    continue
                if k in valid:
                    setattr(self._cfg.reward, k, v)
                    applied.append(k)
                    filtered_patch[k] = v
                else:
                    ignored.append(k)
        # Propagar a environments en subprocesos
        if filtered_patch and self._trainer.is_running():
            self._trainer.propagate_reward_config(filtered_patch)
        return applied, ignored

    def config_snapshot(self) -> dict[str, Any]:
        return self._cfg.to_dict()
