"""Configuración tipada del proyecto. Cargada desde YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class EnvConfig:
    env_id: str = "SuperMarioBros-1-2-v0"
    frame_stack: int = 4
    resize_shape: int = 84
    skip_frames: int = 4


@dataclass
class PPOConfig:
    policy: str = "CnnPolicy"
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_linear_lr_schedule: bool = True


@dataclass
class RewardConfig:
    """Coeficientes de reward shaping. Editable en caliente desde el dashboard."""

    forward_reward_coef: float = 1.0
    backward_reward_coef: float = 0.5
    coin_reward: float = 5.0
    score_reward_coef: float = 0.025
    vertical_explore_reward: float = 0.5
    time_penalty_coef: float = -0.001
    time_remaining_bonus_coef: float = 0.1
    flag_bonus: float = 15.0

    death_by_enemy_penalty: float = -15.0
    death_by_time_penalty: float = -8.0

    stuck_penalty_base: float = -0.5
    stuck_threshold_base: int = 30

    enable_cyclic_detection: bool = True
    cyclic_action_penalty: float = -0.2
    cyclic_window: int = 40
    cyclic_min_repeats: int = 3
    cyclic_progress_threshold: int = 8

    enable_death_map: bool = True
    death_bucket_size: int = 32
    death_bucket_coef: float = -0.5
    death_bucket_cap: float = -10.0

    enable_excessive_left: bool = True
    excessive_left_threshold_frames: int = 30
    excessive_left_pixels: int = 40
    excessive_left_penalty: float = -0.1

    enable_micro_movement: bool = True
    micro_movement_window: int = 60
    micro_movement_span_threshold: int = 20
    micro_movement_net_threshold: int = 10
    micro_movement_penalty: float = -1.0

    enable_wall_stuck: bool = True
    wall_stuck_frames: int = 15
    wall_stuck_penalty: float = -0.3

    enable_records: bool = True
    record_distance_bonus: float = 5.0
    record_time_bonus: float = 10.0

    no_progress_limit: int = 200


@dataclass
class TrainingConfig:
    total_timesteps: int = 5_000_000
    checkpoint_freq: int = 50_000
    early_stop_consecutive: int = 5
    seed: int | None = 42
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    reset_memory: bool = False
    n_envs: int = 4


@dataclass
class PathsConfig:
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    model_save_path: Path = Path("models_saved/mario_ppo")
    video_dir: Path = Path("videos")
    death_map_path: Path = Path("checkpoints/death_map.json")
    best_distance_path: Path = Path("checkpoints/best_distance.json")
    best_time_path: Path = Path("checkpoints/best_clear_time.json")


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def ensure_dirs(self) -> None:
        for d in (
            self.paths.log_dir,
            self.paths.checkpoint_dir,
            self.paths.model_save_path.parent,
            self.paths.video_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)


def _dc_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dc_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_dc_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    return obj


def _apply_dict(target: Any, data: dict[str, Any]) -> None:
    """Aplica valores de un dict a un dataclass, convirtiendo Path cuando corresponde."""
    for f in fields(target):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(target, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict(current, value)
        elif isinstance(current, Path):
            setattr(target, f.name, Path(value))
        else:
            setattr(target, f.name, value)


def load_config(path: Path | str) -> Config:
    """Carga un Config desde un archivo YAML. Los campos ausentes usan los defaults."""
    cfg = Config()
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)
    return cfg


def dump_config(cfg: Config, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
