"""Configuración centralizada de hiperparámetros y rutas del proyecto."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    """Hiperparámetros y configuración del agente Mario."""

    # --- Environment ---
    env_id: str = "SuperMarioBros-1-2-v0"
    frame_stack: int = 4
    resize_shape: int = 84

    # --- PPO ---
    policy: str = "CnnPolicy"
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02           # Mayor entropía → más exploración de rutas secretas
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_linear_lr_schedule: bool = True

    # --- Entrenamiento ---
    total_timesteps: int = 5_000_000  # Más steps para descubrir rutas secretas
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # --- Reward shaping ---
    death_penalty: float = -15.0
    flag_bonus: float = 15.0
    time_remaining_bonus_coef: float = 0.1   # Bonus por tiempo restante al completar (premia velocidad)
    time_penalty_coef: float = -0.001
    stuck_penalty: float = -0.5
    forward_reward_coef: float = 1.0
    coin_reward: float = 5.0                 # Bonus por cada moneda (incentiva explorar tuberías)
    score_reward_coef: float = 0.025         # Bonus proporcional al score del juego
    vertical_explore_reward: float = 0.5     # Bonus por explorar nuevas posiciones verticales (tuberías)

    # --- Callbacks ---
    checkpoint_freq: int = 50_000
    early_stop_consecutive: int = 5

    # --- Rutas ---
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    model_save_path: Path = Path("models/mario_ppo")
    video_dir: Path = Path("videos")

    def ensure_dirs(self) -> None:
        """Crea los directorios necesarios si no existen."""
        for d in (self.log_dir, self.checkpoint_dir, self.model_save_path.parent, self.video_dir):
            d.mkdir(parents=True, exist_ok=True)
