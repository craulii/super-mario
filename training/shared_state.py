"""Estado compartido entre el thread de entrenamiento y la UI/terminal.

El RewardConfig vive aquí mutable in-place: los wrappers lo referencian directamente,
así que cambios desde la UI se propagan al próximo env.step() sin restart.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from configs.schema import RewardConfig

Mode = Literal["idle", "training", "paused", "evaluating", "stopping"]

MAX_TRAJECTORIES = 50


@dataclass
class SharedState:
    """Objeto vivo leído por la UI/terminal y escrito por el training loop."""

    reward_config: RewardConfig = field(default_factory=RewardConfig)

    lock: threading.Lock = field(default_factory=threading.Lock)
    # pause_event.is_set() => debe pausar; resume_event.is_set() => puede continuar
    pause_event: threading.Event = field(default_factory=threading.Event)
    resume_event: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)

    mode: Mode = "idle"
    message: str = ""

    training_start_time: float = 0.0
    timesteps: int = 0
    episodes: int = 0
    current_x: int = 0
    max_x_this_ep: int = 0
    max_x_historical: float = 0.0

    reward_avg_100: float = 0.0
    distance_avg_100: float = 0.0
    clear_rate_100: float = 0.0
    coins_avg_100: float = 0.0
    behavior_score: float = 0.0

    last_episode_reward: float = 0.0
    last_clear: bool = False
    zone_survival: dict[int, float] = field(default_factory=dict)

    last_frame: np.ndarray | None = None

    # Demo mode
    demo_episode: int = 0
    demo_score: int = 0
    demo_coins: int = 0
    demo_time: int = 400
    demo_flag: bool = False
    demo_results: list[dict] = field(default_factory=list)
    current_trajectory: list[tuple[int, int]] = field(default_factory=list)
    trajectories: list[list[tuple[int, int]]] = field(default_factory=list)

    metrics_history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # resume_event arranca set (permiso para correr)
        self.resume_event.set()
        self.pause_event.clear()
        self.stop_event.clear()

    # ---- control de estado ----
    def request_pause(self) -> None:
        self.pause_event.set()
        self.resume_event.clear()

    def request_resume(self) -> None:
        self.pause_event.clear()
        self.resume_event.set()

    def request_stop(self) -> None:
        self.stop_event.set()
        self.resume_event.set()  # liberar wait si estaba pausado

    def reset_for_new_run(self) -> None:
        with self.lock:
            self.training_start_time = 0.0
            self.timesteps = 0
            self.episodes = 0
            self.current_x = 0
            self.max_x_this_ep = 0
            self.reward_avg_100 = 0.0
            self.distance_avg_100 = 0.0
            self.clear_rate_100 = 0.0
            self.coins_avg_100 = 0.0
            self.behavior_score = 0.0
            self.last_episode_reward = 0.0
            self.last_clear = False
            self.last_frame = None
            self.current_trajectory.clear()
            self.trajectories.clear()
            self.metrics_history.clear()
            self.zone_survival.clear()
        self.stop_event.clear()
        self.pause_event.clear()
        self.resume_event.set()

    # ---- snapshot de solo lectura para la UI ----
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "mode": self.mode,
                "message": self.message,
                "training_start_time": self.training_start_time,
                "timesteps": self.timesteps,
                "episodes": self.episodes,
                "current_x": self.current_x,
                "max_x_this_ep": self.max_x_this_ep,
                "max_x_historical": self.max_x_historical,
                "reward_avg_100": self.reward_avg_100,
                "distance_avg_100": self.distance_avg_100,
                "clear_rate_100": self.clear_rate_100,
                "coins_avg_100": self.coins_avg_100,
                "behavior_score": self.behavior_score,
                "last_episode_reward": self.last_episode_reward,
                "last_clear": self.last_clear,
                "zone_survival": dict(self.zone_survival),
                "paused": self.pause_event.is_set(),
                "stopping": self.stop_event.is_set(),
                "demo_episode": self.demo_episode,
                "demo_score": self.demo_score,
                "demo_coins": self.demo_coins,
                "demo_time": self.demo_time,
                "demo_flag": self.demo_flag,
                "demo_results": list(self.demo_results),
            }

    def push_trajectory(self) -> None:
        with self.lock:
            if not self.current_trajectory:
                return
            self.trajectories.append(list(self.current_trajectory))
            if len(self.trajectories) > MAX_TRAJECTORIES:
                self.trajectories = self.trajectories[-MAX_TRAJECTORIES:]
            self.current_trajectory = []
