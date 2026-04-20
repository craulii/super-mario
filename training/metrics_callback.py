"""Callback que empuja métricas, frames y trayectorias al SharedState.

Soporta múltiples environments: telemetría en vivo del env 0,
conteo de episodios de todos los envs.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training.shared_state import SharedState


class MetricsStreamCallback(BaseCallback):
    """Alimenta SharedState con progreso en vivo.

    - Cada step: actualiza current_x, timesteps, trayectoria, frame (env 0).
    - Cada fin de episodio (cualquier env): actualiza episodes, last_episode_reward.
    """

    def __init__(
        self,
        state: SharedState,
        n_envs: int = 1,
        frame_every: int = 4,
        trajectory_every: int = 4,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._state = state
        self._n_envs = n_envs
        self._frame_every = max(1, frame_every)
        self._trajectory_every = max(1, trajectory_every)
        self._episode_rewards = [0.0] * n_envs
        self._zone_stats: dict[int, dict[str, int]] = defaultdict(
            lambda: {"survivals": 0, "deaths": 0}
        )

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(self._n_envs))
        infos = self.locals.get("infos", [{}] * self._n_envs)
        dones = self.locals.get("dones", np.zeros(self._n_envs, dtype=bool))

        # Acumular rewards de todos los envs
        for i in range(self._n_envs):
            self._episode_rewards[i] += float(rewards[i])

        # Telemetría en vivo solo del env 0
        info0 = infos[0] or {}
        x = int(info0.get("x_pos", 0))
        y = int(info0.get("y_pos", 0))

        with self._state.lock:
            self._state.timesteps = int(self.num_timesteps)
            self._state.current_x = x
            self._state.max_x_this_ep = max(self._state.max_x_this_ep, x)
            self._state.behavior_score = float(info0.get("behavior_score", 0.0))
            self._state.max_x_historical = max(
                self._state.max_x_historical, float(x)
            )

            if self.n_calls % self._trajectory_every == 0:
                self._state.current_trajectory.append((x, y))

        # Frame subsampleado del env 0
        if self.n_calls % self._frame_every == 0:
            try:
                frame = self.training_env.render(mode="rgb_array")
                if isinstance(frame, list):
                    frame = frame[0] if frame else None
                if frame is not None:
                    with self._state.lock:
                        self._state.last_frame = np.asarray(frame, dtype=np.uint8)
            except Exception:
                pass

        # Procesar fin de episodio de TODOS los envs
        for i in range(self._n_envs):
            if not bool(dones[i]):
                continue

            info_i = infos[i] or {}
            zone = int(info_i.get("zone", 0))
            died = bool(info_i.get("died", False))
            if died:
                self._zone_stats[zone]["deaths"] += 1
            else:
                self._zone_stats[zone]["survivals"] += 1

            survival_map: dict[int, float] = {}
            for z, s in self._zone_stats.items():
                total = s["survivals"] + s["deaths"]
                survival_map[z] = s["survivals"] / total if total > 0 else 0.0

            with self._state.lock:
                self._state.episodes += 1
                self._state.last_episode_reward = self._episode_rewards[i]
                self._state.last_clear = bool(info_i.get("flag_get", False))
                self._state.zone_survival = survival_map

                # Actualizar max_x_historical desde cualquier env
                max_x_i = float(info_i.get("x_pos", 0))
                self._state.max_x_historical = max(
                    self._state.max_x_historical, max_x_i
                )

                # Solo resetear max_x_this_ep y trayectoria para env 0
                if i == 0:
                    self._state.max_x_this_ep = 0

            if i == 0:
                self._state.push_trajectory()

            self._episode_rewards[i] = 0.0

        return True
