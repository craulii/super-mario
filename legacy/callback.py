"""Callbacks custom para entrenamiento: checkpoints, logging y early stopping."""

import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

from config import Config


class MarioTrainingCallback(BaseCallback):
    """Callback que combina checkpoint, logging avanzado y early stopping.

    Funcionalidades:
      - Guarda un checkpoint del modelo cada `checkpoint_freq` steps.
      - Loggea a TensorBoard: reward promedio, distancia recorrida, episodios completados.
      - Detiene el entrenamiento si el agente completa el nivel N veces consecutivas.
    """

    def __init__(self, cfg: Config, verbose: int = 1):
        super().__init__(verbose)
        self.cfg = cfg
        self.checkpoint_dir = cfg.checkpoint_dir
        self.checkpoint_freq = cfg.checkpoint_freq
        self.early_stop_consecutive = cfg.early_stop_consecutive

        # Métricas acumuladas por episodio
        self._episode_rewards: list[float] = []
        self._episode_distances: list[int] = []
        self._episode_flags: list[bool] = []
        self._episode_coins: list[int] = []
        self._episode_scores: list[int] = []
        self._current_reward = 0.0
        self._max_x_pos = 0
        self._max_coins = 0
        self._max_score = 0
        self._consecutive_clears = 0

    def _on_step(self) -> bool:
        # Acumular reward del step actual
        reward = self.locals["rewards"][0]
        self._current_reward += reward

        # Obtener info del environment
        info = self.locals["infos"][0]
        x_pos = info.get("x_pos", 0)
        self._max_x_pos = max(self._max_x_pos, x_pos)
        self._max_coins = max(self._max_coins, info.get("coins", 0))
        self._max_score = max(self._max_score, info.get("score", 0))

        # Fin de episodio
        done = self.locals["dones"][0]
        if done:
            flag_get = info.get("flag_get", False)
            time_left = info.get("time", 0)

            self._episode_rewards.append(self._current_reward)
            self._episode_distances.append(self._max_x_pos)
            self._episode_flags.append(flag_get)
            self._episode_coins.append(self._max_coins)
            self._episode_scores.append(self._max_score)

            # Logging a TensorBoard
            self.logger.record("mario/episode_reward", self._current_reward)
            self.logger.record("mario/max_x_pos", self._max_x_pos)
            self.logger.record("mario/flag_get", int(flag_get))
            self.logger.record("mario/coins", self._max_coins)
            self.logger.record("mario/score", self._max_score)
            self.logger.record("mario/time_remaining", time_left)
            self.logger.record("mario/total_episodes", len(self._episode_rewards))

            # Promedios sobre los últimos 100 episodios
            last_100_rewards = self._episode_rewards[-100:]
            last_100_distances = self._episode_distances[-100:]
            last_100_flags = self._episode_flags[-100:]
            last_100_coins = self._episode_coins[-100:]
            self.logger.record("mario/avg_reward_100", np.mean(last_100_rewards))
            self.logger.record("mario/avg_distance_100", np.mean(last_100_distances))
            self.logger.record("mario/clear_rate_100", np.mean(last_100_flags))
            self.logger.record("mario/avg_coins_100", np.mean(last_100_coins))

            # Early stopping: N niveles completados consecutivos
            if flag_get:
                self._consecutive_clears += 1
                if self.verbose:
                    print(f"  [Nivel completado] Consecutivos: {self._consecutive_clears}/{self.early_stop_consecutive} | Monedas: {self._max_coins} | Tiempo: {time_left}")
            else:
                self._consecutive_clears = 0

            if self._consecutive_clears >= self.early_stop_consecutive:
                if self.verbose:
                    print(f"\n=== Early stopping: {self.early_stop_consecutive} niveles completados consecutivamente ===")
                self.model.save(str(self.cfg.model_save_path) + "_early_stop")
                return False

            # Imprimir progreso en consola
            if self.verbose and len(self._episode_rewards) % 10 == 0:
                ep = len(self._episode_rewards)
                avg_r = np.mean(last_100_rewards)
                avg_d = np.mean(last_100_distances)
                avg_c = np.mean(last_100_coins)
                cr = np.mean(last_100_flags) * 100
                print(f"  Ep {ep} | Steps {self.num_timesteps} | Reward: {avg_r:.1f} | Dist: {avg_d:.0f} | Coins: {avg_c:.1f} | Clear: {cr:.0f}%")

            # Reset métricas del episodio
            self._current_reward = 0.0
            self._max_x_pos = 0
            self._max_coins = 0
            self._max_score = 0

        # Checkpoint periódico
        if self.n_calls % self.checkpoint_freq == 0:
            path = self.checkpoint_dir / f"mario_ppo_{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose:
                print(f"  [Checkpoint guardado] {path}")

        return True
