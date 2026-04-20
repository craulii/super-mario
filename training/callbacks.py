"""Callback principal de entrenamiento: checkpoints, TensorBoard, early stopping.

Usa RingBuffer para medias O(1) y se integra con SharedState para exponer métricas.
Soporta múltiples environments paralelos (SubprocVecEnv).
Genera un CSV de log por run en logs/run_YYYYMMDD_HHMMSS.csv.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from configs.schema import Config
from training.shared_state import SharedState
from utils.ring_buffer import RingBuffer

CSV_COLUMNS = [
    "timestamp", "timesteps", "episode", "reward", "max_x", "coins",
    "score", "time_left", "flag_get", "reward_avg100", "distance_avg100",
    "clear_rate100", "coins_avg100", "max_x_historical",
]


class MarioTrainingCallback(BaseCallback):
    def __init__(
        self,
        cfg: Config,
        state: SharedState,
        checkpoint_dir: Path | None = None,
        run_id: str = "",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._cfg = cfg
        self._state = state
        self._n_envs = cfg.training.n_envs
        self._checkpoint_freq = cfg.training.checkpoint_freq
        self._early_stop = cfg.training.early_stop_consecutive
        self._checkpoint_dir = checkpoint_dir or cfg.paths.checkpoint_dir
        self._run_id = run_id or time.strftime("%Y%m%d_%H%M%S")

        self._rewards = RingBuffer(100)
        self._distances = RingBuffer(100)
        self._flags = RingBuffer(100)
        self._coins = RingBuffer(100)

        # CSV log por run
        log_dir = cfg.paths.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_dir / f"run_{self._run_id}.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_COLUMNS)
        self._max_x_historical = 0.0

        # Acumuladores por environment
        self._cur_reward = [0.0] * self._n_envs
        self._cur_max_x = [0] * self._n_envs
        self._cur_max_coins = [0] * self._n_envs
        self._cur_max_score = [0] * self._n_envs
        self._consecutive_clears = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i in range(self._n_envs):
            reward = float(rewards[i])
            info = infos[i] or {}
            done = bool(dones[i])

            self._cur_reward[i] += reward
            self._cur_max_x[i] = max(self._cur_max_x[i], int(info.get("x_pos", 0)))
            self._cur_max_coins[i] = max(self._cur_max_coins[i], int(info.get("coins", 0)))
            self._cur_max_score[i] = max(self._cur_max_score[i], int(info.get("score", 0)))

            if done:
                flag_get = bool(info.get("flag_get", False))
                time_left = int(info.get("time", 0))

                self._rewards.append(self._cur_reward[i])
                self._distances.append(self._cur_max_x[i])
                self._flags.append(1.0 if flag_get else 0.0)
                self._coins.append(self._cur_max_coins[i])

                avg_r = self._rewards.mean()
                avg_d = self._distances.mean()
                clear_rate = self._flags.mean()
                avg_c = self._coins.mean()

                self.logger.record("mario/episode_reward", self._cur_reward[i])
                self.logger.record("mario/max_x_pos", self._cur_max_x[i])
                self.logger.record("mario/flag_get", int(flag_get))
                self.logger.record("mario/coins", self._cur_max_coins[i])
                self.logger.record("mario/score", self._cur_max_score[i])
                self.logger.record("mario/time_remaining", time_left)
                self.logger.record("mario/total_episodes", len(self._rewards))
                self.logger.record("mario/avg_reward_100", avg_r)
                self.logger.record("mario/avg_distance_100", avg_d)
                self.logger.record("mario/clear_rate_100", clear_rate)
                self.logger.record("mario/avg_coins_100", avg_c)
                self.logger.record("mario/progress_ratio", float(info.get("progress_ratio", 0.0)))
                self.logger.record("mario/behavior_score", float(info.get("behavior_score", 0.0)))

                with self._state.lock:
                    self._state.reward_avg_100 = avg_r
                    self._state.distance_avg_100 = avg_d
                    self._state.clear_rate_100 = clear_rate
                    self._state.coins_avg_100 = avg_c
                    self._state.metrics_history.append(
                        {
                            "ep": len(self._rewards),
                            "reward": self._cur_reward[i],
                            "distance": float(self._cur_max_x[i]),
                            "clear": 1.0 if flag_get else 0.0,
                            "reward_avg_100": avg_r,
                            "clear_rate_100": clear_rate,
                        }
                    )
                    if len(self._state.metrics_history) > 2000:
                        self._state.metrics_history = self._state.metrics_history[-2000:]

                # CSV log
                self._max_x_historical = max(self._max_x_historical, float(self._cur_max_x[i]))
                self._csv_writer.writerow([
                    time.strftime("%H:%M:%S"),
                    self.num_timesteps,
                    len(self._rewards),
                    round(self._cur_reward[i], 1),
                    self._cur_max_x[i],
                    self._cur_max_coins[i],
                    self._cur_max_score[i],
                    time_left,
                    int(flag_get),
                    round(avg_r, 1),
                    round(avg_d, 1),
                    round(clear_rate, 3),
                    round(avg_c, 1),
                    round(self._max_x_historical, 0),
                ])
                self._csv_file.flush()

                if flag_get:
                    self._consecutive_clears += 1
                    if self.verbose:
                        print(
                            f"[clear] consec={self._consecutive_clears}/{self._early_stop} "
                            f"coins={self._cur_max_coins[i]} time_left={time_left}"
                        )
                else:
                    self._consecutive_clears = 0

                if self._consecutive_clears >= self._early_stop:
                    emergency = self._cfg.paths.checkpoint_dir / "early_stop"
                    emergency.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(emergency))
                    if self.verbose:
                        print(f"[early-stop] guardado en {emergency}")
                    return False

                if self.verbose and len(self._rewards) % 10 == 0:
                    print(
                        f"ep={len(self._rewards)} steps={self.num_timesteps} "
                        f"r_avg={avg_r:.1f} dist_avg={avg_d:.0f} clear={clear_rate*100:.0f}%"
                    )

                # Reset acumuladores de este env
                self._cur_reward[i] = 0.0
                self._cur_max_x[i] = 0
                self._cur_max_coins[i] = 0
                self._cur_max_score[i] = 0

        if self._checkpoint_freq > 0 and self.num_timesteps % self._checkpoint_freq < self._n_envs:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self._checkpoint_dir / f"mario_ppo_{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose:
                print(f"[checkpoint] {path}")

        return True

    def _on_training_end(self) -> None:
        self._csv_file.close()
