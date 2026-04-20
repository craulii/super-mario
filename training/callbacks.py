"""Callback principal de entrenamiento: checkpoints, TensorBoard, early stopping.

Usa RingBuffer para medias O(1) y se integra con SharedState para exponer métricas.
Soporta múltiples environments paralelos (SubprocVecEnv).
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from configs.schema import Config
from training.shared_state import SharedState
from utils.ring_buffer import RingBuffer


class MarioTrainingCallback(BaseCallback):
    def __init__(self, cfg: Config, state: SharedState, verbose: int = 1):
        super().__init__(verbose)
        self._cfg = cfg
        self._state = state
        self._n_envs = cfg.training.n_envs
        self._checkpoint_freq = cfg.training.checkpoint_freq
        self._early_stop = cfg.training.early_stop_consecutive

        self._rewards = RingBuffer(100)
        self._distances = RingBuffer(100)
        self._flags = RingBuffer(100)
        self._coins = RingBuffer(100)

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

        if self._checkpoint_freq > 0 and self.n_calls % self._checkpoint_freq == 0:
            ckpt_dir = self._cfg.paths.checkpoint_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"mario_ppo_{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose:
                print(f"[checkpoint] {path}")

        return True
