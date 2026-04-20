"""Reward shaping avanzado para Super Mario Bros 1-2.

El wrapper guarda una REFERENCIA al RewardConfig. Cambios in-place en ese
objeto se aplican en vivo sin reiniciar el entrenamiento.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import gym
import numpy as np

from configs.schema import RewardConfig
from env.action_history import ActionPatternDetector
from env.death_map import DeathLocationTracker
from env.level_metrics import X_MAX_LEVEL_1_2, progress_ratio, zone_of

# Acciones de COMPLEX_MOVEMENT que presionan RIGHT (ver gym_super_mario_bros.actions).
# COMPLEX_MOVEMENT = [
#  0: NOOP            1: right           2: right+A         3: right+B
#  4: right+A+B       5: A               6: left            7: left+A
#  8: left+B          9: left+A+B       10: down           11: up ]
RIGHT_ACTIONS: frozenset[int] = frozenset({1, 2, 3, 4})
LEFT_ACTIONS: frozenset[int] = frozenset({6, 7, 8, 9})


def _load_scalar_json(path: Path, default: float) -> float:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("value", default))
    except (json.JSONDecodeError, ValueError):
        return default


def _save_scalar_json(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"value": float(value)}), encoding="utf-8")


class AdvancedRewardWrapper(gym.Wrapper):
    """Reward shaping con 9+ componentes activables por flags en RewardConfig.

    El `reward_cfg` se pasa por referencia: mutarlo in-place (setattr) reconfigura
    el wrapper en tiempo de ejecución sin reinicializar el entrenamiento.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_cfg: RewardConfig,
        death_map: DeathLocationTracker | None = None,
        best_distance_path: Path | None = None,
        best_time_path: Path | None = None,
    ):
        super().__init__(env)
        self.reward_cfg = reward_cfg
        self._death_map = death_map

        self._best_distance_path = best_distance_path
        self._best_time_path = best_time_path
        self._max_x_historical: float = (
            _load_scalar_json(best_distance_path, 0.0) if best_distance_path else 0.0
        )
        self._best_time_historical: float = (
            _load_scalar_json(best_time_path, 0.0) if best_time_path else 0.0
        )
        self._max_x_historical_at_ep_start: float = self._max_x_historical

        # Estado por episodio
        self._prev_x: int = 0
        self._prev_y: int = 79
        self._prev_time: int = 400
        self._prev_coins: int = 0
        self._prev_score: int = 0
        self._prev_life: int = 2

        self._stuck_counter: int = 0
        self._visited_y_buckets: set[int] = set()

        self._max_x_this_ep: int = 0
        self._no_progress_counter: int = 0

        self._pattern = ActionPatternDetector(
            window=reward_cfg.cyclic_window,
            min_repeats=reward_cfg.cyclic_min_repeats,
        )
        self._pattern_progress_x_window: deque[int] = deque(
            maxlen=reward_cfg.cyclic_window
        )

        self._excessive_left_counter: int = 0
        self._wall_stuck_counter: int = 0
        self._microshift_window: deque[int] = deque(
            maxlen=reward_cfg.micro_movement_window
        )
        self._steps_in_episode: int = 0

        # Componentes acumulados por episodio (para métricas)
        self._penalty_cyclic_total: float = 0.0
        self._penalty_wall_total: float = 0.0
        self._penalty_micro_total: float = 0.0
        self._penalty_left_total: float = 0.0
        self._record_bonus_given: bool = False

    # ---- propiedades expuestas para callbacks ----
    @property
    def max_x_this_ep(self) -> int:
        return self._max_x_this_ep

    @property
    def behavior_score(self) -> float:
        return -(
            self._penalty_cyclic_total
            + self._penalty_wall_total
            + self._penalty_micro_total
            + self._penalty_left_total
        )

    @property
    def max_x_historical(self) -> float:
        return self._max_x_historical

    def reset(self, **kwargs):
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._prev_x = 0
        self._prev_y = 79
        self._prev_time = 400
        self._prev_coins = 0
        self._prev_score = 0
        self._prev_life = 2
        self._stuck_counter = 0
        self._visited_y_buckets.clear()
        self._max_x_this_ep = 0
        self._no_progress_counter = 0
        self._pattern.reset()
        self._pattern_progress_x_window.clear()
        self._excessive_left_counter = 0
        self._wall_stuck_counter = 0
        self._microshift_window.clear()
        self._steps_in_episode = 0
        self._penalty_cyclic_total = 0.0
        self._penalty_wall_total = 0.0
        self._penalty_micro_total = 0.0
        self._penalty_left_total = 0.0
        self._record_bonus_given = False
        self._max_x_historical_at_ep_start = self._max_x_historical
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, _base_reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, _base_reward, done, info = result
            terminated, truncated = done, False
        cfg = self.reward_cfg
        shaped = 0.0

        x = int(info.get("x_pos", 0))
        y = int(info.get("y_pos", 79))
        coins = int(info.get("coins", 0))
        score = int(info.get("score", 0))
        cur_time = int(info.get("time", 400))
        life = int(info.get("life", self._prev_life))
        flag = bool(info.get("flag_get", False))

        self._steps_in_episode += 1
        self._pattern.push(int(action))
        self._pattern_progress_x_window.append(x)
        self._microshift_window.append(x)

        # 1) Progreso horizontal (stuck adaptativo)
        delta_x = x - self._prev_x
        if delta_x > 0:
            shaped += delta_x * cfg.forward_reward_coef
            # bonus extra si avanza en territorio nunca antes explorado
            if cfg.enable_records and x > self._max_x_historical_at_ep_start:
                shaped += delta_x * cfg.forward_reward_coef * 0.25
            self._stuck_counter = 0
        elif delta_x == 0:
            self._stuck_counter += 1
            threshold = max(
                5,
                int(cfg.stuck_threshold_base * (1 - progress_ratio(x) * 0.3)),
            )
            if self._stuck_counter > threshold:
                shaped += cfg.stuck_penalty_base
        else:
            shaped += delta_x * cfg.forward_reward_coef * cfg.backward_reward_coef
            self._stuck_counter = 0

        if x > self._max_x_this_ep:
            self._max_x_this_ep = x
            self._no_progress_counter = 0
        else:
            self._no_progress_counter += 1

        # Truncar episodio si no hay progreso en N steps
        if (
            cfg.no_progress_limit > 0
            and self._no_progress_counter >= cfg.no_progress_limit
            and not done
        ):
            truncated = True
            done = True
            shaped += cfg.death_by_time_penalty

        # 2) Exploración vertical en buckets
        y_bucket = y // 16
        if y_bucket not in self._visited_y_buckets:
            self._visited_y_buckets.add(y_bucket)
            shaped += cfg.vertical_explore_reward

        # 3) Monedas y score
        delta_coins = coins - self._prev_coins
        if delta_coins > 0:
            shaped += delta_coins * cfg.coin_reward

        delta_score = score - self._prev_score
        if delta_score > 0:
            shaped += delta_score * cfg.score_reward_coef

        # 4) Presión de tiempo
        dt = self._prev_time - cur_time
        if dt > 0:
            shaped += dt * cfg.time_penalty_coef

        # 5) Ciclos de acciones
        if cfg.enable_cyclic_detection and len(self._pattern_progress_x_window) == cfg.cyclic_window:
            is_cyclic, _period = self._pattern.detect()
            if is_cyclic:
                span_x = (
                    self._pattern_progress_x_window[-1]
                    - self._pattern_progress_x_window[0]
                )
                if span_x < cfg.cyclic_progress_threshold:
                    shaped += cfg.cyclic_action_penalty
                    self._penalty_cyclic_total += cfg.cyclic_action_penalty

        # 6) Retroceso excesivo
        if cfg.enable_excessive_left:
            if x < self._max_x_this_ep - cfg.excessive_left_pixels:
                self._excessive_left_counter += 1
                if self._excessive_left_counter > cfg.excessive_left_threshold_frames:
                    shaped += cfg.excessive_left_penalty
                    self._penalty_left_total += cfg.excessive_left_penalty
            else:
                self._excessive_left_counter = 0

        # 7) Atascado en muro (acción right sin movimiento)
        if cfg.enable_wall_stuck:
            if int(action) in RIGHT_ACTIONS and delta_x == 0:
                self._wall_stuck_counter += 1
                if self._wall_stuck_counter > cfg.wall_stuck_frames:
                    shaped += cfg.wall_stuck_penalty
                    self._penalty_wall_total += cfg.wall_stuck_penalty
            else:
                self._wall_stuck_counter = 0

        # 8) Micro-movimientos
        if (
            cfg.enable_micro_movement
            and len(self._microshift_window) == cfg.micro_movement_window
            and self._steps_in_episode % cfg.micro_movement_window == 0
        ):
            xs = list(self._microshift_window)
            span = max(xs) - min(xs)
            net = xs[-1] - xs[0]
            if span < cfg.micro_movement_span_threshold and abs(net) < cfg.micro_movement_net_threshold:
                shaped += cfg.micro_movement_penalty
                self._penalty_micro_total += cfg.micro_movement_penalty

        # 9) Récord histórico de distancia (one-shot por episodio)
        new_record_distance = False
        if cfg.enable_records and not self._record_bonus_given:
            if x > self._max_x_historical:
                self._max_x_historical = float(x)
                if self._best_distance_path is not None:
                    _save_scalar_json(self._best_distance_path, self._max_x_historical)
                shaped += cfg.record_distance_bonus
                self._record_bonus_given = True
                new_record_distance = True

        # 10) Muerte: distinguir enemigo vs tiempo
        died = life < self._prev_life or (done and not flag)
        if died:
            if cur_time <= 1:
                shaped += cfg.death_by_time_penalty
            else:
                shaped += cfg.death_by_enemy_penalty
            if cfg.enable_death_map and self._death_map is not None:
                count = self._death_map.record_death(x)
                extra = max(cfg.death_bucket_cap, cfg.death_bucket_coef * count)
                shaped += extra

        # 11) Completar nivel
        new_record_time = False
        if flag:
            shaped += cfg.flag_bonus
            shaped += cur_time * cfg.time_remaining_bonus_coef
            if cfg.enable_records and cur_time > self._best_time_historical:
                self._best_time_historical = float(cur_time)
                if self._best_time_path is not None:
                    _save_scalar_json(self._best_time_path, self._best_time_historical)
                shaped += cfg.record_time_bonus
                new_record_time = True

        # Actualizar estado previo
        self._prev_x = x
        self._prev_y = y
        self._prev_time = cur_time
        self._prev_coins = coins
        self._prev_score = score
        self._prev_life = life

        # Enriquecer info
        info = dict(info)
        info["max_x_this_ep"] = self._max_x_this_ep
        info["zone"] = zone_of(x)
        info["progress_ratio"] = progress_ratio(x)
        info["behavior_score"] = self.behavior_score
        info["died"] = died
        info["new_record_distance"] = new_record_distance
        info["new_record_time"] = new_record_time

        return obs, float(shaped), terminated, truncated, info
