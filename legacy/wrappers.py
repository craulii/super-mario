"""Custom wrappers para el environment de Super Mario Bros."""

import gym
import numpy as np
from gym import spaces
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from config import Config


class CustomRewardWrapper(gym.Wrapper):
    """Wrapper de reward optimizado para velocidad, puntuación y exploración de rutas secretas.

    Bonificaciones:
      - Avanzar a la derecha (progreso horizontal).
      - Monedas recogidas (incentiva explorar zonas secretas de tuberías).
      - Score del juego (enemigos eliminados, bloques golpeados, etc.).
      - Exploración vertical (entrar en tuberías requiere bajar).
      - Tiempo restante al completar el nivel (premia velocidad).
      - Completar el nivel (flag).
    Penalizaciones:
      - Quedarse quieto o retroceder.
      - Morir.
      - Tiempo transcurrido (presión por avanzar).
    """

    def __init__(self, env: gym.Env, cfg: Config):
        super().__init__(env)
        self.cfg = cfg
        self._prev_x_pos = 0
        self._prev_y_pos = 0
        self._prev_time = 400
        self._prev_coins = 0
        self._prev_score = 0
        self._stuck_counter = 0
        self._visited_y_positions: set[int] = set()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_x_pos = 0
        self._prev_y_pos = 79  # posición Y inicial de Mario en el suelo
        self._prev_time = 400
        self._prev_coins = 0
        self._prev_score = 0
        self._stuck_counter = 0
        self._visited_y_positions = set()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # --- Reward shaping ---
        shaped_reward = 0.0

        # 1) Progreso horizontal
        x_pos = info.get("x_pos", 0)
        delta_x = x_pos - self._prev_x_pos

        if delta_x > 0:
            shaped_reward += delta_x * self.cfg.forward_reward_coef
            self._stuck_counter = 0
        elif delta_x == 0:
            self._stuck_counter += 1
            if self._stuck_counter > 30:
                shaped_reward += self.cfg.stuck_penalty
        else:
            # Retroceder: penalizar menos si está explorando verticalmente (puede estar
            # saliendo de una tubería hacia una zona diferente)
            shaped_reward += delta_x * self.cfg.forward_reward_coef * 0.5
            self._stuck_counter = 0

        self._prev_x_pos = x_pos

        # 2) Exploración vertical — recompensa por descubrir nuevas alturas (tuberías)
        y_pos = info.get("y_pos", 79)
        y_bucket = y_pos // 16  # agrupar en celdas de 16px para no premiar micro-movimientos
        if y_bucket not in self._visited_y_positions:
            self._visited_y_positions.add(y_bucket)
            shaped_reward += self.cfg.vertical_explore_reward
        self._prev_y_pos = y_pos

        # 3) Monedas — las zonas secretas de tuberías tienen muchas monedas
        coins = info.get("coins", 0)
        delta_coins = coins - self._prev_coins
        if delta_coins > 0:
            shaped_reward += delta_coins * self.cfg.coin_reward
        self._prev_coins = coins

        # 4) Score del juego — captura enemigos eliminados, bloques, etc.
        score = info.get("score", 0)
        delta_score = score - self._prev_score
        if delta_score > 0:
            shaped_reward += delta_score * self.cfg.score_reward_coef
        self._prev_score = score

        # 5) Penalización leve por tiempo transcurrido (presión por velocidad)
        current_time = info.get("time", 400)
        shaped_reward += self.cfg.time_penalty_coef * max(0, self._prev_time - current_time)
        self._prev_time = current_time

        # 6) Muerte
        if info.get("life", 2) < 2 or (done and info.get("flag_get", False) is False):
            shaped_reward += self.cfg.death_penalty

        # 7) Completar nivel — bonus base + bonus por velocidad (tiempo restante)
        if info.get("flag_get", False):
            shaped_reward += self.cfg.flag_bonus
            shaped_reward += current_time * self.cfg.time_remaining_bonus_coef

        return obs, shaped_reward, done, info


class SkipFrame(gym.Wrapper):
    """Repite la misma acción durante `skip` frames y acumula la reward."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


def make_env(cfg: Config, render_mode: str | None = None) -> gym.Env:
    """Crea y envuelve el environment de Mario con todos los wrappers.

    Orden de wrappers:
      1. JoypadSpace (SIMPLE_MOVEMENT)
      2. CustomRewardWrapper
      3. SkipFrame (4 frames)
      4. GrayScaleObservation
      5. ResizeObservation (84x84)
      6. FrameStack (4 frames)

    Args:
        cfg: Configuración del proyecto.
        render_mode: 'human' para renderizar, None para entrenamiento.

    Returns:
        Environment listo para usar con stable-baselines3.
    """
    if render_mode == "human":
        env = gym_super_mario_bros.make(cfg.env_id, render_mode="human", apply_api_compatibility=False)
    else:
        env = gym_super_mario_bros.make(cfg.env_id, apply_api_compatibility=False)

    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = CustomRewardWrapper(env, cfg)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=cfg.resize_shape)
    env = FrameStack(env, num_stack=cfg.frame_stack)

    return env
