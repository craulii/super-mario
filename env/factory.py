"""Factory para construir el environment Mario con todos los wrappers."""

from __future__ import annotations

from pathlib import Path

import gym
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from configs.schema import Config
from env.death_map import DeathLocationTracker
from env.reward_shaping import AdvancedRewardWrapper
from env.wrappers_obs import SkipFrame, SqueezeChannelObservation


class ConfigurableEnvWrapper(gym.Wrapper):
    """Wrapper externo que expone métodos accesibles via SubprocVecEnv.env_method().

    gym.Wrapper no delega getattr arbitrario, así que este wrapper mantiene
    una referencia directa al AdvancedRewardWrapper interno para poder
    actualizar el RewardConfig y recoger death counts desde el proceso principal.

    También convierte LazyFrames → numpy array para evitar el crash de SB3
    al procesar terminal_observation con SubprocVecEnv.
    """

    def __init__(self, env: gym.Env, reward_wrapper: AdvancedRewardWrapper):
        super().__init__(env)
        self._reward_wrapper = reward_wrapper

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return np.asarray(obs), info
        return np.asarray(result)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return np.asarray(obs), reward, terminated, truncated, info

    def update_reward_config(self, patch: dict) -> None:
        """Aplica cambios parciales al RewardConfig del wrapper de reward."""
        cfg = self._reward_wrapper.reward_cfg
        for k, v in patch.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    def get_death_counts(self) -> dict:
        """Devuelve conteos de muerte en memoria (para fusión desde proceso principal)."""
        if self._reward_wrapper._death_map is not None:
            return self._reward_wrapper._death_map.as_dict()
        return {}


def make_env(
    cfg: Config,
    render_mode: str | None = None,
    death_map: DeathLocationTracker | None = None,
) -> gym.Env:
    """Construye el env de Mario aplicando los wrappers en orden.

    Orden:
      1. JoypadSpace(COMPLEX_MOVEMENT)
      2. AdvancedRewardWrapper (reward shaping — lee RewardConfig por referencia)
      3. SkipFrame
      4. GrayScaleObservation
      5. ResizeObservation (tupla, no int — bug corregido)
      6. SqueezeChannelObservation
      7. FrameStack
      8. ConfigurableEnvWrapper (capa externa para env_method)
    """
    effective_render_mode = render_mode or "rgb_array"
    base = gym_super_mario_bros.make(
        cfg.env.env_id,
        render_mode=effective_render_mode,
        apply_api_compatibility=True,
    )

    actions = SIMPLE_MOVEMENT if cfg.env.action_space == "simple" else COMPLEX_MOVEMENT
    env: gym.Env = JoypadSpace(base, actions)
    reward_wrapper = AdvancedRewardWrapper(
        env,
        reward_cfg=cfg.reward,
        death_map=death_map,
        best_distance_path=cfg.paths.best_distance_path,
        best_time_path=cfg.paths.best_time_path,
    )
    env = reward_wrapper
    env = SkipFrame(env, skip=cfg.env.skip_frames)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(cfg.env.resize_shape, cfg.env.resize_shape))
    env = SqueezeChannelObservation(env)
    env = FrameStack(env, num_stack=cfg.env.frame_stack)
    env = ConfigurableEnvWrapper(env, reward_wrapper)
    return env


def make_vec_env(
    cfg: Config,
    death_map_path: Path | None = None,
) -> "SubprocVecEnv":
    """Crea N environments en subprocesos paralelos.

    Cada subproceso tiene su propia copia del env, reward config y death map.
    El proceso principal puede propagar cambios via env_method().
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    n_envs = cfg.training.n_envs
    bucket_size = cfg.reward.death_bucket_size

    def _make_fn(rank: int):
        def _init():
            dm = None
            if death_map_path is not None:
                dm = DeathLocationTracker(death_map_path, bucket_size=bucket_size)
                dm.load()
            return make_env(cfg, death_map=dm)
        return _init

    return SubprocVecEnv([_make_fn(i) for i in range(n_envs)])
