"""Wrappers de preprocesado (sin reward shaping)."""

from __future__ import annotations

import gym
import numpy as np
from gym import spaces


class SqueezeChannelObservation(gym.ObservationWrapper):
    """Si la última dimensión es 1 (canal gris suelto), la elimina."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = env.observation_space.shape
        if len(shape) == 3 and shape[-1] == 1:
            new_shape = shape[:-1]
        else:
            new_shape = shape
        low = env.observation_space.low
        high = env.observation_space.high
        self.observation_space = spaces.Box(
            low=low.reshape(new_shape) if hasattr(low, "reshape") else 0,
            high=high.reshape(new_shape) if hasattr(high, "reshape") else 255,
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr.squeeze(-1)
        return arr


class SkipFrame(gym.Wrapper):
    """Repite la misma acción durante `skip` frames y acumula reward."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        obs, terminated, truncated, info = None, False, False, {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
