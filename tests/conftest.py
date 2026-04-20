"""Fixtures comunes para tests."""

from __future__ import annotations

import sys
from pathlib import Path

import gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyMarioEnv(gym.Env):
    """Env de juguete que simula las claves relevantes de Super Mario Bros.

    El test controla el cursor de info via `set_next_info()`. Así se pueden probar
    los componentes del reward shaping sin emular el NES.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 255, (4,), np.uint8)
        self.action_space = gym.spaces.Discrete(13)
        self._next_info: dict = {}
        self._done_next: bool = False

    def set_next_info(self, info: dict, done: bool = False) -> None:
        self._next_info = info
        self._done_next = done

    def reset(self, **kwargs):
        self._next_info = {
            "x_pos": 0, "y_pos": 79, "time": 400, "coins": 0,
            "score": 0, "life": 2, "flag_get": False
        }
        self._done_next = False
        return np.zeros(4, dtype=np.uint8), {}

    def step(self, action):
        return (
            np.zeros(4, dtype=np.uint8),
            0.0,
            self._done_next,
            False,
            dict(self._next_info),
        )


@pytest.fixture
def dummy_env():
    return DummyMarioEnv()
