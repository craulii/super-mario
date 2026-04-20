"""Detector de patrones cíclicos de acciones."""

from __future__ import annotations

from collections import deque


class ActionPatternDetector:
    """Detecta si las últimas N acciones forman un ciclo de período p∈[2, p_max]."""

    def __init__(self, window: int = 40, min_repeats: int = 3, p_max: int = 10):
        self._window = window
        self._min_repeats = min_repeats
        self._p_max = p_max
        self._buf: deque[int] = deque(maxlen=window)

    def push(self, action: int) -> None:
        self._buf.append(int(action))

    def detect(self) -> tuple[bool, int]:
        n = len(self._buf)
        if n < self._min_repeats * 2:
            return False, 0
        seq = list(self._buf)
        for p in range(2, min(self._p_max, n // self._min_repeats) + 1):
            # comparar las últimas min_repeats ventanas de tamaño p
            ok = True
            tail = seq[-p:]
            for k in range(1, self._min_repeats):
                block = seq[-p * (k + 1) : -p * k]
                if block != tail:
                    ok = False
                    break
            if ok:
                return True, p
        return False, 0

    def reset(self) -> None:
        self._buf.clear()
