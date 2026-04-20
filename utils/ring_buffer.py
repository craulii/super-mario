"""Buffer circular con media O(1)."""

from __future__ import annotations

from collections import deque
from typing import Iterable


class RingBuffer:
    """Mantiene los últimos N valores con media incremental."""

    def __init__(self, maxlen: int = 100):
        if maxlen <= 0:
            raise ValueError("maxlen debe ser > 0")
        self._maxlen = maxlen
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._sum: float = 0.0

    def append(self, value: float) -> None:
        if len(self._buf) == self._maxlen:
            self._sum -= self._buf[0]
        self._buf.append(float(value))
        self._sum += float(value)

    def extend(self, values: Iterable[float]) -> None:
        for v in values:
            self.append(v)

    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return self._sum / len(self._buf)

    def last(self) -> float:
        return self._buf[-1] if self._buf else 0.0

    def __len__(self) -> int:
        return len(self._buf)

    def values(self) -> list[float]:
        return list(self._buf)
