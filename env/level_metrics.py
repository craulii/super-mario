"""Métricas del nivel 1-2: X máximo, zonas y cálculos de progreso."""

from __future__ import annotations

X_MAX_LEVEL_1_2: int = 3360

ZONES: list[tuple[int, int]] = [
    (0, 800),
    (800, 1600),
    (1600, 2400),
    (2400, X_MAX_LEVEL_1_2),
]


def zone_of(x: int) -> int:
    for i, (lo, hi) in enumerate(ZONES):
        if lo <= x < hi:
            return i
    return len(ZONES) - 1


def progress_ratio(x: int) -> float:
    return max(0.0, min(1.0, x / X_MAX_LEVEL_1_2))
