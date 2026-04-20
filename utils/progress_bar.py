"""Helper para barra horizontal de progreso del nivel."""

from __future__ import annotations


def render_bar(current: int, maximum: int, width: int = 30) -> str:
    if maximum <= 0:
        return "[" + " " * width + "] n/a"
    ratio = max(0.0, min(1.0, current / maximum))
    filled = int(ratio * width)
    empty = width - filled
    pct = ratio * 100
    return f"[{'█' * filled}{'░' * empty}] {pct:5.1f}%  x={current}/{maximum}"
