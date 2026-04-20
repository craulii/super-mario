"""Mapa persistente de muertes por zona del nivel."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class DeathLocationTracker:
    """Conteo de muertes por bucket horizontal, persistente en disco."""

    def __init__(self, path: Path, bucket_size: int = 32):
        self._path = Path(path)
        self._bucket_size = bucket_size
        self._counts: dict[int, int] = defaultdict(int)

    def load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._counts = defaultdict(int, {int(k): int(v) for k, v in data.items()})
            except (json.JSONDecodeError, ValueError):
                self._counts = defaultdict(int)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(dict(self._counts)), encoding="utf-8")

    def reset(self) -> None:
        self._counts.clear()

    def bucket(self, x: int) -> int:
        return x // self._bucket_size

    def record_death(self, x: int) -> int:
        b = self.bucket(x)
        self._counts[b] += 1
        return self._counts[b]

    def count_at(self, x: int) -> int:
        return self._counts.get(self.bucket(x), 0)

    def as_dict(self) -> dict[int, int]:
        return dict(self._counts)

    def merge_counts(self, other_counts: dict[int, int]) -> None:
        """Fusiona conteos de otro tracker (útil para recoger datos de subprocesos)."""
        for bucket, count in other_counts.items():
            self._counts[int(bucket)] += int(count)
