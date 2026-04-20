"""Callback que graba un clip mp4 del episodio cuando se logra un récord."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from utils.logging_setup import get_logger

logger = get_logger("video")


class RecordVideoCallback(BaseCallback):
    """Graba clips mp4 de episodios con récord.

    Política de guardado:
      - Si el episodio completó el nivel (flag_get=True) **y** batió un récord:
        se crea un archivo nuevo con timestamp en `clears/`, se preserva para siempre.
      - Si batió récord pero no completó el nivel:
        se sobrescribe un único archivo `best_incomplete.mp4` (solo queda el último
        mejor intento incompleto).

    Los clears nunca se borran ni se sobrescriben — cada clear con récord vive en
    `<output_dir>/clears/clear_{ts}_dist{x}_time{t}.mp4`.
    El mejor incompleto vive en `<output_dir>/best_incomplete.mp4`.
    """

    def __init__(
        self,
        output_dir: Path,
        max_frames: int = 6000,
        fps: int = 30,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._output_dir = Path(output_dir)
        self._clears_dir = self._output_dir / "clears"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._clears_dir.mkdir(parents=True, exist_ok=True)
        self._incomplete_path = self._output_dir / "best_incomplete.mp4"
        self._max_frames = max_frames
        self._fps = fps

        self._frames: deque[np.ndarray] = deque(maxlen=max_frames)
        self._recorded_distance: bool = False
        self._recorded_time: bool = False
        self._max_x_in_episode: int = 0
        self._final_time: int = 0
        self._flag_get: bool = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        info = infos[0] or {}
        done = bool(dones[0])

        try:
            frame = self.training_env.render(mode="rgb_array")
            if isinstance(frame, list):
                frame = frame[0] if frame else None
            if frame is not None:
                self._frames.append(np.asarray(frame, dtype=np.uint8))
        except Exception:
            pass

        if info.get("new_record_distance"):
            self._recorded_distance = True
        if info.get("new_record_time"):
            self._recorded_time = True

        self._max_x_in_episode = max(self._max_x_in_episode, int(info.get("x_pos", 0)))
        self._final_time = int(info.get("time", 0))
        if info.get("flag_get"):
            self._flag_get = True

        if done:
            if self._recorded_distance or self._recorded_time:
                self._save_clip()
            self._frames.clear()
            self._recorded_distance = False
            self._recorded_time = False
            self._max_x_in_episode = 0
            self._final_time = 0
            self._flag_get = False

        return True

    def _save_clip(self) -> None:
        if not self._frames:
            return

        if self._flag_get:
            ts = time.strftime("%Y%m%d_%H%M%S")
            tags: list[str] = [f"dist{self._max_x_in_episode}"]
            if self._recorded_time:
                tags.append(f"time{self._final_time}")
            name = f"clear_{ts}_" + "_".join(tags) + ".mp4"
            path = self._clears_dir / name
            label = "clear"
        else:
            path = self._incomplete_path
            label = "incomplete"

        self._write_video(path)
        logger.info("[%s] clip guardado: %s (%d frames)", label, path, len(self._frames))

    def _write_video(self, path: Path) -> None:
        first = self._frames[0]
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h)
        )
        try:
            for frame in self._frames:
                if frame.ndim == 3 and frame.shape[2] == 3:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    bgr = frame
                writer.write(bgr)
        finally:
            writer.release()
