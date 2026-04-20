"""Callback que implementa pausa y stop cooperativo."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from training.shared_state import SharedState


class PauseStopCallback(BaseCallback):
    """Bloquea el thread de training si `pause_event` está set. Detiene si `stop_event`."""

    def __init__(self, state: SharedState, verbose: int = 0):
        super().__init__(verbose)
        self._state = state

    def _on_step(self) -> bool:
        if self._state.stop_event.is_set():
            return False
        if self._state.pause_event.is_set():
            with self._state.lock:
                self._state.mode = "paused"
            # bloquea hasta que resume_event sea set
            self._state.resume_event.wait()
            if self._state.stop_event.is_set():
                return False
            with self._state.lock:
                self._state.mode = "training"
        return True
