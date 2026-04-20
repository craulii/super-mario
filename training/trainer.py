"""Trainer: encapsula model.learn() en un thread con pause/resume/stop/save/load."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure as sb3_configure

from configs.schema import Config
from env.death_map import DeathLocationTracker
from env.factory import make_env, make_vec_env
from models.io import load_model, save_model
from models.ppo_factory import make_model
from training.callbacks import MarioTrainingCallback
from training.control_callback import PauseStopCallback
from training.metrics_callback import MetricsStreamCallback
from training.shared_state import SharedState
from training.video_callback import RecordVideoCallback
from utils.logging_setup import get_logger
from utils.seed import set_global_seed

logger = get_logger("trainer")


class Trainer:
    """Orquesta entrenamiento en un thread dedicado.

    Reglas:
      - start(): crea env + modelo + thread y arranca learn().
      - pause()/resume(): cooperativo vía SharedState events.
      - stop(): señaliza stop_event; el callback devuelve False y learn() termina.
      - save()/load(): permitidos solo en estado pausado o idle.
    """

    def __init__(self, cfg: Config, state: SharedState):
        self._cfg = cfg
        self._state = state
        # Alineamos el RewardConfig del state con el del config (misma referencia).
        self._state.reward_config = cfg.reward

        self._thread: threading.Thread | None = None
        self._model: Any | None = None
        self._env: Any | None = None
        self._death_map: DeathLocationTracker | None = None
        self._run_id: str = ""

    # ------------------------------------------------------------------
    # Ciclo de vida del training
    # ------------------------------------------------------------------
    def start(self, resume_path: Path | str | None = None) -> None:
        if self.is_running():
            logger.warning("start() ignorado: ya hay un training activo")
            return
        self._state.reset_for_new_run()
        self._cfg.ensure_dirs()
        set_global_seed(self._cfg.training.seed)
        self._run_id = time.strftime("%Y%m%d_%H%M%S")
        # Crear subcarpeta de checkpoints para este run
        self._run_checkpoint_dir = self._cfg.paths.checkpoint_dir / f"run_{self._run_id}"
        self._run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with self._state.lock:
            self._state.mode = "training"
            self._state.message = "arrancando"
            self._state.training_start_time = time.time()

        self._thread = threading.Thread(
            target=self._run,
            args=(Path(resume_path) if resume_path else None,),
            daemon=True,
            name="mario-trainer",
        )
        self._thread.start()

    def pause(self) -> None:
        if not self.is_running():
            return
        self._state.request_pause()

    def resume(self) -> None:
        if not self.is_running():
            return
        self._state.request_resume()
        with self._state.lock:
            self._state.mode = "training"

    def stop(self, timeout: float = 30.0) -> None:
        if not self.is_running():
            return
        self._state.request_stop()
        with self._state.lock:
            self._state.mode = "stopping"
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        with self._state.lock:
            self._state.mode = "idle"

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_paused(self) -> bool:
        return self._state.pause_event.is_set()

    # ------------------------------------------------------------------
    # Guardar / cargar
    # ------------------------------------------------------------------
    def save(self, path: Path | str | None = None) -> Path:
        if self.is_running() and not self.is_paused():
            raise RuntimeError("save() requiere training detenido o pausado")
        if self._model is None:
            raise RuntimeError("no hay modelo cargado")
        target = Path(path) if path else self._cfg.paths.model_save_path
        return save_model(
            self._model,
            target,
            metadata={
                "timesteps": int(getattr(self._model, "num_timesteps", 0)),
                "reward_avg_100": float(self._state.reward_avg_100),
                "clear_rate_100": float(self._state.clear_rate_100),
                "config": self._cfg.to_dict(),
            },
        )

    def load(self, path: Path | str) -> dict[str, Any]:
        if self.is_running():
            raise RuntimeError("load() requiere trainer detenido")
        # El modelo cargado no tiene env; se asigna al hacer start si se reanuda.
        model, meta = load_model(path, env=None, device=self._cfg.training.device)
        self._model = model
        return meta

    # ------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------
    def evaluate(
        self,
        model_path: Path | str | None = None,
        n_episodes: int = 5,
        render: bool = False,
    ) -> list[dict[str, Any]]:
        """Ejecuta N episodios de evaluación y devuelve métricas por episodio."""
        if self.is_running():
            raise RuntimeError("evaluate() requiere trainer detenido")
        self._cfg.ensure_dirs()
        set_global_seed(self._cfg.training.seed)

        env = make_env(
            self._cfg,
            render_mode="human" if render else None,
            death_map=self._make_death_map(),
        )
        try:
            path = Path(model_path) if model_path else self._cfg.paths.model_save_path
            model, _ = load_model(path, env=env, device=self._cfg.training.device)
        except FileNotFoundError:
            env.close()
            raise

        self._state.reset_for_new_run()
        with self._state.lock:
            self._state.mode = "evaluating"

        results: list[dict[str, Any]] = []
        try:
            for ep in range(1, n_episodes + 1):
                if self._state.stop_event.is_set():
                    break
                reset_result = env.reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                done = False
                total_reward = 0.0
                max_x = 0
                flag = False
                coins = 0
                score = 0
                time_left = 0
                start_t = time.time()
                while not done and not self._state.stop_event.is_set():
                    action, _ = model.predict(obs, deterministic=True)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result
                    total_reward += float(reward)
                    x = int(info.get("x_pos", 0))
                    max_x = max(max_x, x)
                    flag = bool(info.get("flag_get", False))
                    coins = int(info.get("coins", coins))
                    score = int(info.get("score", score))
                    time_left = int(info.get("time", time_left))

                    with self._state.lock:
                        self._state.current_x = x
                        self._state.max_x_this_ep = max_x
                        self._state.current_trajectory.append((x, int(info.get("y_pos", 0))))

                elapsed = time.time() - start_t
                self._state.push_trajectory()
                results.append(
                    {
                        "episode": ep,
                        "reward": total_reward,
                        "max_x": max_x,
                        "flag_get": flag,
                        "coins": coins,
                        "score": score,
                        "time_left": time_left,
                        "elapsed_s": elapsed,
                    }
                )
        finally:
            env.close()
            with self._state.lock:
                self._state.mode = "idle"
        return results

    # ------------------------------------------------------------------
    # Demo (evaluación en loop con streaming de frames)
    # ------------------------------------------------------------------
    def run_demo(
        self,
        model_path: Path | str | None = None,
        n_episodes: int = 0,
    ) -> None:
        """Ejecuta episodios de evaluación en loop, streameando frames al SharedState.

        n_episodes=0 → loop infinito hasta Ctrl+C.
        """
        if self.is_running():
            raise RuntimeError("demo requiere trainer detenido")
        self._cfg.ensure_dirs()

        env = make_env(self._cfg, death_map=None)
        try:
            path = Path(model_path) if model_path else self._cfg.paths.model_save_path
            model, _ = load_model(path, env=env, device=self._cfg.training.device)
        except FileNotFoundError:
            env.close()
            raise

        with self._state.lock:
            self._state.mode = "evaluating"
            self._state.demo_results = []

        ep = 0
        total = n_episodes if n_episodes > 0 else float("inf")
        try:
            while ep < total and not self._state.stop_event.is_set():
                ep += 1
                reset_result = env.reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                done = False
                total_reward = 0.0
                max_x = 0
                flag = False
                coins = 0
                score = 0
                time_left = 400

                with self._state.lock:
                    self._state.demo_episode = ep
                    self._state.demo_score = 0
                    self._state.demo_coins = 0
                    self._state.demo_time = 400
                    self._state.demo_flag = False

                while not done and not self._state.stop_event.is_set():
                    action, _ = model.predict(obs, deterministic=True)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result
                    total_reward += float(reward)
                    x = int(info.get("x_pos", 0))
                    max_x = max(max_x, x)
                    flag = bool(info.get("flag_get", False))
                    coins = int(info.get("coins", coins))
                    score = int(info.get("score", score))
                    time_left = int(info.get("time", time_left))

                    # Capturar frame
                    try:
                        frame = env.render()
                        if isinstance(frame, tuple):
                            frame = frame[0]
                        if frame is not None:
                            with self._state.lock:
                                self._state.last_frame = np.asarray(frame, dtype=np.uint8)
                    except Exception:
                        pass

                    with self._state.lock:
                        self._state.current_x = x
                        self._state.max_x_this_ep = max_x
                        self._state.demo_score = score
                        self._state.demo_coins = coins
                        self._state.demo_time = time_left
                        self._state.demo_flag = flag
                        self._state.current_trajectory.append(
                            (x, int(info.get("y_pos", 0)))
                        )

                # Episodio terminado
                result = {
                    "episode": ep,
                    "max_x": max_x,
                    "flag_get": flag,
                    "coins": coins,
                    "score": score,
                    "time_left": time_left,
                    "progress": round(max_x / 3360 * 100, 1),
                }
                with self._state.lock:
                    self._state.demo_results.append(result)
                    if len(self._state.demo_results) > 50:
                        self._state.demo_results = self._state.demo_results[-50:]
                self._state.push_trajectory()

        finally:
            env.close()
            with self._state.lock:
                self._state.mode = "idle"

    # ------------------------------------------------------------------
    # Propagación de reward config a subprocesos
    # ------------------------------------------------------------------
    def propagate_reward_config(self, patch: dict) -> None:
        """Envía cambios de RewardConfig a todos los environments en subprocesos."""
        if self._env is not None and hasattr(self._env, "env_method"):
            try:
                self._env.env_method("update_reward_config", patch)
            except Exception as exc:
                logger.warning("no se pudo propagar reward config: %s", exc)

    # ------------------------------------------------------------------
    # Thread de entrenamiento
    # ------------------------------------------------------------------
    def _run(self, resume_path: Path | None) -> None:
        n_envs = self._cfg.training.n_envs
        try:
            self._death_map = self._make_death_map()

            if n_envs > 1:
                logger.info("creando %d environments paralelos (SubprocVecEnv)", n_envs)
                self._env = make_vec_env(
                    self._cfg,
                    death_map_path=self._cfg.paths.death_map_path,
                )
            else:
                self._env = make_env(self._cfg, death_map=self._death_map)

            if resume_path is not None:
                logger.info("reanudando desde %s", resume_path)
                model, _ = load_model(resume_path, env=self._env, device=self._cfg.training.device)
                self._model = model
            elif self._model is not None:
                self._model.set_env(self._env)
            else:
                self._model = make_model(self._cfg, self._env)

            # Redirigir logger de SB3 solo a TensorBoard para no ensuciar la TUI.
            self._cfg.paths.log_dir.mkdir(parents=True, exist_ok=True)
            self._model.set_logger(
                sb3_configure(str(self._cfg.paths.log_dir), ["tensorboard"])
            )

            callbacks = CallbackList(
                [
                    PauseStopCallback(self._state),
                    MetricsStreamCallback(self._state, n_envs=n_envs),
                    MarioTrainingCallback(
                        self._cfg, self._state,
                        checkpoint_dir=self._run_checkpoint_dir,
                        run_id=self._run_id,
                        verbose=0,
                    ),
                    RecordVideoCallback(self._cfg.paths.video_dir / "records"),
                ]
            )

            total = int(self._cfg.training.total_timesteps)
            logger.info(
                "entrenando %s steps en %s (%d envs paralelos)",
                total, self._cfg.training.device, n_envs,
            )

            self._model.learn(
                total_timesteps=total,
                callback=callbacks,
                tb_log_name="mario_ppo",
                reset_num_timesteps=(resume_path is None),
            )

            # Recoger death maps de subprocesos y fusionar
            self._merge_subprocess_death_maps()

            # Guardar al finalizar (en carpeta global + en carpeta del run)
            save_model(
                self._model,
                self._cfg.paths.model_save_path,
                metadata={"final": True, "timesteps": int(self._model.num_timesteps),
                           "run_id": self._run_id},
            )
            save_model(
                self._model,
                self._run_checkpoint_dir / "final",
                metadata={"final": True, "timesteps": int(self._model.num_timesteps),
                           "run_id": self._run_id},
            )
            if self._death_map is not None:
                self._death_map.save()
        except Exception as exc:
            logger.exception("error en training thread: %s", exc)
            with self._state.lock:
                self._state.message = f"error: {exc}"
        finally:
            try:
                if self._env is not None:
                    self._env.close()
            except Exception:
                pass
            with self._state.lock:
                self._state.mode = "idle"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_death_map(self) -> DeathLocationTracker:
        tracker = DeathLocationTracker(
            self._cfg.paths.death_map_path,
            bucket_size=self._cfg.reward.death_bucket_size,
        )
        if self._cfg.training.reset_memory:
            tracker.reset()
        else:
            tracker.load()
        return tracker

    def _merge_subprocess_death_maps(self) -> None:
        """Recoge conteos de muerte de todos los subprocesos y los fusiona."""
        if self._death_map is None or self._env is None:
            return
        if not hasattr(self._env, "env_method"):
            return
        try:
            all_counts = self._env.env_method("get_death_counts")
            for counts in all_counts:
                if counts:
                    self._death_map.merge_counts(counts)
        except Exception as exc:
            logger.warning("no se pudieron fusionar death maps: %s", exc)
