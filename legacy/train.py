"""Script principal de entrenamiento del agente Mario con PPO."""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn

from config import Config
from wrappers import make_env
from callback import MarioTrainingCallback


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenar agente Mario con PPO")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total de timesteps de entrenamiento (default: valor en config)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar entrenamiento desde el último modelo guardado")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Ruta al modelo para reanudar entrenamiento")
    return parser.parse_args()


def make_model(cfg: Config, env, resume_path: str | None = None) -> PPO:
    """Crea o carga un modelo PPO.

    Args:
        cfg: Configuración del proyecto.
        env: Environment de entrenamiento.
        resume_path: Ruta a un modelo existente para reanudar. None para crear uno nuevo.

    Returns:
        Modelo PPO listo para entrenar.
    """
    if resume_path:
        print(f"Reanudando entrenamiento desde: {resume_path}")
        model = PPO.load(resume_path, env=env, device=cfg.device)
        return model

    lr = cfg.learning_rate
    if cfg.use_linear_lr_schedule:
        lr = get_linear_fn(cfg.learning_rate, cfg.learning_rate * 0.1, 1.0)

    model = PPO(
        policy=cfg.policy,
        env=env,
        learning_rate=lr,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        tensorboard_log=str(cfg.log_dir),
        device=cfg.device,
        verbose=1,
    )
    return model


def main():
    args = parse_args()
    cfg = Config()
    cfg.ensure_dirs()

    if args.timesteps:
        cfg.total_timesteps = args.timesteps

    print("=" * 60)
    print("  Entrenamiento de agente Mario Bros 1-2 con PPO")
    print("=" * 60)
    print(f"  Device:     {cfg.device}")
    print(f"  Timesteps:  {cfg.total_timesteps:,}")
    print(f"  LR:         {cfg.learning_rate}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Env:        {cfg.env_id}")
    print("=" * 60)

    # Crear environment
    env = make_env(cfg)

    # Crear o cargar modelo
    resume_path = args.model_path if args.resume else None
    if args.resume and not resume_path:
        resume_path = str(cfg.model_save_path)
    model = make_model(cfg, env, resume_path)

    # Callback
    callback = MarioTrainingCallback(cfg, verbose=1)

    # Entrenar
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callback,
            tb_log_name="mario_ppo",
        )
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")

    # Guardar modelo final
    model.save(str(cfg.model_save_path))
    print(f"\nModelo guardado en: {cfg.model_save_path}")

    env.close()


if __name__ == "__main__":
    main()
