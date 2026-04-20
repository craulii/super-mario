"""Script para evaluar y visualizar el agente entrenado."""

import argparse
import time

import numpy as np
from gym.wrappers import RecordVideo
from stable_baselines3 import PPO

from config import Config
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Evaluar agente Mario entrenado")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Ruta al modelo entrenado (default: models/mario_ppo)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Número de episodios a evaluar (default: 5)")
    parser.add_argument("--record", action="store_true",
                        help="Grabar video del gameplay")
    parser.add_argument("--no-render", action="store_true",
                        help="No renderizar visualmente (útil con --record)")
    return parser.parse_args()


def evaluate(model_path: str, n_episodes: int, record: bool, render: bool):
    """Ejecuta la evaluación del agente.

    Args:
        model_path: Ruta al modelo .zip entrenado.
        n_episodes: Cantidad de episodios a ejecutar.
        record: Si True, graba video del gameplay.
        render: Si True, renderiza el juego en pantalla.
    """
    cfg = Config()

    # Crear environment con o sin render
    render_mode = "human" if render else None
    env = make_env(cfg, render_mode=render_mode)

    # Envolver con RecordVideo si se pide grabar
    if record:
        cfg.video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, str(cfg.video_dir), episode_trigger=lambda _: True)
        print(f"Grabando videos en: {cfg.video_dir}")

    # Cargar modelo
    model = PPO.load(model_path, device=cfg.device)
    print(f"Modelo cargado desde: {model_path}")
    print(f"Evaluando {n_episodes} episodios...\n")

    # Métricas
    rewards = []
    distances = []
    flags = []
    coins_list = []
    scores_list = []
    times_left = []
    times_elapsed = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        max_x = 0
        flag_get = False
        ep_coins = 0
        ep_score = 0
        ep_time_left = 0
        start = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            max_x = max(max_x, info.get("x_pos", 0))
            flag_get = info.get("flag_get", False)
            ep_coins = info.get("coins", 0)
            ep_score = info.get("score", 0)
            ep_time_left = info.get("time", 0)

        elapsed = time.time() - start
        rewards.append(total_reward)
        distances.append(max_x)
        flags.append(flag_get)
        coins_list.append(ep_coins)
        scores_list.append(ep_score)
        times_left.append(ep_time_left)
        times_elapsed.append(elapsed)

        status = "COMPLETADO" if flag_get else "NO completado"
        print(f"  Ep {ep}/{n_episodes} | Reward: {total_reward:8.1f} | Dist: {max_x:5d} | Coins: {ep_coins:2d} | Score: {ep_score:6d} | T.Left: {ep_time_left:3d} | {status} | {elapsed:.1f}s")

    env.close()

    # Resumen
    print("\n" + "=" * 60)
    print("  RESUMEN DE EVALUACION")
    print("=" * 60)
    print(f"  Episodios:           {n_episodes}")
    print(f"  Reward promedio:     {np.mean(rewards):.1f} (+/- {np.std(rewards):.1f})")
    print(f"  Distancia maxima:    {max(distances)}")
    print(f"  Distancia promedio:  {np.mean(distances):.0f}")
    print(f"  Monedas promedio:    {np.mean(coins_list):.1f}")
    print(f"  Score promedio:      {np.mean(scores_list):.0f}")
    print(f"  Niveles completados: {sum(flags)}/{n_episodes} ({100*np.mean(flags):.0f}%)")
    if sum(flags) > 0:
        clear_times = [t for t, f in zip(times_left, flags) if f]
        print(f"  Tiempo restante avg: {np.mean(clear_times):.0f} (en niveles completados)")
    print(f"  Tiempo real prom:    {np.mean(times_elapsed):.1f}s")
    print("=" * 60)


def main():
    args = parse_args()
    cfg = Config()

    model_path = args.model_path or str(cfg.model_save_path)
    render = not args.no_render

    evaluate(model_path, args.episodes, args.record, render)


if __name__ == "__main__":
    main()
