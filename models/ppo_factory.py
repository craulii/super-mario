"""Factory para construir modelos PPO con la configuración del proyecto."""

from __future__ import annotations

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn

from configs.schema import Config


def make_model(cfg: Config, env: gym.Env) -> PPO:
    """Crea un modelo PPO desde cero con la configuración dada."""
    lr: float | object = cfg.ppo.learning_rate
    if cfg.ppo.use_linear_lr_schedule:
        lr = get_linear_fn(cfg.ppo.learning_rate, cfg.ppo.learning_rate * 0.1, 1.0)

    return PPO(
        policy=cfg.ppo.policy,
        env=env,
        learning_rate=lr,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        tensorboard_log=str(cfg.paths.log_dir),
        device=cfg.training.device,
        verbose=0,
        seed=cfg.training.seed,
    )
