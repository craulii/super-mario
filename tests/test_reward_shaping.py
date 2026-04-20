"""Tests del AdvancedRewardWrapper usando un env sintético.

No ejecuta el ROM. Simplemente alimenta `info` dicts controlados para verificar
que cada componente del reward shaping contribuye como se espera.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configs.schema import RewardConfig
from env.death_map import DeathLocationTracker
from env.reward_shaping import AdvancedRewardWrapper


def _make_wrapper(dummy_env, tmp_path: Path, **overrides) -> AdvancedRewardWrapper:
    cfg = RewardConfig(**overrides) if overrides else RewardConfig()
    death_map = DeathLocationTracker(tmp_path / "dm.json", bucket_size=cfg.death_bucket_size)
    return AdvancedRewardWrapper(
        dummy_env,
        reward_cfg=cfg,
        death_map=death_map,
        best_distance_path=tmp_path / "bd.json",
        best_time_path=tmp_path / "bt.json",
    )


def test_forward_progress_positive(dummy_env, tmp_path):
    w = _make_wrapper(dummy_env, tmp_path)
    w.reset()
    dummy_env.set_next_info({"x_pos": 10, "y_pos": 79, "time": 400, "coins": 0,
                             "score": 0, "life": 2, "flag_get": False})
    _, reward, *_ = w.step(1)
    assert reward > 0


def test_coin_bonus(dummy_env, tmp_path):
    w = _make_wrapper(dummy_env, tmp_path, enable_cyclic_detection=False,
                      enable_micro_movement=False, enable_wall_stuck=False,
                      enable_records=False, enable_death_map=False,
                      enable_excessive_left=False)
    w.reset()
    dummy_env.set_next_info({"x_pos": 10, "y_pos": 79, "time": 400, "coins": 1,
                             "score": 0, "life": 2, "flag_get": False})
    _, reward, *_ = w.step(1)
    assert reward >= 10 * 1.0 + 1 * 5.0  # forward + coin


def test_death_by_enemy_penalty(dummy_env, tmp_path):
    w = _make_wrapper(dummy_env, tmp_path, enable_records=False)
    w.reset()
    dummy_env.set_next_info({"x_pos": 0, "y_pos": 79, "time": 300, "coins": 0,
                             "score": 0, "life": 1, "flag_get": False}, done=True)
    _, reward, terminated, truncated, info = w.step(1)
    assert terminated or truncated
    assert info["died"] is True
    assert reward < 0


def test_death_by_time_penalty_is_softer(dummy_env, tmp_path):
    w1 = _make_wrapper(dummy_env, tmp_path, enable_death_map=False)
    w1.reset()
    dummy_env.set_next_info({"x_pos": 200, "y_pos": 79, "time": 1, "coins": 0,
                             "score": 0, "life": 1, "flag_get": False}, done=True)
    _, r_time, *_ = w1.step(1)

    dummy_env2 = type(dummy_env)()
    w2 = _make_wrapper(dummy_env2, tmp_path, enable_death_map=False)
    w2.reset()
    dummy_env2.set_next_info({"x_pos": 200, "y_pos": 79, "time": 300, "coins": 0,
                              "score": 0, "life": 1, "flag_get": False}, done=True)
    _, r_enemy, *_ = w2.step(1)

    assert r_time > r_enemy


def test_flag_completion_bonus(dummy_env, tmp_path):
    w = _make_wrapper(dummy_env, tmp_path, enable_cyclic_detection=False)
    w.reset()
    dummy_env.set_next_info({"x_pos": 3300, "y_pos": 79, "time": 250, "coins": 5,
                             "score": 1000, "life": 2, "flag_get": True}, done=True)
    _, reward, terminated, truncated, info = w.step(1)
    assert terminated or truncated
    assert info["died"] is False
    assert reward > 15


def test_wall_stuck_detection(dummy_env, tmp_path):
    w = _make_wrapper(dummy_env, tmp_path, enable_cyclic_detection=False,
                      enable_micro_movement=False, enable_excessive_left=False,
                      enable_records=False, enable_death_map=False,
                      wall_stuck_frames=3, wall_stuck_penalty=-10.0)
    w.reset()
    total = 0.0
    for _ in range(5):
        dummy_env.set_next_info({"x_pos": 0, "y_pos": 79, "time": 400, "coins": 0,
                                 "score": 0, "life": 2, "flag_get": False})
        _, r, *_ = w.step(1)
        total += r
    assert total < 0


def test_reward_config_changes_apply_live(dummy_env, tmp_path):
    w = _make_wrapper(dummy_env, tmp_path, enable_cyclic_detection=False,
                      enable_micro_movement=False, enable_wall_stuck=False,
                      enable_excessive_left=False, enable_records=False,
                      enable_death_map=False)
    w.reset()
    dummy_env.set_next_info({"x_pos": 10, "y_pos": 79, "time": 400, "coins": 0,
                             "score": 0, "life": 2, "flag_get": False})
    _, r1, *_ = w.step(1)

    w.reward_cfg.forward_reward_coef = 10.0
    dummy_env.set_next_info({"x_pos": 20, "y_pos": 79, "time": 400, "coins": 0,
                             "score": 0, "life": 2, "flag_get": False})
    _, r2, *_ = w.step(1)

    assert r2 > r1


def test_info_enriched(dummy_env, tmp_path):
    w = _make_wrapper(dummy_env, tmp_path)
    w.reset()
    dummy_env.set_next_info({"x_pos": 1500, "y_pos": 79, "time": 400, "coins": 0,
                             "score": 0, "life": 2, "flag_get": False})
    _, _, _, _, info = w.step(1)
    assert "progress_ratio" in info
    assert "zone" in info
    assert "max_x_this_ep" in info
    assert 0.0 <= info["progress_ratio"] <= 1.0
