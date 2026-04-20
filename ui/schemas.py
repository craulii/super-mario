"""Schemas pydantic para la API del dashboard."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    config_path: str = "configs/default.yaml"
    resume_path: str | None = None


class SavePathRequest(BaseModel):
    path: str


class RewardPatch(BaseModel):
    """Campos editables en caliente. Todos opcionales para parches parciales."""

    forward_reward_coef: float | None = None
    backward_reward_coef: float | None = None
    coin_reward: float | None = None
    score_reward_coef: float | None = None
    vertical_explore_reward: float | None = None
    time_penalty_coef: float | None = None
    time_remaining_bonus_coef: float | None = None
    flag_bonus: float | None = None
    death_by_enemy_penalty: float | None = None
    death_by_time_penalty: float | None = None
    stuck_penalty_base: float | None = None
    stuck_threshold_base: int | None = None
    enable_cyclic_detection: bool | None = None
    cyclic_action_penalty: float | None = None
    enable_death_map: bool | None = None
    death_bucket_coef: float | None = None
    death_bucket_cap: float | None = None
    enable_excessive_left: bool | None = None
    excessive_left_penalty: float | None = None
    enable_micro_movement: bool | None = None
    micro_movement_penalty: float | None = None
    enable_wall_stuck: bool | None = None
    wall_stuck_penalty: float | None = None
    enable_records: bool | None = None
    record_distance_bonus: float | None = None
    record_time_bonus: float | None = None


class ConfigPatch(BaseModel):
    reward: RewardPatch | None = None


class ConfigPatchResponse(BaseModel):
    applied: list[str]
    ignored: list[str] = Field(default_factory=list)
    requires_restart: list[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    mode: str
    message: str = ""
    timesteps: int
    episodes: int
    current_x: int
    max_x_this_ep: int
    max_x_historical: float
    reward_avg_100: float
    distance_avg_100: float
    clear_rate_100: float
    coins_avg_100: float
    behavior_score: float
    last_episode_reward: float
    last_clear: bool
    zone_survival: dict[int, float]
    paused: bool
    stopping: bool


class ActionResponse(BaseModel):
    ok: bool
    message: str = ""
    data: dict[str, Any] | None = None
