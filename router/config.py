"""Project-wide configuration (model costs, reward profiles, presets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


# Empirical average cost and latency per *100 questions*.
# NOTE: cost_per_100 means average USD cost for 100 questions.
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-4.1": {"cost_per_100": 0.32, "latency": 4.7},
    "gpt-5": {"cost_per_100": 1.62, "latency": 26.4},
    "gpt-5-mini": {"cost_per_100": 0.39, "latency": 15.7},
    "gpt-5-nano": {"cost_per_100": 0.06, "latency": 22.3},
    "deepseek_V3.2_reasoning": {"cost_per_100": 0.07, "latency": 54.6},
    "deepseek_V3.2_chat": {"cost_per_100": 0.02, "latency": 10.2},
    "gemini_2.5_pro": {"cost_per_100": 2.31, "latency": 43.8},
    "gemini_2.5_flash_lite": {"cost_per_100": 0.20, "latency": 4.5},
    "gemini_2.5_flash_lite_thinking": {"cost_per_100": 0.68, "latency": 8.4},
    "gemini_2.5_flash": {"cost_per_100": 0.20, "latency": 9.0},
    "gemini_2.5_flash_thinking": {"cost_per_100": 0.59, "latency": 13.0},
    "kimi_k2": {"cost_per_100": 0.08, "latency": 25.3},
}


# Reward profiles (kept identical to the original scripts).
REWARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "accuracy_first": dict(
        correct_reward=2.0,
        wrong_reward=-4.0,
        cost_weight=0.1,
        latency_weight=0.005,
        switch_penalty=0.05,
        first_switch_free=True,
    ),
    "cost_first": dict(
        correct_reward=1.0,
        wrong_reward=-1.0,
        cost_weight=0.5,
        latency_weight=0.05,
        switch_penalty=0.2,
        first_switch_free=True,
    ),
}


@dataclass(frozen=True)
class Preset:
    """A convenience bundle of training/eval hyperparameters."""

    timesteps: int
    eval_freq: int
    n_eval_episodes: int
    n_runs: int


PRESETS: Dict[str, Preset] = {
    # Quick sanity check. Intended to finish fast and verify wiring.
    "smoke": Preset(timesteps=10_000, eval_freq=2_000, n_eval_episodes=20, n_runs=1),
    # Original settings (takes much longer; use for expected-quality results).
    "full": Preset(timesteps=300_000, eval_freq=5_000, n_eval_episodes=100, n_runs=10),
}
