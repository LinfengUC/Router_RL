from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

from .config import MODEL_COSTS, REWARD_PROFILES, PRESETS, Preset
from .data import load_labels_csv, load_irt_csv, intersect_models
from .response_store import load_responses, available_models
from .env import ModelRoutingEnv
from .embedder import TextEmbedder
from .wrappers import EmbeddingWrapper
from .callbacks import RouterEvalCallback, MetricsWriter


def build_model_costs(model_names: List[str]) -> Dict[str, Dict[str, float]]:
    missing = [m for m in model_names if m not in MODEL_COSTS]
    if missing:
        raise ValueError(
            "Missing cost/latency entries in MODEL_COSTS for: " + ", ".join(missing)
        )
    return {m: MODEL_COSTS[m] for m in model_names}


def make_env(
    *,
    dataset,
    responses,
    model_names: List[str],
    model_costs,
    reward_cfg: Dict[str, Any],
    irt_info=None,
    hard_gate: bool = False,
    invalid_action_penalty: float = -3.0,
    add_irt_features: bool = False,
    base_model_name: str = "deepseek_V3.2_no_reasoning",
    embedder: Optional[TextEmbedder] = None,
    embedder_device: str = "auto",
) -> gym.Env:
    base_env = ModelRoutingEnv(
        dataset=dataset,
        responses=responses,
        model_names=model_names,
        model_costs=model_costs,
        irt_info=irt_info,
        hard_gate=hard_gate,
        invalid_action_penalty=invalid_action_penalty,
        base_model_name=base_model_name,
        max_response_chars=5000,
        **reward_cfg,
    )

    if embedder is None:
        embedder = TextEmbedder(device=embedder_device)

    env = EmbeddingWrapper(base_env, embedder=embedder, add_irt_features=add_irt_features)
    return env


def run_experiment(
    *,
    mode: str,
    preset_name: str,
    profiles: List[str],
    data_dir: str,
    output_root: str,
    base_seed: int = 42,
    runs_override: Optional[int] = None,
    timesteps_override: Optional[int] = None,
    eval_freq_override: Optional[int] = None,
    n_eval_episodes_override: Optional[int] = None,
    embedder_device: str = "auto",
    ppo_device: str = "cpu",
):
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Choose from {list(PRESETS)}")
    preset: Preset = PRESETS[preset_name]

    timesteps = int(timesteps_override or preset.timesteps)
    eval_freq = int(eval_freq_override or preset.eval_freq)
    n_eval_episodes = int(n_eval_episodes_override or preset.n_eval_episodes)
    n_runs = int(runs_override or preset.n_runs)

    # Load normalized data
    labels_train, models_train = load_labels_csv(os.path.join(data_dir, "labels", "train.csv"))
    labels_test, models_test = load_labels_csv(os.path.join(data_dir, "labels", "test.csv"))

    responses_dir = os.path.join(data_dir, "responses")
    resp_models_train = available_models(responses_dir, "train")
    resp_models_test = available_models(responses_dir, "test")

    common_models = sorted(list(intersect_models(models_train, models_test, resp_models_train, resp_models_test)))
    if not common_models:
        raise ValueError("No common models between labels and responses for both splits.")

    model_costs = build_model_costs(common_models)

    responses_train = load_responses(responses_dir, "train", common_models)
    responses_test = load_responses(responses_dir, "test", common_models)

    # IRT data (optional)
    irt_train = None
    irt_test = None
    if mode in ("irt", "hard_gate"):
        irt_train = load_irt_csv(os.path.join(data_dir, "irt", "train.csv"))
        irt_test = load_irt_csv(os.path.join(data_dir, "irt", "test.csv"))

    # Mode knobs
    hard_gate = (mode == "hard_gate")
    add_irt_features = (mode == "irt")
    invalid_action_penalty = -3.0

    # Keep these explicit so inference can reproduce routing.
    base_model_name = "deepseek_V3.2_no_reasoning"
    embedder_model_name = "sentence-transformers/all-mpnet-base-v2"

    # Shared embedder instance (caches across envs/runs)
    embedder = TextEmbedder(model_name=embedder_model_name, device=embedder_device)

    # PPO config (same as original scripts)
    policy_kwargs = dict(net_arch=dict(pi=[512, 512], vf=[512, 512]))

    os.makedirs(output_root, exist_ok=True)

    for profile in profiles:
        if profile not in REWARD_PROFILES:
            raise ValueError(f"Unknown profile {profile!r}. Choose from {list(REWARD_PROFILES)}")

        reward_cfg = REWARD_PROFILES[profile]

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{mode}_{profile}_seed{seed}"
            log_dir = os.path.join(output_root, mode, profile, run_id)
            os.makedirs(log_dir, exist_ok=True)

            best_dir = os.path.join(log_dir, "best")
            os.makedirs(best_dir, exist_ok=True)

            # Save run config for reproducibility
            config_out = {
                "mode": mode,
                "preset": preset_name,
                "profile": profile,
                "seed": seed,
                "timesteps": timesteps,
                "eval_freq": eval_freq,
                "n_eval_episodes": n_eval_episodes,
                "common_models": common_models,
                "base_model_name": base_model_name,
                "reward_cfg": reward_cfg,
                "hard_gate": hard_gate,
                "add_irt_features": add_irt_features,
                "invalid_action_penalty": invalid_action_penalty,
                "embedder_device": embedder_device,
                "embedder_model_name": embedder_model_name,
                "ppo_device": ppo_device,
            }
            with open(os.path.join(log_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(config_out, f, indent=2)

            train_env = make_env(
                dataset=labels_train,
                responses=responses_train,
                model_names=common_models,
                model_costs=model_costs,
                reward_cfg=reward_cfg,
                irt_info=irt_train,
                hard_gate=hard_gate,
                invalid_action_penalty=invalid_action_penalty,
                add_irt_features=add_irt_features,
                base_model_name=base_model_name,
                embedder=embedder,
                embedder_device=embedder_device,
            )

            eval_env = make_env(
                dataset=labels_test,
                responses=responses_test,
                model_names=common_models,
                model_costs=model_costs,
                reward_cfg=reward_cfg,
                irt_info=irt_test,
                hard_gate=hard_gate,
                invalid_action_penalty=invalid_action_penalty,
                add_irt_features=add_irt_features,
                base_model_name=base_model_name,
                embedder=embedder,
                embedder_device=embedder_device,
            )

            metrics_writer = MetricsWriter(log_dir)
            eval_callback = RouterEvalCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                verbose=1,
                save_best_path=best_dir,
                best_metric="mean_reward",
                metrics_writer=metrics_writer,
            )

            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                seed=seed,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.001,
                clip_range=0.2,
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir,
                device=ppo_device,  # default cpu to avoid GPU warnings with SB3
            )

            print(f"\n========== {mode=} {profile=} run_idx={run_idx} seed={seed} ==========")
            print(f"Log dir: {log_dir}")
            print(f"Timesteps: {timesteps}  | Eval: every {eval_freq} steps, {n_eval_episodes} episodes")

            model.learn(total_timesteps=timesteps, callback=eval_callback, log_interval=10)

            latest_path = os.path.join(log_dir, "latest_model")
            model.save(latest_path)

            print(f"Saved latest model: {latest_path}.zip")
            print(f"Best model (by {eval_callback.best_metric}) stored in: {best_dir}")
            train_env.close()
            eval_env.close()
