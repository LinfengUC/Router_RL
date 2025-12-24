from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class EvalMetrics:
    step: int
    wall_time_sec: float
    accuracy: float
    mean_reward: float
    mean_cost: float
    mean_latency: float
    mean_calls: float
    per_model_final_usage: Dict[str, float]


class MetricsWriter:
    """Append metrics to both JSONL and CSV."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, "metrics.jsonl")
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self._csv_initialized = False

    def write(self, m: EvalMetrics):
        # JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

        # CSV (flat per_model_final_usage into columns)
        row = asdict(m).copy()
        usage = row.pop("per_model_final_usage")
        for k, v in usage.items():
            row[f"final_usage__{k}"] = v

        fieldnames = list(row.keys())
        if not self._csv_initialized and not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
            self._csv_initialized = True

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)


class RouterEvalCallback(BaseCallback):
    """Evaluation callback for the router environment.

    Logs:
      - accuracy
      - mean_reward
      - mean_cost (USD per question)
      - mean_latency (seconds per question)
      - mean_calls (# model calls per episode)
      - per-model final usage

    Saves:
      - best model checkpoint by best_metric (default: mean_reward)
      - metrics.jsonl and metrics.csv in the run log directory
    """

    def __init__(
        self,
        eval_env,
        *,
        eval_freq: int = 5_000,
        n_eval_episodes: int = 100,
        verbose: int = 1,
        save_best_path: Optional[str] = None,
        best_metric: str = "mean_reward",
        metrics_writer: Optional[MetricsWriter] = None,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.save_best_path = save_best_path
        self.best_metric = best_metric
        self.best_score = -float("inf")
        self.metrics_writer = metrics_writer

        if self.save_best_path is not None:
            os.makedirs(self.save_best_path, exist_ok=True)

        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            metrics = self._evaluate()
            # SB3 logger
            self.logger.record("eval/accuracy", metrics.accuracy)
            self.logger.record("eval/mean_reward", metrics.mean_reward)
            self.logger.record("eval/mean_cost", metrics.mean_cost)
            self.logger.record("eval/mean_latency", metrics.mean_latency)
            self.logger.record("eval/mean_calls", metrics.mean_calls)
            for name, frac in metrics.per_model_final_usage.items():
                self.logger.record(f"eval/final_usage/{name}", frac)

            if self.metrics_writer is not None:
                self.metrics_writer.write(metrics)

            # Save best
            score = getattr(metrics, self.best_metric)
            if score > self.best_score:
                self.best_score = score
                if self.save_best_path is not None:
                    best_path = os.path.join(self.save_best_path, "best_model")
                    self.model.save(best_path)
                    if self.verbose:
                        print(f"[Eval] New best {self.best_metric}={score:.4f}; saved to {best_path}.zip")

        return True

    def _evaluate(self) -> EvalMetrics:
        base_env = self.eval_env.unwrapped
        model_names: List[str] = list(base_env.model_names)

        counts = {name: 0 for name in model_names}
        total_correct = 0
        total_rewards = 0.0
        total_cost = 0.0
        total_latency = 0.0
        total_calls = 0.0

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            final_info: Dict[str, Any] = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
                if done:
                    final_info = info

            total_rewards += ep_reward
            total_correct += int(final_info.get("final_correct", False))
            total_cost += float(final_info.get("total_cost_dollars", 0.0))
            total_latency += float(final_info.get("total_latency", 0.0))
            total_calls += float(final_info.get("num_calls", 0.0))

            final_model = final_info.get("final_model_name")
            if final_model in counts:
                counts[final_model] += 1

        accuracy = total_correct / self.n_eval_episodes
        mean_reward = total_rewards / self.n_eval_episodes
        mean_cost = total_cost / self.n_eval_episodes
        mean_latency = total_latency / self.n_eval_episodes
        mean_calls = total_calls / self.n_eval_episodes

        per_model_usage = {name: counts[name] / self.n_eval_episodes for name in model_names}

        return EvalMetrics(
            step=int(self.n_calls),
            wall_time_sec=float(time.time() - self._start_time),
            accuracy=float(accuracy),
            mean_reward=float(mean_reward),
            mean_cost=float(mean_cost),
            mean_latency=float(mean_latency),
            mean_calls=float(mean_calls),
            per_model_final_usage=per_model_usage,
        )
