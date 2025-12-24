from __future__ import annotations

import random
from typing import Dict, Any, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .response_store import ResponseRecord


class ModelRoutingEnv(gym.Env):
    """Multi-step routing environment.

    - Reset:
        * samples a prompt_id
        * optionally pre-calls a base model (default deepseek_V3.2_no_reasoning)
          so the first observation includes an initial answer.

    - Actions:
        0..(N-1): call model i
        N:        ACCEPT current answer

    - Reward:
        * each model call: -cost_weight * cost_dollars - latency_weight * latency
          plus a switch penalty after the (optional) first free switch
        * accept: +correct_reward if correct else +wrong_reward
        * if max steps reached without accept: treated as wrong_reward

    Modes:
        * base: no IRT usage
        * irt: env provides get_irt_feature_vector() for wrapper to append
        * hard_gate: disallow calling models where (s_bert==0 and m_bert==0)
          for the current prompt (with a fallback that allows all if IRT says no to everyone).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: Dict[str, Dict[str, Any]],
        responses: Dict[str, Dict[str, ResponseRecord]],
        model_names: List[str],
        model_costs: Dict[str, Dict[str, float]],
        *,
        irt_info: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
        hard_gate: bool = False,
        invalid_action_penalty: float = -3.0,
        base_model_name: str = "deepseek_V3.2_no_reasoning",
        max_response_chars: int = 5000,
        correct_reward: float = 1.0,
        wrong_reward: float = -2.0,
        cost_weight: float = 0.1,
        latency_weight: float = 0.01,
        switch_penalty: float = 0.1,
        first_switch_free: bool = True,
    ):
        super().__init__()

        self.dataset = dataset
        self.responses = responses
        self.model_names = list(model_names)
        self.model_costs = model_costs

        self.irt_info = irt_info
        self.hard_gate = bool(hard_gate)
        self.invalid_action_penalty = float(invalid_action_penalty)

        self.num_models = len(self.model_names)
        self.max_response_chars = int(max_response_chars)

        self.correct_reward = float(correct_reward)
        self.wrong_reward = float(wrong_reward)
        self.cost_weight = float(cost_weight)
        self.latency_weight = float(latency_weight)
        self.switch_penalty = float(switch_penalty)
        self.first_switch_free = bool(first_switch_free)

        self.base_model_name = base_model_name

        # Valid prompt_ids: must have labels AND responses for ALL models
        prompt_ids: List[str] = []
        for pid, d in self.dataset.items():
            has_all_labels = all(m in d["labels"] for m in self.model_names)
            has_all_responses = all(pid in self.responses[m] for m in self.model_names)
            if has_all_labels and has_all_responses:
                prompt_ids.append(pid)
        if not prompt_ids:
            raise ValueError("No prompts with both labels and responses for all models.")
        self.prompt_ids = prompt_ids

        # Actions: [0..N-1] model call; N = ACCEPT
        self.action_space = spaces.Discrete(self.num_models + 1)

        # Text dict obs (wrapper converts to embeddings)
        self.observation_space = spaces.Dict(
            {
                "question": spaces.Text(max_length=10_000),
                "current_output": spaces.Text(max_length=self.max_response_chars),
            }
        )

        # Episode state
        self.current_prompt_id: Optional[str] = None
        self.current_prompt_text: str = ""
        self.current_model: Optional[str] = None
        self.current_response_text: str = ""

        self.total_cost_dollars: float = 0.0
        self.total_latency: float = 0.0
        self.num_calls: int = 0
        self.step_count: int = 0
        self.max_steps: int = 2 * self.num_models  # safety cap

        # Hard-gate mask (bool per model action)
        self.allowed_mask = np.ones(self.num_models, dtype=bool)

    # ---------- helpers ----------

    def _truncate_from_end(self, text: str) -> str:
        if text is None:
            return ""
        if len(text) <= self.max_response_chars:
            return text
        return text[-self.max_response_chars :]

    def _build_truncated_reasoning_text(self, output: str, reasoning: str) -> str:
        output = output or ""
        reasoning = reasoning or ""
        max_chars = self.max_response_chars

        remaining = max_chars - len(output) - 1  # -1 for newline
        if remaining <= 0:
            return output

        tail = reasoning if len(reasoning) <= remaining else reasoning[-remaining:]
        return output + "\n" + tail

    def _compute_call_cost_and_latency(self, model_name: str) -> tuple[float, float]:
        cfg = self.model_costs[model_name]
        cost_dollars = cfg["cost_per_100"] / 100.0
        latency = cfg["latency"]
        return float(cost_dollars), float(latency)

    def _update_allowed_mask(self):
        n = self.num_models
        mask = np.ones(n, dtype=bool)

        if not self.hard_gate or self.irt_info is None or self.current_prompt_id is None:
            self.allowed_mask = mask
            return

        info_for_prompt = self.irt_info.get(self.current_prompt_id, {})
        any_true = False
        new_mask = np.zeros(n, dtype=bool)

        for i, name in enumerate(self.model_names):
            preds = info_for_prompt.get(name)
            if preds is None:
                new_mask[i] = True
            else:
                allowed = (preds.get("s_bert_pred", 0) == 1) or (preds.get("m_bert_pred", 0) == 1)
                new_mask[i] = bool(allowed)
                if allowed:
                    any_true = True

        # If IRT says "no" to everyone, allow all.
        if not any_true:
            new_mask[:] = True

        self.allowed_mask = new_mask

    def get_irt_feature_vector(self) -> np.ndarray:
        """Vector [s_0, m_0, s_1, m_1, ...] for the *current* prompt."""
        n = self.num_models
        feat = np.zeros(2 * n, dtype=np.float32)
        if self.irt_info is None or self.current_prompt_id is None:
            return feat

        info_for_prompt = self.irt_info.get(self.current_prompt_id, {})
        idx = 0
        for name in self.model_names:
            preds = info_for_prompt.get(name, {})
            s = float(int(preds.get("s_bert_pred", 0)))
            m = float(int(preds.get("m_bert_pred", 0)))
            feat[idx] = s
            feat[idx + 1] = m
            idx += 2
        return feat

    def _initial_base_model_call(self, model_name: str):
        if model_name not in self.model_names:
            # No base model in this experiment.
            return

        rec = self.responses[model_name][self.current_prompt_id]  # type: ignore[index]

        full_for_cost = rec.full_text_for_cost()

        # For embedding: deepseek reasoning gets special handling; others truncate from end.
        if model_name == "deepseek_V3.2_reasoning":
            truncated_for_embed = self._build_truncated_reasoning_text(rec.output, rec.reasoning)
        else:
            truncated_for_embed = self._truncate_from_end(full_for_cost)

        self.current_model = model_name
        self.current_response_text = truncated_for_embed
        self.num_calls = 1

        cost, latency = self._compute_call_cost_and_latency(model_name)
        self.total_cost_dollars = cost
        self.total_latency = latency

    # ---------- gym API ----------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.total_cost_dollars = 0.0
        self.total_latency = 0.0
        self.num_calls = 0
        self.current_model = None
        self.current_response_text = ""

        idx = self.np_random.integers(0, len(self.prompt_ids))
        self.current_prompt_id = self.prompt_ids[int(idx)]
        self.current_prompt_text = self.dataset[self.current_prompt_id]["prompt"]

        self._update_allowed_mask()
        self._initial_base_model_call(self.base_model_name)

        obs = {"question": self.current_prompt_text, "current_output": self.current_response_text or ""}
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        done = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {}

        accept_action = self.num_models

        if int(action) == accept_action:
            # ACCEPT
            if self.current_model is None:
                correct = False
            else:
                label = self.dataset[self.current_prompt_id]["labels"][self.current_model]  # type: ignore[index]
                correct = bool(label)

            reward += self.correct_reward if correct else self.wrong_reward
            done = True
            info = {
                "final_model_name": self.current_model,
                "final_correct": correct,
                "total_cost_dollars": self.total_cost_dollars,
                "total_latency": self.total_latency,
                "num_calls": self.num_calls,
            }

            obs = {"question": self.current_prompt_text, "current_output": self.current_response_text or ""}
            return obs, reward, done, truncated, info

        # CALL a model
        model_index = int(action)
        if model_index < 0 or model_index >= self.num_models:
            reward += self.wrong_reward
            done = True
            info = {
                "final_model_name": None,
                "final_correct": False,
                "total_cost_dollars": self.total_cost_dollars,
                "total_latency": self.total_latency,
                "num_calls": self.num_calls,
                "invalid_action": True,
            }
            obs = {"question": self.current_prompt_text, "current_output": self.current_response_text or ""}
            return obs, reward, done, truncated, info

        if self.hard_gate and not bool(self.allowed_mask[model_index]):
            reward += self.invalid_action_penalty
            done = True
            info = {
                "final_model_name": None,
                "final_correct": False,
                "total_cost_dollars": self.total_cost_dollars,
                "total_latency": self.total_latency,
                "num_calls": self.num_calls,
                "invalid_action": True,
            }
            obs = {"question": self.current_prompt_text, "current_output": self.current_response_text or ""}
            return obs, reward, done, truncated, info

        model_name = self.model_names[model_index]
        self.current_model = model_name
        self.num_calls += 1

        rec = self.responses[model_name][self.current_prompt_id]  # type: ignore[index]
        full_for_cost = rec.full_text_for_cost()

        if model_name == "deepseek_V3.2_reasoning":
            truncated_for_embed = self._build_truncated_reasoning_text(rec.output, rec.reasoning)
        else:
            truncated_for_embed = self._truncate_from_end(full_for_cost)

        self.current_response_text = truncated_for_embed

        cost_dollars, latency = self._compute_call_cost_and_latency(model_name)
        self.total_cost_dollars += cost_dollars
        self.total_latency += latency

        reward -= self.cost_weight * cost_dollars
        reward -= self.latency_weight * latency

        # Extra call penalty: first switch can be free.
        if self.num_calls > 1:
            if not (self.first_switch_free and self.num_calls == 2):
                reward -= self.switch_penalty

        obs = {"question": self.current_prompt_text, "current_output": self.current_response_text or ""}

        if self.step_count >= self.max_steps and not done:
            done = True
            reward += self.wrong_reward
            info.setdefault("final_model_name", self.current_model)
            info.setdefault("final_correct", False)
            info.setdefault("total_cost_dollars", self.total_cost_dollars)
            info.setdefault("total_latency", self.total_latency)
            info.setdefault("num_calls", self.num_calls)
            info["terminated_by_max_steps"] = True

        return obs, reward, done, truncated, info
