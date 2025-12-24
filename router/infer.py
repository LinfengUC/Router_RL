from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from .config import MODEL_COSTS
from .data import IRTInfo, load_irt_csv
from .embedder import TextEmbedder
from .response_store import load_responses, ResponseMap


# -----------------------------
# Public types
# -----------------------------


@dataclass
class RouteResult:
    prompt_id: str
    chosen_model: Optional[str]
    accepted: bool
    num_calls: int
    est_cost_dollars: float
    est_latency_seconds: float
    action_trace: List[int]


class ModelCaller:
    """Interface for calling a model by name.

    Implement this in production to call your real model endpoints.

    Must return a plain text output string.
    """

    def call(self, *, model_name: str, prompt_id: str, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class StoredResponseCaller(ModelCaller):
    """Demo-only ModelCaller that reads from data/responses/<split>/<model>.csv.

    This lets you run `predict` end-to-end without having access to real model
    endpoints. In production, replace this with your own API caller.
    """

    def __init__(self, responses_dir: str, split: str, model_names: Sequence[str]):
        self.responses: ResponseMap = load_responses(responses_dir, split, model_names)

    def call(self, *, model_name: str, prompt_id: str, prompt: str) -> str:
        rec = self.responses[model_name].get(str(prompt_id))
        if rec is None:
            raise KeyError(f"No stored response for model={model_name} prompt_id={prompt_id}")
        return rec.full_text_for_cost()


class IRTProvider:
    """Interface for providing IRT predictions.

    Return a tuple (s_bert_pred, m_bert_pred) where each element is 0/1 or None.
    If only one is available, return (pred, None) or (None, pred).
    """

    def predict(self, *, model_name: str, prompt_id: str, prompt: str) -> Tuple[Optional[int], Optional[int]]:  # pragma: no cover
        raise NotImplementedError


class CsvIRTProvider(IRTProvider):
    def __init__(self, irt_info: IRTInfo):
        self.irt_info = irt_info

    def predict(self, *, model_name: str, prompt_id: str, prompt: str) -> Tuple[Optional[int], Optional[int]]:
        preds = self.irt_info.get(str(prompt_id), {}).get(model_name)
        if not preds:
            return None, None
        return preds.get("s_bert_pred"), preds.get("m_bert_pred")


class DummySingleIRTProvider(IRTProvider):
    """Example IRT provider that returns only ONE IRT signal.

    This intentionally simulates the real-world situation where only one IRT
    model exists / is available. We return (pred, None) and downstream code
    mirrors it into the missing slot.
    """

    def predict(self, *, model_name: str, prompt_id: str, prompt: str) -> Tuple[Optional[int], Optional[int]]:
        # Toy heuristic: longer prompts are "harder" -> predict 0 sometimes.
        pred = 1 if len(prompt) < 500 else 0
        return pred, None


# -----------------------------
# Loading helpers
# -----------------------------


def resolve_checkpoint_path(path: str) -> str:
    """Accept either a directory or a model zip prefix."""
    if os.path.isdir(path):
        # common layouts produced by training
        for cand in [
            os.path.join(path, "latest_model.zip"),
            os.path.join(path, "latest_model"),
            os.path.join(path, "best", "best_model.zip"),
            os.path.join(path, "best", "best_model"),
        ]:
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"No model checkpoint found under directory: {path}")
    return path


def load_run_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_router(checkpoint_path: str, *, run_config_path: Optional[str] = None) -> Tuple[PPO, Dict[str, Any]]:
    ckpt = resolve_checkpoint_path(checkpoint_path)

    if run_config_path is None:
        # Try to infer run_config.json next to checkpoint.
        base_dir = os.path.dirname(ckpt)
        cand = os.path.join(base_dir, "run_config.json")
        if os.path.exists(cand):
            run_config_path = cand

    if run_config_path is None or not os.path.exists(run_config_path):
        raise FileNotFoundError(
            "Could not find run_config.json. Provide --run-config explicitly (it is written by training)."
        )

    cfg = load_run_config(run_config_path)
    model = PPO.load(ckpt, device=cfg.get("ppo_device", "cpu"))
    return model, cfg


# -----------------------------
# Core routing logic
# -----------------------------


def _coerce01(x: Optional[int]) -> Optional[int]:
    if x is None:
        return None
    return 1 if int(x) == 1 else 0


def _fill_missing_irt(s: Optional[int], m: Optional[int]) -> Tuple[int, int]:
    """Apply the "mirror missing" rule.

    - If one IRT signal missing, copy the existing one.
    - If both missing, default (1,1) to avoid hard failures.
    """
    s = _coerce01(s)
    m = _coerce01(m)
    if s is None and m is not None:
        s = m
    if m is None and s is not None:
        m = s
    if s is None and m is None:
        s, m = 1, 1
    return int(s), int(m)


def build_irt_feature_vector(
    *,
    model_names: Sequence[str],
    prompt_id: str,
    prompt: str,
    irt_provider: Optional[IRTProvider],
) -> np.ndarray:
    """Return [s0, m0, s1, m1, ...] aligned to model_names."""
    feat = np.zeros(2 * len(model_names), dtype=np.float32)
    if irt_provider is None:
        return feat
    j = 0
    for name in model_names:
        s, m = irt_provider.predict(model_name=name, prompt_id=prompt_id, prompt=prompt)
        s2, m2 = _fill_missing_irt(s, m)
        feat[j] = float(s2)
        feat[j + 1] = float(m2)
        j += 2
    return feat


def build_allowed_mask(
    *,
    model_names: Sequence[str],
    prompt_id: str,
    prompt: str,
    irt_provider: Optional[IRTProvider],
) -> np.ndarray:
    """Hard-gate mask: allow a model if (s==1 or m==1)."""
    mask = np.ones(len(model_names), dtype=bool)
    if irt_provider is None:
        return mask

    any_true = False
    for i, name in enumerate(model_names):
        s, m = irt_provider.predict(model_name=name, prompt_id=prompt_id, prompt=prompt)
        s2, m2 = _fill_missing_irt(s, m)
        allowed = (s2 == 1) or (m2 == 1)
        mask[i] = bool(allowed)
        if allowed:
            any_true = True

    # Same fallback as training: if IRT says "no" to everyone, allow all.
    if not any_true:
        mask[:] = True
    return mask


def _policy_best_allowed_action(model: PPO, obs: np.ndarray, allowed_mask: np.ndarray) -> int:
    """Pick the highest-probability action subject to a model-call mask.

    allowed_mask is length N over model-call actions [0..N-1].
    ACCEPT action is N and is always allowed.
    """
    n = int(len(allowed_mask))

    # Get action probabilities from the policy.
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    dist = model.policy.get_distribution(obs_t)
    probs_t = getattr(dist.distribution, "probs", None)
    if probs_t is None:
        # Fall back to logits -> softmax
        logits_t = getattr(dist.distribution, "logits", None)
        if logits_t is None:
            # As a last resort, just use the deterministic action.
            a, _ = model.predict(obs, deterministic=True)
            return int(a)
        probs_t = torch.softmax(logits_t, dim=-1)

    probs = probs_t.detach().cpu().numpy().reshape(-1)

    # Always allow ACCEPT.
    best_a = n
    best_p = probs[n]

    for i in range(n):
        if not bool(allowed_mask[i]):
            continue
        if probs[i] > best_p:
            best_p = probs[i]
            best_a = i

    return int(best_a)


def route_prompt(
    *,
    sb3_model: PPO,
    mode: str,
    embedder: TextEmbedder,
    model_names: Sequence[str],
    model_caller: ModelCaller,
    prompt_id: str,
    prompt: str,
    irt_provider: Optional[IRTProvider] = None,
    base_model_name: str = "deepseek_V3.2_no_reasoning",
    deterministic: bool = True,
    max_steps: Optional[int] = None,
) -> RouteResult:
    """Route a single prompt using a trained RL router.

    This is the *online* version of the training environment:
      1) call base_model_name once to get an initial answer
      2) embed (prompt, current_output)
      3) let the policy choose ACCEPT or switch to another model
      4) repeat until ACCEPT or step cap
    """
    pid = str(prompt_id)
    n = len(model_names)
    accept_action = n
    if max_steps is None:
        max_steps = 2 * n

    # Cost/latency estimates use config.
    def call_cost_latency(mname: str) -> Tuple[float, float]:
        cfg = MODEL_COSTS[mname]
        return float(cfg["cost_per_100"]) / 100.0, float(cfg["latency"])

    action_trace: List[int] = []
    total_cost = 0.0
    total_latency = 0.0
    num_calls = 0

    current_model: Optional[str] = None
    current_output = ""

    # --- initial base call (matches training reset) ---
    if base_model_name in model_names:
        current_model = base_model_name
        current_output = model_caller.call(model_name=base_model_name, prompt_id=pid, prompt=prompt)
        c, l = call_cost_latency(base_model_name)
        total_cost += c
        total_latency += l
        num_calls += 1

    # --- policy loop ---
    for _ in range(int(max_steps)):
        obs = embedder.embed(prompt, current_output)

        if mode == "irt":
            irt_vec = build_irt_feature_vector(
                model_names=model_names, prompt_id=pid, prompt=prompt, irt_provider=irt_provider
            )
            obs = np.concatenate([obs, irt_vec], axis=0).astype(np.float32)
        else:
            obs = obs.astype(np.float32)

        if mode == "hard_gate":
            allowed = build_allowed_mask(
                model_names=model_names, prompt_id=pid, prompt=prompt, irt_provider=irt_provider
            )
            # If the model would pick an invalid action, pick the best allowed instead.
            action = _policy_best_allowed_action(sb3_model, obs, allowed)
        else:
            action, _ = sb3_model.predict(obs, deterministic=deterministic)
            action = int(action)

        action_trace.append(int(action))

        if action == accept_action:
            return RouteResult(
                prompt_id=pid,
                chosen_model=current_model,
                accepted=True,
                num_calls=num_calls,
                est_cost_dollars=total_cost,
                est_latency_seconds=total_latency,
                action_trace=action_trace,
            )

        # Switch/call a model
        if action < 0 or action >= n:
            break

        next_model = str(model_names[int(action)])
        current_model = next_model
        current_output = model_caller.call(model_name=next_model, prompt_id=pid, prompt=prompt)
        c, l = call_cost_latency(next_model)
        total_cost += c
        total_latency += l
        num_calls += 1

    # If we hit the cap, return the last model as "chosen".
    return RouteResult(
        prompt_id=pid,
        chosen_model=current_model,
        accepted=False,
        num_calls=num_calls,
        est_cost_dollars=total_cost,
        est_latency_seconds=total_latency,
        action_trace=action_trace,
    )


# -----------------------------
# Batch prediction helpers
# -----------------------------


def load_prompt_list_csv(path: str) -> List[Tuple[str, str]]:
    """Load prompts from a CSV.

    Required columns: prompt_id, prompt
    Extra columns are ignored.
    """
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not {"prompt_id", "prompt"}.issubset(r.fieldnames or set()):
            raise ValueError(f"Input CSV must contain prompt_id and prompt. Got: {r.fieldnames}")
        for row in r:
            pid = str(row["prompt_id"])
            if pid in seen:
                continue
            seen.add(pid)
            out.append((pid, str(row["prompt"])))
    return out


def write_route_results_csv(path: str, results: Iterable[RouteResult], *, model_names: Sequence[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "prompt_id",
                "chosen_model",
                "accepted",
                "num_calls",
                "est_cost_dollars",
                "est_latency_seconds",
                "action_trace",
            ]
        )
        for rr in results:
            w.writerow(
                [
                    rr.prompt_id,
                    rr.chosen_model or "",
                    int(rr.accepted),
                    rr.num_calls,
                    f"{rr.est_cost_dollars:.6f}",
                    f"{rr.est_latency_seconds:.6f}",
                    json.dumps(rr.action_trace),
                ]
            )


def load_irt_provider_from_path(path: Optional[str]) -> Optional[IRTProvider]:
    if not path:
        return None
    irt_info = load_irt_csv(path)
    return CsvIRTProvider(irt_info)
