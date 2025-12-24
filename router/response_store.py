from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional, Tuple, Set


@dataclass(frozen=True)
class ResponseRecord:
    prompt_id: str
    prompt: str
    output: str
    reasoning: str = ""


    def full_text_for_cost(self) -> str:
        # Cost uses the full output + reasoning (when available).
        parts = [self.output or ""]
        if self.reasoning:
            parts.append(self.reasoning)
        return "\n".join(parts).strip()


ResponseMap = Dict[str, Dict[str, ResponseRecord]]  # model_name -> prompt_id -> record


def available_models(responses_dir: str, split: str) -> Set[str]:
    split_dir = os.path.join(responses_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing responses split directory: {split_dir}")
    models = set()
    for fname in os.listdir(split_dir):
        if fname.endswith(".csv"):
            models.add(fname[:-4])
    return models


def load_responses(responses_dir: str, split: str, model_names: Optional[Iterable[str]] = None) -> ResponseMap:
    """Load normalized response files into memory.

    Expected per-file columns:
      - prompt_id
      - prompt
      - output
      - reasoning

    Layout:
      responses_dir/<split>/<model_name>.csv
    """
    split_dir = os.path.join(responses_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing responses split directory: {split_dir}")

    if model_names is None:
        model_names = sorted(available_models(responses_dir, split))

    responses: ResponseMap = {}

    for model in model_names:
        path = os.path.join(split_dir, f"{model}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing response file for model={model}: {path}")

        model_map: Dict[str, ResponseRecord] = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"prompt_id", "prompt", "output", "reasoning"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"Response file {path} missing columns. Need {required}, got {reader.fieldnames}")

            for row in reader:
                pid = str(row["prompt_id"])
                rec = ResponseRecord(
                    prompt_id=pid,
                    prompt=row["prompt"],
                    output=row.get("output", "") or "",
                    reasoning=row.get("reasoning", "") or "",
                )
                model_map[pid] = rec

        responses[model] = model_map

    return responses
