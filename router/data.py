from __future__ import annotations

import csv
from typing import Dict, Tuple, Set, Any, Optional


LabelDataset = Dict[str, Dict[str, Any]]
IRTInfo = Dict[str, Dict[str, Dict[str, int]]]


def load_labels_csv(path: str) -> Tuple[LabelDataset, Set[str]]:
    """Load normalized labels.

    Expected columns:
      - prompt_id (str/int)
      - prompt (str)
      - model_name (str)
      - label (0/1)

    Returns:
      dataset: {prompt_id: {"prompt": str, "labels": {model_name: 0/1}}}
      model_names: set[str]
    """
    dataset: LabelDataset = {}
    model_names: Set[str] = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"prompt_id", "prompt", "model_name", "label"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Labels file missing columns. Need {required}, got {reader.fieldnames}")

        for row in reader:
            pid = str(row["prompt_id"])
            prompt = row["prompt"]
            model = row["model_name"]
            raw = str(row["label"]).strip()
            try:
                label = int(round(float(raw)))
            except ValueError as e:
                raise ValueError(f"Bad label value {raw!r} in {path}") from e
            if label not in (0, 1):
                raise ValueError(f"Label must be 0/1; got {label} for prompt_id={pid}, model={model}")

            model_names.add(model)
            if pid not in dataset:
                dataset[pid] = {"prompt": prompt, "labels": {}}
            dataset[pid]["labels"][model] = label

    return dataset, model_names


def load_irt_csv(path: str) -> IRTInfo:
    """Load normalized IRT predictions.

    Expected columns (flexible):
      - prompt_id
      - model_name
      - s_bert_pred (0/1)   [optional]
      - m_bert_pred (0/1)   [optional]

    Real-world note:
      Sometimes you may only have *one* IRT model's feedback (e.g. only s-bert
      or only m-bert). In that case this loader will **copy the existing
      prediction** into the missing column so downstream code continues to work.
      If both are missing for a row, we conservatively set (1,1) = "allowed".

    Returns:
      irt_info: {prompt_id: {model_name: {"s_bert_pred":0/1, "m_bert_pred":0/1}}}
    """
    irt_info: IRTInfo = {}

    def _parse_pred(row: dict, key: str) -> Optional[int]:
        if key not in row:
            return None
        raw = row.get(key)
        if raw is None:
            return None
        s = str(raw).strip()
        if s == "" or s.lower() == "nan":
            return None
        try:
            v = int(round(float(s)))
        except ValueError:
            return None
        return 1 if v == 1 else 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        base_required = {"prompt_id", "model_name"}
        if not base_required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"IRT file missing columns. Need {base_required}, got {reader.fieldnames}"
            )

        has_s = "s_bert_pred" in (reader.fieldnames or [])
        has_m = "m_bert_pred" in (reader.fieldnames or [])
        if not (has_s or has_m):
            raise ValueError(
                "IRT file must contain at least one of {'s_bert_pred','m_bert_pred'}. "
                f"Got columns: {reader.fieldnames}"
            )

        for row in reader:
            pid = str(row["prompt_id"])
            model = row["model_name"]

            s_pred = _parse_pred(row, "s_bert_pred")
            m_pred = _parse_pred(row, "m_bert_pred")

            # If only one IRT model provided feedback, mirror it into the missing slot.
            if s_pred is None and m_pred is not None:
                s_pred = m_pred
            if m_pred is None and s_pred is not None:
                m_pred = s_pred

            # If both missing/unparseable, default to "allowed" to avoid hard failures.
            if s_pred is None and m_pred is None:
                s_pred, m_pred = 1, 1

            if pid not in irt_info:
                irt_info[pid] = {}
            irt_info[pid][model] = {"s_bert_pred": s_pred, "m_bert_pred": m_pred}

    return irt_info


def intersect_models(*model_sets: Set[str]) -> Set[str]:
    """Intersection helper with a decent error message."""
    if not model_sets:
        return set()
    common = set(model_sets[0])
    for s in model_sets[1:]:
        common &= set(s)
    return common
