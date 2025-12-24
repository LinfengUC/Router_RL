from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    """Mean-pooled transformer embedder with caching.

    Default model: sentence-transformers/all-mpnet-base-v2

    Notes:
      - Uses GPU automatically when available unless device is overridden.
      - Caches embeddings per text string to avoid recomputing.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        max_length: int = 512,
        device: str = "auto",  # "auto" | "cpu" | "cuda"
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.max_length = int(max_length)
        self.embedding_dim = int(self.model.config.hidden_size)

        self._cache: Dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _encode_one(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if text in self._cache:
            return self._cache[text]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)
        token_embeddings = out.last_hidden_state  # [1, seq, hid]
        attention_mask = inputs["attention_mask"]  # [1, seq]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts

        emb = pooled.squeeze(0).cpu().numpy().astype(np.float32)
        self._cache[text] = emb
        return emb

    def embed(self, question: str, current_output: str) -> np.ndarray:
        """Return concatenated [q_embed, output_embed]."""
        q = self._encode_one(question)
        o = self._encode_one(current_output) if current_output else np.zeros_like(q)
        return np.concatenate([q, o]).astype(np.float32)
