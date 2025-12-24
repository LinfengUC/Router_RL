from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .embedder import TextEmbedder


class EmbeddingWrapper(gym.Wrapper):
    """Convert text dict observations into a numeric embedding vector.

    If add_irt_features=True, appends IRT feature vector:
      [s_0, m_0, s_1, m_1, ...]
    """

    def __init__(self, env: gym.Env, embedder: TextEmbedder, add_irt_features: bool = False):
        super().__init__(env)
        self.embedder = embedder
        self.add_irt_features = bool(add_irt_features)

        base_dim = self.embedder.embedding_dim * 2
        irt_dim = 0
        if self.add_irt_features:
            irt_dim = 2 * self.env.unwrapped.num_models  # type: ignore[attr-defined]
        self.irt_dim = irt_dim

        obs_dim = base_dim + irt_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _augment(self, emb: np.ndarray) -> np.ndarray:
        if not self.add_irt_features:
            return emb.astype(np.float32)
        base_env = self.env.unwrapped
        if not hasattr(base_env, "get_irt_feature_vector"):
            return emb.astype(np.float32)
        irt_vec = base_env.get_irt_feature_vector()
        return np.concatenate([emb, irt_vec], axis=0).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        emb = self.embedder.embed(obs["question"], obs["current_output"])
        emb = self._augment(emb)
        return emb, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        emb = self.embedder.embed(obs["question"], obs["current_output"])
        emb = self._augment(emb)
        return emb, reward, terminated, truncated, info
