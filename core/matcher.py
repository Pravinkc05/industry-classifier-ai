"""
core/matcher.py

Ranks a company's embedded business profile against the cached industry
code embeddings using cosine similarity, returning the top-k candidate
codes with their similarity scores.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CodeMatch:
    code: str
    description: str
    score: float


def top_k_matches(
    profile_embedding: np.ndarray,
    code_embeddings: np.ndarray,
    codes: list[dict],
    k: int = 5,
) -> list[CodeMatch]:
    """
    codes: list of dicts like {"code": "541511", "description": "Custom
    Computer Programming Services"}, aligned by index with code_embeddings.
    """
    similarities = cosine_similarity(profile_embedding.reshape(1, -1), code_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]

    return [
        CodeMatch(
            code=codes[i]["code"],
            description=codes[i]["description"],
            score=float(similarities[i]),
        )
        for i in top_indices
    ]
