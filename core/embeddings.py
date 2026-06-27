"""
core/embeddings.py

Generates vector embeddings for text, with two interchangeable providers
switched via core.config.EMBEDDING_PROVIDER:
  - "openai": OpenAI's embedding API
  - "local":  sentence-transformers, running entirely on-device

Also handles loading/caching the precomputed embeddings for the standard
industry code reference set, so they're only computed once.
"""

from __future__ import annotations
import os

import numpy as np

from core import config

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _local_model


def _embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.embeddings.create(model=config.OPENAI_EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data])


def _embed_local(texts: list[str]) -> np.ndarray:
    model = _get_local_model()
    return np.array(model.encode(texts, show_progress_bar=False))


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings, returning an (n_texts, embedding_dim) array."""
    if config.EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    elif config.EMBEDDING_PROVIDER == "local":
        return _embed_local(texts)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{config.EMBEDDING_PROVIDER}'. Use 'openai' or 'local'."
        )


def embed_text(text: str) -> np.ndarray:
    """Convenience wrapper for embedding a single string."""
    return embed_texts([text])[0]


def load_cached_code_embeddings() -> tuple[np.ndarray, list[str]]:
    """
    Loads the precomputed embeddings for the industry code reference set,
    along with their corresponding code labels. Run
    precompute_code_embeddings.py first to generate this cache.
    """
    if not os.path.exists(config.CODE_EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"No cached embeddings found at {config.CODE_EMBEDDINGS_PATH}. "
            "Run `python precompute_code_embeddings.py` first."
        )

    data = np.load(config.CODE_EMBEDDINGS_PATH, allow_pickle=True).item()
    return data["embeddings"], data["codes"]
