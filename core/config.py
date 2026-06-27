"""
core/config.py

Central config for switching between providers without touching any
other code. Set these via environment variables, or just edit the
defaults below for local testing.

LLM_PROVIDER:        "openai" or "ollama"
EMBEDDING_PROVIDER:  "openai" or "local"   (local = sentence-transformers)
"""

import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")          # "openai" | "ollama"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # "openai" | "local"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CODE_EMBEDDINGS_PATH = os.getenv("CODE_EMBEDDINGS_PATH", "data/code_embeddings.npy")
NAICS_CSV_PATH = os.getenv("NAICS_CSV_PATH", "data/naics_codes.csv")
