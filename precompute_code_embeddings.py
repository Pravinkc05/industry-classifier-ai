"""
precompute_code_embeddings.py

One-time script: embeds every description in the industry code reference
set (data/naics_codes.csv) and caches the result to disk, so the app
doesn't need to re-embed the whole reference table on every run.

Run this once after setup, and again any time you edit naics_codes.csv.
"""

import csv

import numpy as np

from core import config
from core.embeddings import embed_texts


def load_codes(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{"code": row["code"], "description": row["description"]} for row in reader]


def main():
    codes = load_codes(config.NAICS_CSV_PATH)
    descriptions = [c["description"] for c in codes]

    print(f"Embedding {len(descriptions)} industry code descriptions "
          f"using provider='{config.EMBEDDING_PROVIDER}'...")
    embeddings = embed_texts(descriptions)

    np.save(config.CODE_EMBEDDINGS_PATH, {"embeddings": embeddings, "codes": codes})
    print(f"Saved embeddings to {config.CODE_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
