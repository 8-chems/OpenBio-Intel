"""Idempotent initialization of all Qdrant collections used across OpenBio-Intel.

Ensures that auxiliary tools (Landscape matrix, Catalyst tracker, PubMed,
SEC filings, Corporate news, CRLs, and PDF literature) never fail with 404
errors when queried before bulk ETL jobs have run.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load env from .env if present
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from embeddings import vector_params

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

ALL_COLLECTIONS = [
    os.getenv("TRIALS_COLLECTION", "clinical_trials"),
    "openfda_drugsfda",
    "clinical_trials_pdf_extracts",
    "pubmed_literature",
    "sec_filings",
    "corporate_news",
    "fda_crls",
]


def ensure_all_collections() -> None:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10.0)
    v_params = vector_params()

    print(f"[init_collections] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    created_count = 0
    existing_count = 0

    for col in ALL_COLLECTIONS:
        if not client.collection_exists(col):
            print(f"[init_collections] Creating collection: '{col}' (size={v_params.size})...")
            client.create_collection(
                collection_name=col,
                vectors_config=v_params,
            )
            created_count += 1
        else:
            existing_count += 1

    print(
        f"[init_collections] Done: {existing_count} existing, "
        f"{created_count} created. All {len(ALL_COLLECTIONS)} collections ready."
    )


if __name__ == "__main__":
    ensure_all_collections()
