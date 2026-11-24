"""Embedding client. DEFAULT: local fastembed (ONNX, CPU, no API key).

EMBEDDING_PROVIDER=local   (default) -> fastembed, model from EMBEDDING_MODEL
EMBEDDING_PROVIDER=openai  -> legacy text-embedding-3-small, 1536-dim

Public API unchanged: EMBEDDING_MODEL, EMBEDDING_DIM, DISTANCE,
vector_params(), embed_documents(), embed_query().

Query and document vectors MUST come from the same model. Changing provider
or model changes the dimension -> every collection must be recreated and
re-embedded.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client.http import models as qmodels

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
DISTANCE = qmodels.Distance.COSINE

_KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

if PROVIDER == "openai":
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    _dim_env = os.getenv("EMBEDDING_DIM")
    if _dim_env:
        EMBEDDING_DIM = int(_dim_env)
    elif EMBEDDING_MODEL in _KNOWN_DIMS:
        EMBEDDING_DIM = _KNOWN_DIMS[EMBEDDING_MODEL]
    else:  # unknown model: probe once (loads the model at import time)
        from fastembed import TextEmbedding as _TE
        EMBEDDING_DIM = len(next(iter(_TE(model_name=EMBEDDING_MODEL).embed(["x"]))))


def vector_params() -> qmodels.VectorParams:
    """Qdrant vectors_config for a fresh collection at this model's dimension."""
    return qmodels.VectorParams(size=EMBEDDING_DIM, distance=DISTANCE)


# ---------------------------------------------------------------- local ----
_model = None
_lock = threading.Lock()


def _local():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            # First use downloads the ONNX model from Hugging Face.
            _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def _local_docs(texts: list[str]) -> list[list[float]]:
    # passage_embed / query_embed apply the model's own prefixes (e5, nomic)
    return [v.tolist() for v in _local().passage_embed(texts)]


def _local_query(text: str) -> list[float]:
    return next(iter(_local().query_embed(text))).tolist()


# --------------------------------------------------------------- openai ----
def _make_openai_backend():
    """Legacy path. Imports are lazy so the local default needs neither
    openai, tenacity, langchain-openai nor a key."""
    import openai
    import tenacity
    from langchain_openai import OpenAIEmbeddings

    retryable = (openai.RateLimitError, openai.APIConnectionError,
                 openai.APITimeoutError, openai.InternalServerError)
    holder: dict = {}

    def client():
        if "c" not in holder:
            if not os.getenv("OPENAI_API_KEY"):
                raise SystemExit("[embeddings] EMBEDDING_PROVIDER=openai but "
                                 "OPENAI_API_KEY is not set.")
            holder["c"] = OpenAIEmbeddings(model=EMBEDDING_MODEL,
                                           chunk_size=300, max_retries=5)
        return holder["c"]

    deco = tenacity.retry(
        retry=tenacity.retry_if_exception_type(retryable),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
        stop=tenacity.stop_after_attempt(6), reraise=True)

    @deco
    def docs(texts):
        return client().embed_documents(texts)

    @deco
    def query(text):
        return client().embed_query(text)

    return docs, query


if PROVIDER == "openai":
    _docs_impl, _query_impl = _make_openai_backend()
else:
    _docs_impl, _query_impl = _local_docs, _local_query


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _docs_impl(texts)


def embed_query(text: str) -> list[float]:
    return _query_impl(text)