"""Smoke-test an OpenAI-compatible endpoint (e.g. the Unistra gateway) for the
three things research_agent.py needs: plain chat, tool calling, structured
output. Also probes whether the gateway exposes an embeddings endpoint.

    python test_llm_provider.py

Reads CUSTOM_LLM_BASE_URL / CUSTOM_LLM_API_KEY / CUSTOM_LLM_MODEL from .env.
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

BASE = os.getenv("CUSTOM_LLM_BASE_URL", "https://conversation.ia.unistra.fr/api/")
KEY = os.getenv("CUSTOM_LLM_API_KEY")
MODEL = os.getenv("CUSTOM_LLM_MODEL", "ministral")
if not KEY:
    sys.exit("CUSTOM_LLM_API_KEY is not set in .env")

llm = ChatOpenAI(model=MODEL, base_url=BASE, api_key=KEY, temperature=0,
                 max_tokens=512, timeout=60, max_retries=0)

class Verdict(BaseModel):
    is_in_domain: bool = Field(description="True if about clinical trials / pharma.")
    reason: str = Field(description="One short sentence.")

@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)

def step(name, fn):
    try:
        out = fn()
        print(f"[ OK ] {name}: {out}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:300]}")
        return False

print(f"endpoint={BASE}  model={MODEL}\n")

step("plain chat", lambda: llm.invoke([HumanMessage(content="Say 'ok'.")]).content[:60])

step("tool calling", lambda: llm.bind_tools([add]).invoke(
    [HumanMessage(content="Use the tool to add 2 and 3.")]).tool_calls)

for method in ("function_calling", "json_schema", "json_mode"):
    step(f"structured output [{method}]", lambda m=method: llm.with_structured_output(
        Verdict, method=m).invoke([HumanMessage(content=(
            "Is this about clinical trials? 'Phase 3 trials of pembrolizumab'. "
            "Reply as JSON with is_in_domain and reason."))]))

def list_models():
    r = requests.get(BASE.rstrip("/") + "/models",
                     headers={"Authorization": f"Bearer {KEY}"}, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", r.json())
    return [m.get("id") for m in data][:30]

step("list models (look for an embedding model)", list_models)

def try_embeddings():
    emb_model = os.getenv("CUSTOM_EMBED_MODEL", "")
    if not emb_model:
        return "skipped (set CUSTOM_EMBED_MODEL to an embedding model id from the list above)"
    r = requests.post(BASE.rstrip("/") + "/embeddings",
                      headers={"Authorization": f"Bearer {KEY}"},
                      json={"model": emb_model, "input": ["hello"]}, timeout=30)
    r.raise_for_status()
    return f"dim={len(r.json()['data'][0]['embedding'])}"

step("embeddings endpoint", try_embeddings) 