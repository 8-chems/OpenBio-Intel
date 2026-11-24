#!/usr/bin/env bash
# 5-minute quickstart: boots the full stack (Qdrant, Neo4j, Postgres, FastAPI backend,
# Worker, Next.js frontend) via Docker Compose, initializes all Qdrant collections,
# and seeds a sample corpus so http://localhost:3000 is immediately interactive.
#
# Works with Anthropic, OpenAI, Kimi, Nvidia, Groq, or Custom local/enterprise
# LLM Gateways (e.g. Unistra, vLLM, Ollama).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { printf "${BOLD}==>${RESET} %s\n" "$1"; }
warn() { printf "${YELLOW}==>${RESET} %s\n" "$1"; }
fail() { printf "${RED}==>${RESET} %s\n" "$1" >&2; exit 1; }

# --- 1. Check for Docker and Docker Compose ----------------------------------
command -v docker >/dev/null 2>&1 || fail "Docker is not installed. Get it at https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is not available (docker compose, not docker-compose). It ships with modern Docker Desktop / Docker Engine."
docker info >/dev/null 2>&1 || fail "Docker daemon isn't running -- start Docker Desktop (or dockerd) and try again."
log "Docker and Docker Compose found."

# --- 2. Copy .env.example to .env if missing ---------------------------------
if [ ! -f .env ]; then
    cp .env.example .env
    warn "Created .env from .env.example."
    warn "Open .env now and configure your API keys (OPENAI_API_KEY + ANTHROPIC_API_KEY, or LLM_PROVIDER=custom), then re-run ./quickstart.sh."
    exit 0
fi

# Detect provider and validate required configuration
provider="$(grep -E '^LLM_PROVIDER=' .env 2>/dev/null | cut -d'=' -f2 | tr -d ' "\r' || true)"
provider="${provider:-anthropic}"

missing=()
if [ "$provider" = "custom" ]; then
    grep -qE '^CUSTOM_LLM_BASE_URL=.+' .env || missing+=("CUSTOM_LLM_BASE_URL")
    grep -qE '^CUSTOM_LLM_MODEL=.+' .env || missing+=("CUSTOM_LLM_MODEL")
    emb_prov="$(grep -E '^EMBEDDING_PROVIDER=' .env 2>/dev/null | cut -d'=' -f2 | tr -d ' "\r' || true)"
    if [ "$emb_prov" != "local" ]; then
        grep -qE '^OPENAI_API_KEY=.+' .env || missing+=("OPENAI_API_KEY (or set EMBEDDING_PROVIDER=local)")
    fi
elif [ "$provider" = "kimi" ]; then
    grep -qE '^KIMI_API_KEY=.+' .env || missing+=("KIMI_API_KEY")
    grep -qE '^OPENAI_API_KEY=.+' .env || missing+=("OPENAI_API_KEY")
elif [ "$provider" = "nvidia" ]; then
    grep -qE '^NVIDIA_API_KEY=.+' .env || missing+=("NVIDIA_API_KEY")
    grep -qE '^OPENAI_API_KEY=.+' .env || missing+=("OPENAI_API_KEY")
elif [ "$provider" = "groq" ]; then
    grep -qE '^GROQ_API_KEY=.+' .env || missing+=("GROQ_API_KEY")
    grep -qE '^OPENAI_API_KEY=.+' .env || missing+=("OPENAI_API_KEY")
else
    # default: anthropic
    grep -qE '^ANTHROPIC_API_KEY=.+' .env || missing+=("ANTHROPIC_API_KEY")
    grep -qE '^OPENAI_API_KEY=.+' .env || missing+=("OPENAI_API_KEY")
fi

if [ "${#missing[@]}" -gt 0 ]; then
    fail "Missing required value(s) in .env for provider '$provider': ${missing[*]}. Edit .env and re-run."
fi
log ".env found with valid keys set for provider '$provider'."

# --- 3. Boot full stack (Postgres, Qdrant, Neo4j, backend, worker, frontend) ---
log "Building and starting the stack (qdrant, neo4j, postgres, backend, worker, frontend)..."
docker compose up -d --build postgres qdrant neo4j backend worker frontend

log "Waiting for the backend to report healthy..."
attempts=0
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q backend)" 2>/dev/null)" = "healthy" ]; do
    attempts=$((attempts + 1))
    if [ "$attempts" -gt 60 ]; then
        fail "Backend did not become healthy in time. Check logs: docker compose logs backend"
    fi
    sleep 5
done
log "Backend is healthy."

# --- 4. Defensive collection initialization in Qdrant --------------------------
log "Initializing all vector collections in Qdrant..."
docker compose exec backend python -c "import os; from qdrant_client import QdrantClient; from embeddings import vector_params; c = QdrantClient(host=os.getenv('QDRANT_HOST', 'qdrant'), port=int(os.getenv('QDRANT_PORT', '6333'))); vp = vector_params(); [c.create_collection(name, vectors_config=vp) for name in ['clinical_trials', 'openfda_drugsfda', 'clinical_trials_pdf_extracts', 'pubmed_literature', 'sec_filings', 'corporate_news', 'fda_crls'] if not c.collection_exists(name)]; print('All Qdrant collections initialized successfully.')" || warn "Collection initialization check completed with warning."

# --- 5. Sample seed -----------------------------------------------------------
log "Seeding a sample corpus (50 clinical trials + 50 FDA approval records)..."
log "This uses --skip-s3 and --skip-neo4j for fast local startup."
docker compose --profile seed run --rm seed \
    python seed_bulk_data.py --source all --endpoint drugsfda --limit 50 --skip-s3 --skip-neo4j

# Seed upcoming Phase 2/3 trials so Catalyst Tracker works out-of-the-box
log "Seeding sample active Phase 2/3 trials for Catalyst Tracker..."
docker compose --profile seed run --rm seed \
    python fetch_and_embed_trials.py --condition "Rheumatoid Arthritis" --limit 20 --skip-s3 --skip-neo4j || true

echo
printf "${GREEN}${BOLD}Ready.${RESET} Open ${BOLD}http://localhost:3000${RESET} and try one of the example queries.\n"
echo "API docs: http://127.0.0.1:8080/docs (or http://127.0.0.1:8000/docs)"
echo "Logs:     docker compose logs -f backend worker"
echo "Stop:     docker compose down   (add -v to also delete the seeded data)"
