# OpenBio-Intel Windows Quickstart PowerShell Script
# Boots the full stack (Postgres, Qdrant, Neo4j, FastAPI backend, Worker, Next.js frontend)
# via Docker Compose, ensures all Qdrant collections are initialized, and seeds sample data.

$ErrorActionPreference = "Stop"

function Write-Log { param($msg) Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn { param($msg) Write-Host "==> $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "==> $msg" -ForegroundColor Red }

# 1. Check Docker & Docker Compose
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Err "Docker is not installed or not in PATH. Please install Docker Desktop."
    exit 1
}

$composeVer = docker compose version 2>$null
if (-not $composeVer) {
    Write-Err "Docker Compose v2 is required ('docker compose', not 'docker-compose')."
    exit 1
}

docker info >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "Docker daemon is not running. Please start Docker Desktop."
    exit 1
}
Write-Log "Docker and Docker Compose found."

# 2. Check .env
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Warn "Created .env from .env.example."
    Write-Warn "Open .env and set your API keys (or LLM_PROVIDER=custom), then rerun .\quickstart.ps1."
    exit 0
}

# Provider check
$envContent = Get-Content .env
$provider = ($envContent | Where-Object { $_ -match "^LLM_PROVIDER=" } | ForEach-Object { $_.Split("=")[1].Trim() })
if (-not $provider) { $provider = "anthropic" }

Write-Log "Detected LLM Provider: $provider"

# 3. Build & start services
Write-Log "Building and starting services via Docker Compose..."
docker compose up -d --build postgres qdrant neo4j backend worker frontend

# 4. Wait for backend health
Write-Log "Waiting for backend to report healthy..."
$attempts = 0
$healthy = $false
while ($attempts -lt 60) {
    Start-Sleep -Seconds 3
    $attempts++
    $containerId = (docker compose ps -q backend).Trim()
    if ($containerId) {
        $status = (docker inspect -f '{{.State.Health.Status}}' $containerId 2>$null).Trim()
        if ($status -eq "healthy") {
            $healthy = $true
            break
        }
    }
}

if (-not $healthy) {
    Write-Err "Backend failed to report healthy within 3 minutes. Check: docker compose logs backend"
    exit 1
}
Write-Log "Backend is healthy."

# 5. Initialize vector collections in Qdrant
Write-Log "Initializing all vector collections in Qdrant..."
docker compose exec backend python -c "import os; from qdrant_client import QdrantClient; from embeddings import vector_params; c = QdrantClient(host=os.getenv('QDRANT_HOST', 'qdrant'), port=int(os.getenv('QDRANT_PORT', '6333'))); vp = vector_params(); [c.create_collection(name, vectors_config=vp) for name in ['clinical_trials', 'openfda_drugsfda', 'clinical_trials_pdf_extracts', 'pubmed_literature', 'sec_filings', 'corporate_news', 'fda_crls'] if not c.collection_exists(name)]; print('All Qdrant collections initialized successfully.')"

# 6. Seed sample data
Write-Log "Seeding sample corpus (50 trials + 50 FDA records)..."
docker compose --profile seed run --rm seed python seed_bulk_data.py --source all --endpoint drugsfda --limit 50 --skip-s3 --skip-neo4j

Write-Log "Seeding sample active Phase 2/3 trials for Catalyst Tracker..."
docker compose --profile seed run --rm seed python fetch_and_embed_trials.py --condition "Rheumatoid Arthritis" --limit 20 --skip-s3 --skip-neo4j

Write-Host ""
Write-Host "Ready! Open http://localhost:3000 in your browser." -ForegroundColor Green
Write-Host "API docs: http://127.0.0.1:8080/docs"
Write-Host "Logs:     docker compose logs -f backend worker"
Write-Host "Stop:     docker compose down"
