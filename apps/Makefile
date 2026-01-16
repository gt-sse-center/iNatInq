# =============================================================================
# iNatInq Apps Makefile
# =============================================================================
# Convenient targets for development, testing, and Docker operations.

.PHONY: help
help:
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║                         iNatInq Apps Makefile                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "┌── Quick Start ──────────────────────────────────────────────────────────────┐"
	@echo "│ make up                Start all services (alias for docker-up)             │"
	@echo "│ make down              Stop all services (alias for docker-down)            │"
	@echo "│ make status            Show service status and health                       │"
	@echo "│ make lazydocker        Launch LazyDocker UI                                 │"
	@echo "├── Development ──────────────────────────────────────────────────────────────┤"
	@echo "│ make test              Run all unit tests                                   │"
	@echo "│ make test-cov          Run tests with coverage report                       │"
	@echo "│ make lint              Run linters (ruff)                                   │"
	@echo "│ make format            Format code (ruff)                                   │"
	@echo "│ make typecheck         Run type checker (mypy)                              │"
	@echo "│ make dev               Start development server (local, no Docker)          │"
	@echo "├── Docker Compose ───────────────────────────────────────────────────────────┤"
	@echo "│ make docker-up         Start all services                                   │"
	@echo "│ make docker-down       Stop all services                                    │"
	@echo "│ make docker-restart    Restart all services                                 │"
	@echo "│ make docker-logs       Tail logs for all services                           │"
	@echo "│ make docker-ps         Show running containers                              │"
	@echo "│ make docker-build-base Build base image (heavy deps, run once)              │"
	@echo "│ make docker-build      Build pipeline image (fast after base)               │"
	@echo "│ make docker-rebuild    Rebuild and restart pipeline                         │"
	@echo "│ make docker-clean      Stop services and remove volumes                     │"
	@echo "│ make docker-health     Check health of all services                         │"
	@echo "├── Docker Service Logs ──────────────────────────────────────────────────────┤"
	@echo "│ make logs-pipeline     Tail pipeline logs                                   │"
	@echo "│ make logs-ollama       Tail ollama logs                                     │"
	@echo "│ make logs-qdrant       Tail qdrant logs                                     │"
	@echo "│ make logs-ray          Tail ray-head logs                                   │"
	@echo "│ make logs-minio        Tail minio logs                                      │"
	@echo "├── Docker Shell Access ──────────────────────────────────────────────────────┤"
	@echo "│ make shell-pipeline    Shell into pipeline container                        │"
	@echo "│ make shell-ollama      Shell into ollama container                          │"
	@echo "├── Ollama Models ────────────────────────────────────────────────────────────┤"
	@echo "│ make ollama-pull       Pull nomic-embed-text model                          │"
	@echo "│ make ollama-list       List installed Ollama models                         │"
	@echo "├── Web UIs (opens in browser) ───────────────────────────────────────────────┤"
	@echo "│ make ui-all            Open all web UIs                                     │"
	@echo "│ make ui-pipeline       Open Pipeline API docs (Swagger)                     │"
	@echo "│ make ui-minio          Open MinIO Console                                   │"
	@echo "│ make ui-qdrant         Open Qdrant Dashboard                                │"
	@echo "│ make ui-ray            Open Ray Dashboard                                   │"
	@echo "│ make ui-weaviate       Open Weaviate GraphQL Console                        │"
	@echo "├── Docker Scaling ───────────────────────────────────────────────────────────┤"
	@echo "│ make ray-scale N=3     Scale Ray workers to N replicas                      │"
	@echo "├── Synthetic Data ───────────────────────────────────────────────────────────┤"
	@echo "│ make syntheticdata-generate  Generate test documents from Moby Dick         │"
	@echo "│ make syntheticdata-upload    Upload documents to MinIO                       │"
	@echo "│ make syntheticdata-setup     Generate and upload in one step                 │"
	@echo "│ make syntheticdata-clean     Remove generated documents                      │"
	@echo "└─────────────────────────────────────────────────────────────────────────────┘"

# =============================================================================
# Variables
# =============================================================================
COMPOSE_FILE := zarf/compose/dev/docker-compose.yaml
DOCKER_COMPOSE := docker compose -f $(COMPOSE_FILE)
SMOKE_ENV_FILE ?= zarf/compose/dev/.env.local

# =============================================================================
# Quick Start Aliases
# =============================================================================

.PHONY: up
up: docker-up

.PHONY: down
down: docker-down

.PHONY: status
status: docker-health

.PHONY: lazydocker
lazydocker:
	@command -v lazydocker >/dev/null 2>&1 || { echo "❌ lazydocker not installed. Run: brew install lazydocker"; exit 1; }
	@lazydocker

# =============================================================================
# Development
# =============================================================================

.PHONY: test
test:
	@echo "Running unit tests..."
	uv run pytest tests/unit/ -v

.PHONY: test-cov
test-cov:
	@echo "Running tests with coverage..."
	uv run pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

.PHONY: lint
lint:
	@echo "Running linters..."
	uv run ruff check src/ tests/

.PHONY: format
format:
	@echo "Formatting code..."
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

.PHONY: typecheck
typecheck:
	@echo "Running type checker..."
	uv run mypy src/

.PHONY: dev
dev:
	@echo "Starting development server..."
	uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# =============================================================================
# Docker Compose Operations
# =============================================================================

.PHONY: docker-up
docker-up:
	@echo "Starting all services..."
	$(DOCKER_COMPOSE) up -d
	@echo ""
	@echo "✅ Services started!"
	@echo ""
	@echo "📝 Service endpoints:"
	@echo "   Pipeline API:    http://localhost:8000"
	@echo "   Pipeline Docs:   http://localhost:8000/docs"
	@echo "   MinIO Console:   http://localhost:9001"
	@echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "   Ray Dashboard:   http://localhost:8265"
	@echo "   Ollama:          http://localhost:11434"

.PHONY: docker-down
docker-down:
	@echo "Stopping all services..."
	$(DOCKER_COMPOSE) down

.PHONY: docker-logs
docker-logs:
	$(DOCKER_COMPOSE) logs -f

.PHONY: docker-ps
docker-ps:
	$(DOCKER_COMPOSE) ps

.PHONY: docker-build-base
docker-build-base:
	@echo "Building pipeline base image (heavy dependencies)..."
	@echo "This may take 5-10 minutes on first build..."
	docker build -f zarf/docker/base/Dockerfile.pipeline-base -t inatinq/pipeline-base:0.1.0 .
	@echo "✅ Base image built: inatinq/pipeline-base:0.1.0"

.PHONY: docker-build
docker-build:
	@echo "Building pipeline image..."
	@docker image inspect inatinq/pipeline-base:0.1.0 >/dev/null 2>&1 || { \
		echo "⚠️  Base image not found. Building it first..."; \
		$(MAKE) docker-build-base; \
	}
	$(DOCKER_COMPOSE) build pipeline

.PHONY: docker-rebuild
docker-rebuild:
	@echo "Rebuilding and restarting pipeline..."
	@docker image inspect inatinq/pipeline-base:0.1.0 >/dev/null 2>&1 || { \
		echo "⚠️  Base image not found. Building it first..."; \
		$(MAKE) docker-build-base; \
	}
	$(DOCKER_COMPOSE) up -d --build pipeline

.PHONY: docker-clean
docker-clean:
	@echo "⚠️  Stopping services and removing volumes..."
	$(DOCKER_COMPOSE) down -v
	@echo "✅ Cleanup complete"

.PHONY: docker-restart
docker-restart:
	@echo "Restarting all services..."
	$(DOCKER_COMPOSE) restart
	@echo "✅ Services restarted"

.PHONY: docker-health
docker-health:
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║                         Service Health Status                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(DOCKER_COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@ENV_FILE=$(SMOKE_ENV_FILE) bash zarf/scripts/providers_health.sh
	@echo "┌── Local Compose Health Checks ───────────────────────────────────────────────┐"
	@printf "│ Pipeline API:    "; curl -sf http://localhost:8000/healthz >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@printf "│ Qdrant:          "; curl -sf http://localhost:6333/readyz >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@printf "│ Weaviate:        "; curl -sf http://localhost:8080/v1/.well-known/ready >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@printf "│ Ollama:          "; curl -sf http://localhost:11434/api/tags >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@printf "│ MinIO:           "; curl -sf http://localhost:9000/minio/health/live >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@printf "│ Ray Dashboard:   "; curl -sf http://localhost:8265 >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@echo "└──────────────────────────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "📝 Local Compose Endpoints:"
	@echo "   Pipeline API:     http://localhost:8000"
	@echo "   Pipeline Docs:    http://localhost:8000/docs"
	@echo "   MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)"
	@echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "   Ray Dashboard:    http://localhost:8265"
	@echo "   Weaviate:         http://localhost:8080"
	@echo "   Ollama:           http://localhost:11434"

.PHONY: smoke-providers
smoke-providers:
	@ENV_FILE=$(SMOKE_ENV_FILE) bash zarf/scripts/smoke_providers.sh

.PHONY: smoke-test
smoke-test: providers-health smoke-providers

.PHONY: providers-health
providers-health:
	@ENV_FILE=$(SMOKE_ENV_FILE) bash zarf/scripts/providers_health.sh

# =============================================================================
# Docker Service Logs
# =============================================================================

.PHONY: logs-pipeline
logs-pipeline:
	$(DOCKER_COMPOSE) logs -f pipeline

.PHONY: logs-ollama
logs-ollama:
	$(DOCKER_COMPOSE) logs -f ollama

.PHONY: logs-qdrant
logs-qdrant:
	$(DOCKER_COMPOSE) logs -f qdrant

.PHONY: logs-ray
logs-ray:
	$(DOCKER_COMPOSE) logs -f ray-head

.PHONY: logs-minio
logs-minio:
	$(DOCKER_COMPOSE) logs -f minio

# =============================================================================
# Docker Scaling
# =============================================================================

N ?= 1
.PHONY: ray-scale
ray-scale:
	@echo "Scaling Ray workers to $(N) replicas..."
	$(DOCKER_COMPOSE) up -d --scale ray-worker=$(N)

# =============================================================================
# Docker Shell Access
# =============================================================================

.PHONY: shell-pipeline
shell-pipeline:
	@echo "Opening shell in pipeline container..."
	$(DOCKER_COMPOSE) exec pipeline /bin/bash

.PHONY: shell-ollama
shell-ollama:
	@echo "Opening shell in ollama container..."
	$(DOCKER_COMPOSE) exec ollama /bin/bash

# =============================================================================
# Ollama Model Management
# =============================================================================

.PHONY: ollama-pull
ollama-pull:
	@echo "Pulling nomic-embed-text model..."
	$(DOCKER_COMPOSE) exec ollama ollama pull nomic-embed-text
	@echo "✅ Model pulled successfully"

.PHONY: ollama-list
ollama-list:
	@echo "Installed Ollama models:"
	$(DOCKER_COMPOSE) exec ollama ollama list

MODEL ?= nomic-embed-text
.PHONY: ollama-pull-model
ollama-pull-model:
	@echo "Pulling model: $(MODEL)..."
	$(DOCKER_COMPOSE) exec ollama ollama pull $(MODEL)
	@echo "✅ Model $(MODEL) pulled successfully"

# =============================================================================
# Web UIs (opens in browser)
# =============================================================================

# Detect OS for open command
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OPEN_CMD := open
else ifeq ($(UNAME_S),Linux)
    OPEN_CMD := xdg-open
else
    OPEN_CMD := start
endif

.PHONY: ui-all
ui-all: ui-pipeline ui-minio ui-qdrant ui-ray
	@echo "✅ All UIs opened in browser"

.PHONY: ui-pipeline
ui-pipeline:
	@echo "Opening Pipeline API docs (Swagger)..."
	@$(OPEN_CMD) http://localhost:8000/docs

.PHONY: ui-minio
ui-minio:
	@echo "Opening MinIO Console..."
	@echo "   Login: minioadmin / minioadmin"
	@$(OPEN_CMD) http://localhost:9001

.PHONY: ui-qdrant
ui-qdrant:
	@echo "Opening Qdrant Dashboard..."
	@$(OPEN_CMD) http://localhost:6333/dashboard

.PHONY: ui-ray
ui-ray:
	@echo "Opening Ray Dashboard..."
	@$(OPEN_CMD) http://localhost:8265

.PHONY: ui-weaviate
ui-weaviate:
	@echo "Opening Weaviate GraphQL Console..."
	@$(OPEN_CMD) http://localhost:8080/v1/graphql

# =============================================================================
# Synthetic Data Generation
# =============================================================================

# Default values for synthetic data generation
COUNT ?= 1000
CHUNK_SIZE ?= 500
MINIO_ENDPOINT ?= http://localhost:9000

.PHONY: syntheticdata-generate
syntheticdata-generate:
	@uv run python syntheticdata/synthetic_data.py generate \
		--count $(COUNT) \
		--chunk-size $(CHUNK_SIZE)

.PHONY: syntheticdata-upload
syntheticdata-upload:
	@uv run python syntheticdata/synthetic_data.py upload \
		--endpoint $(MINIO_ENDPOINT)

.PHONY: syntheticdata-setup
syntheticdata-setup:
	@uv run python syntheticdata/synthetic_data.py setup \
		--count $(COUNT) \
		--chunk-size $(CHUNK_SIZE) \
		--endpoint $(MINIO_ENDPOINT)

.PHONY: syntheticdata-clean
syntheticdata-clean:
	@echo "Removing generated synthetic documents..."
	@rm -rf syntheticdata/inputs
	@echo "✅ Cleaned syntheticdata/inputs/"
