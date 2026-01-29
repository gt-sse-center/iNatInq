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
	@echo "├── Testing ──────────────────────────────────────────────────────────────────┤"
	@echo "│ make test              Run unit tests                                       │"
	@echo "│ make test-integration  Run integration tests (requires Docker)              │"
	@echo "│ make test-all          Run all tests (unit + integration)                   │"
	@echo "│ make test-cov          Run unit tests with coverage                         │"
	@echo "│ make test-cov-all      Run all tests with coverage                          │"
	@echo "├── Development ──────────────────────────────────────────────────────────────┤"
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
	@echo "├── Databricks (Azure) ───────────────────────────────────────────────────────┤"
	@echo "│ make azure-databricks-build  Create/update cluster from spec                │"
	@echo "│ make azure-databricks-up     Start cluster                                  │"
	@echo "│ make azure-databricks-down   Terminate cluster                              │"
	@echo "├── Docker Service Logs ──────────────────────────────────────────────────────┤"
	@echo "│ make logs-pipeline     Tail pipeline logs                                   │"
	@echo "│ make logs-ollama       Tail ollama logs                                     │"
	@echo "│ make logs-qdrant       Tail qdrant logs                                     │"
	@echo "│ make logs-ray          Tail ray-head logs                                   │"
	@echo "│ make logs-minio        Tail minio logs                                      │"
	@echo "│ make logs-weaviate     Tail weaviate logs                                  │"
	@echo "│ make logs-clip         Tail clip logs                                      │"
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
	@echo "│ make ui-weaviate       Open Weaviate Console                                 │"
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
DATABRICKS_ENV_FILE ?= zarf/databricks/dev/.env.local
DATABRICKS_CLUSTER_SPEC ?= zarf/databricks/dev/inatinq-azure-databricks-cluster.yml

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

.PHONY: test-unit
test-unit: test  ## Alias for 'test'

.PHONY: test-integration
test-integration:
	@echo "Running integration tests (requires Docker)..."
	uv run pytest tests/integration/ -v -m integration

.PHONY: test-integration-parallel
test-integration-parallel:
	@echo "Running integration tests in parallel..."
	uv run pytest tests/integration/ -v -m integration -n auto

.PHONY: test-all
test-all:
	@echo "Running all tests (unit + integration)..."
	uv run pytest tests/ -v

.PHONY: test-cov
test-cov:
	@echo "Running unit tests with coverage..."
	uv run pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

.PHONY: test-cov-all
test-cov-all:
	@echo "Running all tests with coverage..."
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

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
	@echo "   Pipeline API:     http://localhost:8000"
	@echo "   Pipeline Docs:    http://localhost:8000/docs"
	@echo "   MinIO Console:    http://localhost:9001"
	@echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "   Weaviate Console: http://localhost:8081"
	@echo "   Ray Dashboard:    http://localhost:8265"
	@echo "   Ollama:           http://localhost:11434"
	@echo "   CLIP API:         http://localhost:8001"

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
	@printf "│ CLIP:            "; curl -sf http://localhost:8001/ >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@echo "└──────────────────────────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "📝 Local Compose Endpoints:"
	@echo "   Pipeline API:     http://localhost:8000"
	@echo "   Pipeline Docs:    http://localhost:8000/docs"
	@echo "   MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)"
	@echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "   Weaviate Console: http://localhost:8081"
	@echo "   Ray Dashboard:    http://localhost:8265"
	@echo "   Ollama:           http://localhost:11434"
	@echo "   CLIP API:         http://localhost:8001"

.PHONY: smoke-providers
smoke-providers:
	@ENV_FILE=$(SMOKE_ENV_FILE) bash zarf/scripts/smoke_providers.sh

.PHONY: smoke-test
smoke-test: providers-health smoke-providers

.PHONY: providers-health
providers-health:
	@ENV_FILE=$(SMOKE_ENV_FILE) bash zarf/scripts/providers_health.sh

# =============================================================================
# Azure Databricks
# =============================================================================

.PHONY: azure-databricks-build
azure-databricks-build:
	@ENV_FILE=$(DATABRICKS_ENV_FILE) CLUSTER_SPEC_FILE=$(DATABRICKS_CLUSTER_SPEC) \
		bash zarf/databricks/azure-databricks-build.sh

.PHONY: azure-databricks-up
azure-databricks-up:
	@ENV_FILE=$(DATABRICKS_ENV_FILE) CLUSTER_SPEC_FILE=$(DATABRICKS_CLUSTER_SPEC) \
		bash zarf/databricks/azure-databricks-up.sh

.PHONY: azure-databricks-down
azure-databricks-down:
	@ENV_FILE=$(DATABRICKS_ENV_FILE) CLUSTER_SPEC_FILE=$(DATABRICKS_CLUSTER_SPEC) \
		bash zarf/databricks/azure-databricks-down.sh

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

.PHONY: logs-weaviate
logs-weaviate:
	$(DOCKER_COMPOSE) logs -f weaviate

.PHONY: logs-clip
logs-clip:
	$(DOCKER_COMPOSE) logs -f clip

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
ui-all: ui-pipeline ui-minio ui-qdrant ui-ray ui-weaviate
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
	@echo "Opening Weaviate Console..."
	@$(OPEN_CMD) http://localhost:8081

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

# =============================================================================
# Ray Job Operations
# =============================================================================

S3_PREFIX ?= inputs/
COLLECTION ?= documents

.PHONY: ray-job-submit
ray-job-submit:
	@echo "Submitting Ray job to process S3 prefix: $(S3_PREFIX)"
	@curl -sf -X POST "http://localhost:8000/ray/jobs" \
		-H "Content-Type: application/json" \
		-d '{"s3_prefix": "$(S3_PREFIX)", "collection": "$(COLLECTION)"}' | python3 -c "import sys, json; d=json.load(sys.stdin); print('Job ID:', d.get('job_id', 'N/A'))"

.PHONY: ray-job-status
ray-job-status:
	@echo "Fetching Ray job status..."
	@curl -sf "http://localhost:8265/api/jobs/" 2>/dev/null | python3 -c "import sys, json; jobs=json.load(sys.stdin); \
		[print(f\"{j.get('job_id', 'N/A')[:20]}: {j.get('status', 'unknown')}\") for j in jobs[:5]]" 2>/dev/null || echo "No jobs found or Ray unavailable"

.PHONY: ray-job-logs
ray-job-logs:
	@JOB_ID=$$(curl -sf "http://localhost:8265/api/jobs/" 2>/dev/null | python3 -c "import sys, json; jobs=json.load(sys.stdin); print(jobs[0]['job_id'] if jobs else '')" 2>/dev/null); \
	if [ -n "$$JOB_ID" ]; then \
		echo "Latest job logs ($$JOB_ID):"; \
		curl -sf "http://localhost:8265/api/jobs/$$JOB_ID/logs" 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin).get('logs', '')[-3000:])" 2>/dev/null; \
	else \
		echo "No jobs found"; \
	fi

# =============================================================================
# Document Counts
# =============================================================================

.PHONY: count-qdrant
count-qdrant:
	@echo "Counting documents in Qdrant collection '$(COLLECTION)'..."
	@curl -sf "http://localhost:6333/collections/$(COLLECTION)" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print('Qdrant documents:', d.get('result', {}).get('points_count', 0))" 2>/dev/null || echo "Qdrant: Collection not found or service unavailable"

.PHONY: count-weaviate
count-weaviate:
	@echo "Counting documents in Weaviate collection '$(COLLECTION)'..."
	@COLLECTION_CAP=$$(python3 -c "print('$(COLLECTION)'.capitalize())"); \
	curl -sf -X POST "http://localhost:8080/v1/graphql" \
		-H "Content-Type: application/json" \
		-d "{\"query\":\"{Aggregate{$$COLLECTION_CAP{meta{count}}}}\"}" 2>/dev/null | \
		python3 -c "import sys, json; d=json.load(sys.stdin); \
		count=d.get('data',{}).get('Aggregate',{}).get('$$COLLECTION_CAP',[{}])[0].get('meta',{}).get('count',0); \
		print('Weaviate documents:', count)" 2>/dev/null || echo "Weaviate: Collection not found or service unavailable"

.PHONY: count-all
count-all: count-qdrant count-weaviate

# =============================================================================
# Search Operations
# =============================================================================

QUERY ?= What is the story about?

.PHONY: search-qdrant
search-qdrant:
	@echo "Searching Qdrant for: '$(QUERY)'"
	@ENCODED_QUERY=$$(python3 -c 'import urllib.parse; print(urllib.parse.quote_plus("$(QUERY)"))'); \
	curl -sf "http://localhost:8000/search?query=$$ENCODED_QUERY&provider=qdrant&limit=3" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); \
		print('Results:', len(d.get('results', []))); \
		[print(f\"  - {r.get('payload', {}).get('text', '')[:100]}...\") for r in d.get('results', [])[:3]]" 2>/dev/null || echo "Search failed or service unavailable"

.PHONY: search-weaviate
search-weaviate:
	@echo "Searching Weaviate for: '$(QUERY)'"
	@ENCODED_QUERY=$$(python3 -c 'import urllib.parse; print(urllib.parse.quote_plus("$(QUERY)"))'); \
	curl -sf "http://localhost:8000/search?query=$$ENCODED_QUERY&provider=weaviate&limit=3" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); \
		print('Results:', len(d.get('results', []))); \
		[print(f\"  - {r.get('payload', {}).get('text', '')[:100]}...\") for r in d.get('results', [])[:3]]" 2>/dev/null || echo "Search failed or service unavailable"

.PHONY: search-compare
search-compare:
	@echo "Comparing search results across both providers..."
	@echo ""
	@$(MAKE) search-qdrant QUERY="$(QUERY)"
	@echo ""
	@$(MAKE) search-weaviate QUERY="$(QUERY)"

# =============================================================================
# S3/MinIO Operations
# =============================================================================

.PHONY: s3-count
s3-count:
	@echo "Counting objects in MinIO bucket..."
	@curl -sf "http://localhost:8000/api/v1/s3/objects?prefix=$(S3_PREFIX)&limit=10000" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print('S3 objects:', d.get('count', len(d.get('keys', []))))" 2>/dev/null || echo "S3 count failed or service unavailable"

.PHONY: s3-clear
s3-clear:
	@echo "Clearing S3 bucket..."
	@curl -sf -X DELETE "http://localhost:8000/api/v1/s3/objects?prefix=$(S3_PREFIX)" 2>/dev/null && echo "✅ S3 cleared" || echo "S3 clear failed"

# =============================================================================
# Vector DB Operations
# =============================================================================

.PHONY: qdrant-clear
qdrant-clear:
	@echo "Deleting Qdrant collection '$(COLLECTION)'..."
	@curl -sf -X DELETE "http://localhost:6333/collections/$(COLLECTION)" 2>/dev/null && echo "✅ Qdrant collection deleted" || echo "Qdrant delete failed or collection doesn't exist"

.PHONY: weaviate-clear
weaviate-clear:
	@echo "Deleting Weaviate collection '$(COLLECTION)'..."
	@curl -sf -X DELETE "http://localhost:8080/v1/schema/$(COLLECTION)" 2>/dev/null && echo "✅ Weaviate collection deleted" || echo "Weaviate delete failed or collection doesn't exist"

.PHONY: vectordb-clear
vectordb-clear: qdrant-clear weaviate-clear
	@echo "✅ Both vector DBs cleared"

# =============================================================================
# End-to-End Pipeline
# =============================================================================

.PHONY: e2e-test
e2e-test:
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║                         End-to-End Pipeline Test                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1: Clear vector DBs..."
	@$(MAKE) vectordb-clear COLLECTION=$(COLLECTION)
	@echo ""
	@echo "Step 2: Seed S3 with $(COUNT) documents..."
	@$(MAKE) syntheticdata-setup COUNT=$(COUNT)
	@echo ""
	@echo "Step 3: Submit Ray job..."
	@$(MAKE) ray-job-submit S3_PREFIX=$(S3_PREFIX) COLLECTION=$(COLLECTION)
	@echo ""
	@echo "Waiting 30 seconds for job to complete..."
	@sleep 30
	@echo ""
	@echo "Step 4: Check document counts..."
	@$(MAKE) count-all COLLECTION=$(COLLECTION)
	@echo ""
	@echo "Step 5: Test search..."
	@$(MAKE) search-compare QUERY="$(QUERY)"
	@echo ""
	@echo "✅ End-to-end test complete!"
