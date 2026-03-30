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
	@echo "│ make validate-config   Validate YAML config files                           │"
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
	@echo "│ make azure-databricks-cdc-notebooks Upload + run CDC notebook validation    │"
	@echo "│ make azure-databricks-configure-minio-s3a Configure MinIO S3A secret/conf   │"
	@echo "├── Docker Service Logs ──────────────────────────────────────────────────────┤"
	@echo "│ make logs-pipeline     Tail pipeline logs                                   │"
	@echo "│ make logs-qdrant       Tail qdrant logs                                     │"
	@echo "│ make logs-ray          Tail ray-head logs                                   │"
	@echo "│ make logs-minio        Tail minio logs                                      │"
	@echo "│ make logs-clip         Tail clip logs                                       │"
	@echo "│ make logs-redis        Tail redis logs                                      │"
	@echo "├── Docker Shell Access ──────────────────────────────────────────────────────┤"
	@echo "│ make shell-pipeline    Shell into pipeline container                        │"
	@echo "│ make shell-redis       Shell into redis container cli                       │"
	@echo "├── Web UIs (opens in browser) ───────────────────────────────────────────────┤"
	@echo "│ make ui-all            Open all web UIs                                     │"
	@echo "│ make ui-pipeline       Open Pipeline API docs (Swagger)                     │"
	@echo "│ make ui-minio          Open MinIO Console                                   │"
	@echo "│ make ui-qdrant         Open Qdrant Dashboard                                │"
	@echo "│ make ui-ray            Open Ray Dashboard                                   │"
	@echo "├── Docker Scaling ───────────────────────────────────────────────────────────┤"
	@echo "│ make ray-scale N=3     Scale Ray workers to N replicas                      │"
	@echo "├── Synthetic Images ─────────────────────────────────────────────────────────┤"
	@echo "│ make synthetic-images-generate Generate test images with shapes/colors      │"
	@echo "│ make synthetic-images-upload   Upload images to MinIO                       │"
	@echo "│ make synthetic-images-setup    Generate and upload images                   │"
	@echo "│ make synthetic-images-clean    Remove generated images                      │"
	@echo "├── Image Pipeline ───────────────────────────────────────────────────────────┤"
	@echo "│ make ray-image-job-submit     Submit image processing Ray job               │"
	@echo "│ make count-images-qdrant      Count images in Qdrant                        │"
	@echo "│ make count-images-all         Count images in Qdrant                         │"
	@echo "│ make search-images-qdrant     Text-to-image search (Qdrant)                 │"
	@echo "│ make search-images-compare    Compare search across providers               │"
	@echo "│ make image-search-demo        Search with presigned URLs                    │"
	@echo "│ make image-search-download    Search and download images                    │"
	@echo "│ make image-search-open        Search and open top result in browser         │"
	@echo "│ make vectordb-clear-images    Clear image collections                       │"
	@echo "│ make e2e-test-images          Full E2E image pipeline test                  │"
	@echo "└─────────────────────────────────────────────────────────────────────────────┘"

# =============================================================================
# Variables
# =============================================================================
COMPOSE_FILE := zarf/compose/dev/docker-compose.yaml
DOCKER_COMPOSE := docker compose -f $(COMPOSE_FILE)
SMOKE_ENV_FILE ?= zarf/compose/dev/.env.local
DATABRICKS_ENV_FILE ?= zarf/databricks/dev/.env.local
DATABRICKS_CLUSTER_SPEC ?= zarf/databricks/dev/inatinq-azure-databricks-cluster.json

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

.PHONY: validate-config
validate-config:
	@echo "Validating YAML configuration..."
	@uv run python -c "\
	from config_loader import load_yaml_config; \
	c = load_yaml_config(); \
	print('Base config: OK (' + str(len(c)) + ' top-level keys)'); \
	import os, pathlib; \
	env_dir = pathlib.Path('configs/environments'); \
	[print(f'  + {f.stem}: OK') for f in sorted(env_dir.glob('*.yaml')) if load_yaml_config(env=f.stem) is not None]; \
	print('All configs valid.')"

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
	@echo "   Ray Dashboard:    http://localhost:8265"
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
	@printf "│ MinIO:           "; curl -sf http://localhost:9000/minio/health/live >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@printf "│ Ray:             "; curl -sf http://localhost:8265 >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@printf "│ Redis:           "; docker exec redis redis-cli ping  >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@printf "│ CLIP:            "; curl -sf http://localhost:8001/ >/dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@echo "└──────────────────────────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "📝 Local Compose Endpoints:"
	@echo "   Pipeline API:     http://localhost:8000"
	@echo "   Pipeline Docs:    http://localhost:8000/docs"
	@echo "   MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)"
	@echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "   Ray Dashboard:    http://localhost:8265"
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
azure-databricks-build: databricks-env-check
	@ENV_FILE=$(DATABRICKS_ENV_FILE) CLUSTER_SPEC_FILE=$(DATABRICKS_CLUSTER_SPEC) \
		zarf/databricks/azure-databricks-build.py

.PHONY: azure-databricks-up
azure-databricks-up: databricks-env-check
	@ENV_FILE=$(DATABRICKS_ENV_FILE) CLUSTER_SPEC_FILE=$(DATABRICKS_CLUSTER_SPEC) \
		zarf/databricks/azure-databricks-up.py

.PHONY: azure-databricks-down
azure-databricks-down: databricks-env-check
	@ENV_FILE=$(DATABRICKS_ENV_FILE) CLUSTER_SPEC_FILE=$(DATABRICKS_CLUSTER_SPEC) \
		zarf/databricks/azure-databricks-down.py

.PHONY: azure-databricks-cdc-notebooks
azure-databricks-cdc-notebooks: databricks-env-check
	@uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
		--env-file $(DATABRICKS_ENV_FILE) \
		--upload-notebooks \
		--run-notebooks

.PHONY: azure-databricks-configure-minio-s3a
azure-databricks-configure-minio-s3a: databricks-env-check
	@ENV_FILE=$(DATABRICKS_ENV_FILE) \
		zarf/databricks/azure-databricks-configure-minio-s3a.py

.PHONY: databricks-env-check
databricks-env-check:
	@if [ ! -f "$(DATABRICKS_ENV_FILE)" ]; then \
		echo "Missing Databricks env file: $(DATABRICKS_ENV_FILE)"; \
		echo "Create it from zarf/databricks/dev/env.local.example"; \
		exit 1; \
	fi

# =============================================================================
# Docker Service Logs
# =============================================================================

.PHONY: logs-pipeline
logs-pipeline:
	$(DOCKER_COMPOSE) logs -f pipeline

.PHONY: logs-qdrant
logs-qdrant:
	$(DOCKER_COMPOSE) logs -f qdrant

.PHONY: logs-ray
logs-ray:
	$(DOCKER_COMPOSE) logs -f ray-head

.PHONY: logs-minio
logs-minio:
	$(DOCKER_COMPOSE) logs -f minio

.PHONY: logs-clip
logs-clip:
	$(DOCKER_COMPOSE) logs -f clip

.PHONY: logs-redis
logs-redis:
	$(DOCKER_COMPOSE) logs -f redis

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

.PHONY: shell-redis
shell-redis:
	@echo "Opening red-cli in Redis container..."
	$(DOCKER_COMPOSE) exec -it redis redis-cli

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

# =============================================================================
# Synthetic Data Generation
# =============================================================================

# Default values for synthetic data generation
COUNT ?= 1000
CHUNK_SIZE ?= 500
MINIO_ENDPOINT ?= http://localhost:9000

# Default values for synthetic image generation
IMAGE_COUNT ?= 100
IMAGE_SIZE ?= 512
IMAGE_PREFIX ?= images/
IMAGE_COLLECTION ?= documents
IMAGE_QUERY ?= red circle

# --- Image Generation ---
.PHONY: synthetic-images-generate
synthetic-images-generate:
	@uv run python syntheticdata/synthetic_data.py generate-images \
		--count $(IMAGE_COUNT) \
		--size $(IMAGE_SIZE)

.PHONY: synthetic-images-upload
synthetic-images-upload:
	@uv run python syntheticdata/synthetic_data.py upload-images \
		--endpoint $(MINIO_ENDPOINT) \
		--prefix $(IMAGE_PREFIX)

.PHONY: synthetic-images-setup
synthetic-images-setup:
	@uv run python syntheticdata/synthetic_data.py setup-images \
		--count $(IMAGE_COUNT) \
		--size $(IMAGE_SIZE) \
		--endpoint $(MINIO_ENDPOINT) \
		--prefix $(IMAGE_PREFIX)

.PHONY: synthetic-images-clean
synthetic-images-clean:
	@echo "Removing generated synthetic images..."
	@rm -rf syntheticdata/data/imgs
	@echo "✅ Cleaned syntheticdata/data/imgs/"


.PHONY: syntheticimages-generate syntheticimages-upload syntheticimages-setup syntheticimages-clean
syntheticimages-generate: synthetic-images-generate
syntheticimages-upload: synthetic-images-upload
syntheticimages-setup: synthetic-images-setup
syntheticimages-clean: synthetic-images-clean

# =============================================================================
# Ray Job Operations
# =============================================================================

S3_PREFIX ?= images/
COLLECTION ?= documents

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

.PHONY: ray-image-job-submit
ray-image-job-submit:
	@echo "Submitting Ray image job to process S3 prefix: $(IMAGE_PREFIX)"
	@curl -sf -X POST "http://localhost:8000/ray/jobs/images" \
		-H "Content-Type: application/json" \
		-d '{"s3_bucket": "pipeline", "s3_prefix": "$(IMAGE_PREFIX)", "collection": "$(IMAGE_COLLECTION)"}' | python3 -c "import sys, json; d=json.load(sys.stdin); print('Job ID:', d.get('job_id', 'N/A'))"

# =============================================================================
# Document Counts
# =============================================================================

.PHONY: count-qdrant
count-qdrant:
	@echo "Counting documents in Qdrant collection '$(COLLECTION)'..."
	@curl -sf "http://localhost:6333/collections/$(COLLECTION)" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print('Qdrant documents:', d.get('result', {}).get('points_count', 0))" 2>/dev/null || echo "Qdrant: Collection not found or service unavailable"

.PHONY: count-all
count-all: count-qdrant

# =============================================================================
# Image Counts
# =============================================================================

.PHONY: count-images-qdrant
count-images-qdrant:
	@echo "Counting images in Qdrant collection '$(IMAGE_COLLECTION)'..."
	@curl -sf "http://localhost:6333/collections/$(IMAGE_COLLECTION)" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print('Qdrant images:', d.get('result', {}).get('points_count', 0))" 2>/dev/null || echo "Qdrant: Collection not found or service unavailable"

.PHONY: count-images-all
count-images-all: count-images-qdrant

# =============================================================================
# Search Operations
# =============================================================================

QUERY ?= What is the story about?

.PHONY: search-qdrant
search-qdrant:
	@echo "Searching images in Qdrant for: '$(QUERY)'"
	@ENCODED_QUERY=$$(python3 -c 'import urllib.parse; print(urllib.parse.quote_plus("$(QUERY)"))'); \
	curl -sf "http://localhost:8000/search/images?q=$$ENCODED_QUERY&provider=qdrant&limit=3" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); \
		print('Results:', len(d.get('results', []))); \
		[print(f\"  - {r.get('s3_key', 'N/A')} (score: {r.get('score', 0):.3f})\") for r in d.get('results', [])[:3]]" 2>/dev/null || echo "Search failed or service unavailable"

# =============================================================================
# Image Search Operations
# =============================================================================

.PHONY: search-images-qdrant
search-images-qdrant:
	@echo "Searching images in Qdrant for: '$(IMAGE_QUERY)'"
	@ENCODED_QUERY=$$(python3 -c 'import urllib.parse; print(urllib.parse.quote_plus("$(IMAGE_QUERY)"))'); \
	curl -sf "http://localhost:8000/search/images?q=$$ENCODED_QUERY&provider=qdrant&limit=3" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); \
		print('Results:', len(d.get('results', []))); \
		[print(f\"  - {r.get('s3_key', 'N/A')} (score: {r.get('score', 0):.3f})\") for r in d.get('results', [])[:3]]" 2>/dev/null || echo "Search failed or service unavailable"

.PHONY: search-images-compare
search-images-compare:
	@echo "Searching images..."
	@echo ""
	@$(MAKE) search-images-qdrant IMAGE_QUERY="$(IMAGE_QUERY)"

# Image search demo with presigned URLs and download
.PHONY: image-search-demo
image-search-demo:
	@uv run python scripts/image_search_demo.py "$(IMAGE_QUERY)" --limit 5 --urls

.PHONY: image-search-download
image-search-download:
	@uv run python scripts/image_search_demo.py "$(IMAGE_QUERY)" --limit 5 --download ./image-results

.PHONY: image-search-open
image-search-open:
	@uv run python scripts/image_search_demo.py "$(IMAGE_QUERY)" --limit 1 --open

# =============================================================================
# S3/MinIO Operations
# =============================================================================

.PHONY: s3-count
s3-count:
	@echo "Counting objects in MinIO bucket..."
	@docker compose -f $(COMPOSE_DEV_FILE) exec -T minio mc ls --recursive minio/$(S3_BUCKET)/$(S3_PREFIX) 2>/dev/null | wc -l | xargs -I {} echo "S3 objects: {}" || echo "S3 count failed or service unavailable"

.PHONY: s3-clear
s3-clear:
	@echo "Clearing S3 bucket at prefix $(S3_PREFIX)..."
	@docker compose -f $(COMPOSE_DEV_FILE) exec -T minio mc rm --recursive --force minio/$(S3_BUCKET)/$(S3_PREFIX) 2>/dev/null && echo "✅ S3 cleared" || echo "S3 clear failed"

# =============================================================================
# Vector DB Operations
# =============================================================================

.PHONY: qdrant-clear
qdrant-clear:
	@echo "Deleting Qdrant collection '$(COLLECTION)'..."
	@curl -sf -X DELETE "http://localhost:6333/collections/$(COLLECTION)" 2>/dev/null && echo "✅ Qdrant collection deleted" || echo "Qdrant delete failed or collection doesn't exist"

.PHONY: vectordb-clear
vectordb-clear: qdrant-clear
	@echo "✅ Vector DB cleared"

# Image collection clear targets
.PHONY: qdrant-clear-images
qdrant-clear-images:
	@echo "Deleting Qdrant image collection '$(IMAGE_COLLECTION)'..."
	@curl -sf -X DELETE "http://localhost:6333/collections/$(IMAGE_COLLECTION)" 2>/dev/null && echo "✅ Qdrant image collection deleted" || echo "Qdrant delete failed or collection doesn't exist"

.PHONY: vectordb-clear-images
vectordb-clear-images: qdrant-clear-images
	@echo "✅ Image collection cleared"

# =============================================================================
# End-to-End Image Pipeline
# =============================================================================

.PHONY: e2e-test-images
e2e-test-images:
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║                      End-to-End Image Pipeline Test                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1: Clear image collections..."
	@$(MAKE) vectordb-clear-images IMAGE_COLLECTION=$(IMAGE_COLLECTION)
	@echo ""
	@echo "Step 2: Generate and upload $(IMAGE_COUNT) images..."
	@$(MAKE) synthetic-images-setup IMAGE_COUNT=$(IMAGE_COUNT) IMAGE_SIZE=$(IMAGE_SIZE) IMAGE_PREFIX=$(IMAGE_PREFIX)
	@echo ""
	@echo "Step 3: Submit Ray image job..."
	@$(MAKE) ray-image-job-submit IMAGE_PREFIX=$(IMAGE_PREFIX) IMAGE_COLLECTION=$(IMAGE_COLLECTION)
	@echo ""
	@echo "Waiting 60 seconds for image job to complete..."
	@sleep 60
	@echo ""
	@echo "Step 4: Check image counts..."
	@$(MAKE) count-images-all IMAGE_COLLECTION=$(IMAGE_COLLECTION)
	@echo ""
	@echo "Step 5: Test image search..."
	@$(MAKE) search-images-compare IMAGE_QUERY="$(IMAGE_QUERY)"
	@echo ""
	@echo "✅ End-to-end image test complete!"
