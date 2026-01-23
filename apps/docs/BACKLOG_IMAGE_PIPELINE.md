# Image Indexing Pipeline - Backlog Tickets

These tickets add an image indexing pipeline that runs alongside the existing text pipeline.

---

## Epic: Multi-Modal Embedding Support

### IMG-001: Add multi-modal embedding provider interface

**Points:** 3  
**Level:** Mid  
**Type:** Feature

**Context:**  
The current `EmbeddingProvider` interface in `src/clients/interfaces/embedding.py` only supports text input. We need to extend or create a parallel interface for image embeddings using models like CLIP, OpenCLIP, or Ollama's LLaVA.

**Acceptance Criteria:**

- [ ] Create `ImageEmbeddingProvider` protocol in `src/clients/interfaces/embedding.py`
- [ ] Define `embed_image(image_bytes: bytes) -> list[float]` method
- [ ] Define `embed_image_batch(images: list[bytes]) -> list[list[float]]` method
- [ ] Define `embed_image_async(...)` and `embed_image_batch_async(...)` async variants
- [ ] Add `vector_size` property (CLIP: 512 or 768, depends on model)
- [ ] Keep existing `EmbeddingProvider` unchanged for text

**Files to modify:**

- `src/clients/interfaces/embedding.py`

**Technical Notes:**

```python
class ImageEmbeddingProvider(Protocol):
    """Protocol for image embedding providers (CLIP, LLaVA, etc.)."""
    
    @property
    def vector_size(self) -> int: ...
    
    def embed_image(self, image_bytes: bytes) -> list[float]: ...
    
    async def embed_image_async(self, image_bytes: bytes) -> list[float]: ...
    
    def embed_image_batch(self, images: list[bytes]) -> list[list[float]]: ...
```

---

### IMG-002: Implement CLIP embedding client

**Points:** 5  
**Level:** Senior  
**Depends on:** IMG-001

**Context:**  
CLIP (Contrastive Language-Image Pre-training) is the standard for multi-modal embeddings. We need a client that can generate embeddings for images using a CLIP model hosted via Ollama, Hugging Face, or a dedicated service.

**Acceptance Criteria:**

- [ ] Create `src/clients/clip.py` with `CLIPClient` class
- [ ] Implement `ImageEmbeddingProvider` protocol
- [ ] Support Ollama's `llava` or similar multi-modal model
- [ ] Alternative: Support Hugging Face's `openai/clip-vit-base-patch32`
- [ ] Add circuit breaker and retry logic (follow `OllamaClient` patterns)
- [ ] Add `CLIPConfig` to `src/config.py` with model name, endpoint, timeout
- [ ] Add unit tests in `tests/unit/clients/test_clip.py`

**Files to create:**

- `src/clients/clip.py`

**Files to modify:**

- `src/config.py` - Add `CLIPConfig` or extend `EmbeddingConfig`

**Technical Notes:**

```python
# Option 1: Ollama with llava
POST http://ollama:11434/api/embeddings
{
    "model": "llava",
    "prompt": "",  # Empty for image-only
    "images": ["base64_encoded_image"]
}

# Option 2: Dedicated CLIP service
POST http://clip-service:8080/embed
Content-Type: image/jpeg
<binary image data>
```

---

## Epic: Image Processing Infrastructure

### IMG-003: Add image content fetcher for S3

**Points:** 3  
**Level:** Mid  
**Type:** Feature

**Context:**  
Current `S3ContentFetcher` in `src/core/ingestion/interfaces/operations.py` reads text content and decodes as UTF-8. We need a parallel fetcher that reads binary image data and performs basic preprocessing.

**Acceptance Criteria:**

- [ ] Create `ImageContentFetcher` class in `src/core/ingestion/interfaces/operations.py`
- [ ] Method `fetch_image(s3_key: str) -> ImageContentResult`
- [ ] Return raw bytes plus metadata (size, format detected from magic bytes)
- [ ] Support JPEG, PNG, WebP, GIF formats
- [ ] Add basic validation (min/max size, valid format)
- [ ] Do NOT resize here - leave that to embedding step
- [ ] Add `ImageContentResult` type to `types.py`

**Files to modify:**

- `src/core/ingestion/interfaces/operations.py`
- `src/core/ingestion/interfaces/types.py`

**Technical Notes:**

```python
@attrs.define(frozen=True, slots=True)
class ImageContentResult:
    s3_key: str
    image_bytes: bytes
    format: str  # "jpeg", "png", "webp", "gif"
    width: int | None = None
    height: int | None = None
```

---

### IMG-004: Add image preprocessing utilities

**Points:** 2  
**Level:** Junior/Mid  
**Type:** Feature

**Context:**  
Images need preprocessing before embedding: resize to model's expected size, normalize, convert to RGB. This should be a standalone utility.

**Acceptance Criteria:**

- [ ] Create `src/core/ingestion/image_utils.py`
- [ ] Function `resize_for_embedding(image_bytes, max_size=224) -> bytes`
- [ ] Function `validate_image(image_bytes) -> tuple[bool, str]` (valid, error_msg)
- [ ] Use Pillow (PIL) for image operations
- [ ] Convert all images to RGB (handle RGBA, grayscale)
- [ ] Preserve aspect ratio, pad if needed
- [ ] Add unit tests with sample images

**Files to create:**

- `src/core/ingestion/image_utils.py`
- `tests/unit/core/ingestion/test_image_utils.py`

**Dependencies to add:**

- `Pillow` (likely already installed)

---

## Epic: Image-Specific Vector Storage

### IMG-005: Create image collection schema for Qdrant

**Points:** 2  
**Level:** Mid  
**Type:** Feature

**Context:**  
Image embeddings need different metadata than text: image dimensions, format, thumbnail URL, etc. We need a separate collection or extended schema.

**Acceptance Criteria:**

- [ ] Create `ensure_image_collection_async()` method in `QdrantClientWrapper`
- [ ] Collection name pattern: `{collection}_images` (e.g., `documents_images`)
- [ ] Payload fields: `s3_key`, `s3_uri`, `format`, `width`, `height`, `thumbnail_key`
- [ ] Vector size: configurable (CLIP default: 512)
- [ ] Add to existing `ensure_collection_async()` or create parallel method

**Files to modify:**

- `src/clients/qdrant.py`

---

### IMG-006: Create image collection schema for Weaviate

**Points:** 2  
**Level:** Mid  
**Type:** Feature  
**Depends on:** IMG-005

**Context:**  
Mirror the Qdrant image collection in Weaviate with appropriate class definition.

**Acceptance Criteria:**

- [ ] Create `ensure_image_collection_async()` method in `WeaviateClientWrapper`
- [ ] Class name pattern: `{Collection}Images` (e.g., `DocumentsImages`)
- [ ] Properties: `s3_key`, `s3_uri`, `format`, `width`, `height`, `thumbnail_key`
- [ ] Vector config: CLIP dimensions
- [ ] Add to existing `ensure_collection_async()` or create parallel method

**Files to modify:**

- `src/clients/weaviate.py`

---

## Epic: Image Ingestion Pipeline

### IMG-007: Create ImageProcessingPipeline for Ray

**Points:** 5  
**Level:** Senior  
**Depends on:** IMG-001, IMG-002, IMG-003, IMG-004

**Context:**  
Create a parallel pipeline class that processes images through S3 → Preprocessing → CLIP Embedding → Vector DBs. This should mirror the structure of the existing text pipeline.

**Acceptance Criteria:**

- [ ] Create `src/core/ingestion/ray/image_processing.py`
- [ ] Create `ImageProcessingPipeline` class mirroring `RayProcessingPipeline`
- [ ] Ray remote function: `process_image_batch_ray(keys, config)`
- [ ] Integrate `ImageContentFetcher`, `CLIPClient`, `VectorDBUpserter`
- [ ] Reuse existing rate limiter and circuit breaker infrastructure
- [ ] Add configuration for image-specific batch sizes (smaller than text)
- [ ] Add to `RayJobConfig` in `config.py`: `image_batch_size`, `image_embed_batch_size`

**Files to create:**

- `src/core/ingestion/ray/image_processing.py`

**Files to modify:**

- `src/config.py` - Add image pipeline config

**Technical Notes:**

```python
@ray.remote(num_cpus=1, max_retries=3)
def process_image_batch_ray(
    s3_keys: list[str],
    config: RayProcessingConfig,
) -> list[ProcessingResult]:
    """Process a batch of images through the embedding pipeline."""
    ...
```

---

### IMG-008: Add image job submission API endpoint

**Points:** 3  
**Level:** Mid  
**Depends on:** IMG-007

**Context:**  
Need API endpoint to trigger image ingestion jobs, similar to existing `/ray/jobs` but for images.

**Acceptance Criteria:**

- [ ] Add `POST /ray/jobs/images` endpoint to `src/api/routes.py`
- [ ] Request body: `{"s3_bucket": "...", "s3_prefix": "images/", "collection": "..."}`
- [ ] Filter S3 objects by image extensions (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`)
- [ ] Submit image processing job to Ray
- [ ] Return job ID for status tracking
- [ ] Reuse existing job status/logs endpoints

**Files to modify:**

- `src/api/routes.py`
- `src/api/models.py` - Add request/response models
- `src/core/services/ray_service.py` - Add `submit_image_job()`

---

## Epic: Image Search

### IMG-009: Add image-to-image search endpoint

**Points:** 3  
**Level:** Mid  
**Depends on:** IMG-005, IMG-006

**Context:**  
Search for similar images by uploading an image (reverse image search).

**Acceptance Criteria:**

- [ ] Add `POST /search/images` endpoint
- [ ] Accept `multipart/form-data` with image file
- [ ] Generate embedding for uploaded image using CLIP
- [ ] Search image collections in Qdrant/Weaviate
- [ ] Return similar images with scores and metadata
- [ ] Add `provider` query param (qdrant/weaviate/all)

**Files to modify:**

- `src/api/routes.py`
- `src/api/models.py`

**Request/Response:**

```
POST /search/images?limit=10&provider=qdrant
Content-Type: multipart/form-data

file: <image binary>

Response:
{
    "results": [
        {"s3_key": "images/cat.jpg", "score": 0.95, "thumbnail": "..."},
        ...
    ]
}
```

---

### IMG-010: Add text-to-image search endpoint

**Points:** 3  
**Level:** Mid  
**Depends on:** IMG-002, IMG-005, IMG-006

**Context:**  
CLIP embeddings allow searching images using text queries (e.g., "sunset over ocean" finds matching images).

**Acceptance Criteria:**

- [ ] Add `GET /search/images` endpoint (text query variant)
- [ ] Accept `q` query parameter with text description
- [ ] Generate CLIP text embedding for query
- [ ] Search image collections
- [ ] Return matching images with scores
- [ ] Note: Requires CLIP model that supports both text and image encoding

**Files to modify:**

- `src/api/routes.py`
- `src/core/services/search_service.py`

---

## Epic: Configuration & Operations

### IMG-011: Add image pipeline configuration

**Points:** 2  
**Level:** Junior/Mid  
**Type:** Feature

**Context:**  
Environment variables and config classes for the image pipeline.

**Acceptance Criteria:**

- [ ] Add to `src/config.py`:
  - `IMAGE_EMBEDDING_PROVIDER` (clip, llava, etc.)
  - `CLIP_MODEL` (model name)
  - `CLIP_ENDPOINT` (service URL)
  - `IMAGE_BATCH_SIZE` (default: 10, smaller than text)
  - `IMAGE_MAX_SIZE_MB` (reject images larger than this)
  - `IMAGE_TARGET_SIZE` (resize dimension, default: 224)
- [ ] Add to `pipeline.env` with defaults
- [ ] Document in README

**Files to modify:**

- `src/config.py`
- `zarf/compose/dev/pipeline.env`

---

### IMG-012: Add Makefile commands for image pipeline

**Points:** 1  
**Level:** Junior  
**Depends on:** IMG-008

**Context:**  
Operator commands for image ingestion workflow.

**Acceptance Criteria:**

- [ ] `make seed-images COUNT=100` - Generate/upload test images to S3
- [ ] `make ingest-images PREFIX=images/` - Trigger image ingestion job
- [ ] `make search-image-qdrant FILE=test.jpg` - Image-to-image search
- [ ] `make search-image-text QUERY="cat sitting"` - Text-to-image search

**Files to modify:**

- `Makefile`

---

## Summary

| Ticket | Points | Level | Epic |
|--------|--------|-------|------|
| IMG-001 | 3 | Mid | Multi-Modal Embedding |
| IMG-002 | 5 | Senior | Multi-Modal Embedding |
| IMG-003 | 3 | Mid | Image Processing |
| IMG-004 | 2 | Junior/Mid | Image Processing |
| IMG-005 | 2 | Mid | Vector Storage |
| IMG-006 | 2 | Mid | Vector Storage |
| IMG-007 | 5 | Senior | Ingestion Pipeline |
| IMG-008 | 3 | Mid | Ingestion Pipeline |
| IMG-009 | 3 | Mid | Image Search |
| IMG-010 | 3 | Mid | Image Search |
| IMG-011 | 2 | Junior/Mid | Configuration |
| IMG-012 | 1 | Junior | Configuration |
| IMG-013 | 1 | Junior | Infrastructure |

**Total:** 35 points across 13 tickets

---

## Suggested Sprint Plan

**Sprint 1 (Foundation):** IMG-001, IMG-003, IMG-004, IMG-011 = 10 pts  
**Sprint 2 (Embedding):** IMG-002, IMG-005, IMG-006 = 9 pts  
**Sprint 3 (Pipeline):** IMG-007, IMG-008, IMG-012 = 9 pts  
**Sprint 4 (Search):** IMG-009, IMG-010 = 6 pts

---

### IMG-013: Add ai4all/clip container to docker-compose

**Points:** 1  
**Level:** Junior  
**Type:** Infrastructure

**Context:**  
The image embedding pipeline requires the `ai4all/clip` Docker container for generating embeddings. This container provides a REST API for CLIP embeddings at `/embedding/image` and `/embedding/text`.

**Acceptance Criteria:**

- [ ] Add `clip` service to `zarf/compose/dev/docker-compose.yaml`
- [ ] Use `ai4all/clip:latest` image
- [ ] Expose port 8000
- [ ] Add health check for `/` endpoint
- [ ] Add `CLIP_URL` environment variable to pipeline service
- [ ] Update `.env.local` with default `CLIP_URL=http://clip:8000`
- [ ] Document in README

**Files to modify:**

- `zarf/compose/dev/docker-compose.yaml`
- `zarf/compose/dev/.env.local`

**Technical Notes:**

```yaml
clip:
  image: ai4all/clip:latest
  container_name: clip
  ports:
    - "8000:8000"
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/"]
    interval: 30s
    timeout: 10s
    retries: 3
  restart: unless-stopped
```
