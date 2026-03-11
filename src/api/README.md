# API Routes

FastAPI HTTP layer for the pipeline service: request/response serialization, validation via Pydantic, and error translation from `PipelineError` to HTTP status codes.

## Design

- **Thin controllers**: Routes delegate to services; no business logic in handlers.
- **Validation**: Pydantic models in `api.models` validate requests and drive OpenAPI schema.
- **Errors**: `ExceptionHandlerMiddleware` maps `BadRequestError` → 400, `UpstreamError` → 502, `PipelineError` → 500.

## Endpoints

### Health

| Method | Path       | Description                                                                 |
| ------ | ---------- | --------------------------------------------------------------------------- |
| `GET`  | `/healthz` | Liveness/readiness probe. Returns `{"status": "ok"}`. No dependency checks. |

### Image search

| Method | Path             | Description                                                              |
| ------ | ---------------- | ------------------------------------------------------------------------ |
| `GET`  | `/search/images` | Text-to-image search over vector DB image collections (CLIP embeddings). |

**Query:** `q` (required), `limit` (default 10, max 100), `collection` (optional), `provider` (`qdrant` \| `weaviate`, optional).  
**Response:** `query`, `model`, `collection`, `provider`, `results` (id, score, s3_key, s3_uri, format, width, height, thumbnail_key), `total`.  
**Errors:** 400 (empty/invalid query or limit), 404 (collection missing), 502 (CLIP or vector DB failure).

### Ray jobs (image ingestion)

| Method   | Path                      | Description                                                                        |
| -------- | ------------------------- | ---------------------------------------------------------------------------------- |
| `POST`   | `/ray/jobs/images`        | Submit Ray job to process S3 images → CLIP → vector DB. Returns 202 with `job_id`. |
| `GET`    | `/ray/jobs/{job_id}`      | Job status.                                                                        |
| `GET`    | `/ray/jobs/{job_id}/logs` | Job logs.                                                                          |
| `DELETE` | `/ray/jobs/{job_id}`      | Stop job.                                                                          |

**POST body:** `s3_bucket`, `s3_prefix`, `collection`; optional `image_max_items`, `image_page_size`.  
**Response (202):** `job_id`, `status`, `namespace`, `s3_bucket`, `s3_prefix`, `collection`, `submitted_at`.

### Databricks jobs (image ingestion)

| Method   | Path                             | Description                                                                        |
| -------- | -------------------------------- | ---------------------------------------------------------------------------------- |
| `POST`   | `/databricks/jobs/images`        | Submit Databricks image job (S3 or iNaturalist source). Returns 202 with `run_id`. |
| `POST`   | `/databricks/jobs/cdc-producer` | Submit Databricks CDC producer (Auto Loader) job. Returns 202 with `run_id`.      |
| `GET`    | `/databricks/jobs/{run_id}`      | Run status (life_cycle_state, result_state, state_message).                        |
| `GET`    | `/databricks/jobs/{run_id}/logs` | Run output/logs.                                                                   |
| `DELETE` | `/databricks/jobs/{run_id}`      | Stop run.                                                                          |

**POST body:** `source` (`s3` \| `inat`), `collection`; optional `s3_prefix`, `image_max_items`, `image_page_size`.  
**Response (202):** `run_id`, `status`, `namespace`, `source`, `s3_prefix`, `collection`, `submitted_at`.
  
**CDC producer POST body:** none.  
**CDC producer response (202):** `run_id`, `status`, `namespace`, `submitted_at`.

## Error handling

| Exception              | HTTP |
| ---------------------- | ---- |
| `BadRequestError`      | 400  |
| `UpstreamError`        | 502  |
| `PipelineError` (base) | 500  |

## OpenAPI

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **Schema:** `/openapi.json`

## Dependencies

Routes use `config.get_settings()`, `core.services` (RayService, DatabricksRayService, ImageSearchService), and clients (CLIP, vector DB) created from config. Request/response models live in `api.models`.
