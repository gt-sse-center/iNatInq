# Integration Tests

Integration tests validate that our client wrappers work correctly against real
infrastructure. Unlike unit tests which use mocks, integration tests spin up
actual services in Docker containers to test real network calls, error handling,
and resilience features.

## Philosophy

> **Unit tests validate logic. Integration tests validate reality. Resilience tests validate survival.**

We assume:

- Networks are unreliable
- Dependencies fail
- Timeouts happen
- Rate limits exist

And prove the system behaves safely under all of them.

## Running Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run S3 client tests only
pytest tests/integration/clients/test_s3.py -v

# Run with logging visible
pytest tests/integration/ -v --log-cli-level=INFO

# Run without coverage (faster)
pytest tests/integration/ -v --no-cov
```

## Test Categories

Following our integration test strategy, each client is tested for:

### 1. Happy Path (Baseline Correctness)

- Valid request returns valid response
- Response is correctly parsed
- No retries triggered
- Circuit breaker remains closed

### 2. Transient Failure → Retry Succeeds

- Connection errors trigger retries
- 5xx errors trigger retries
- Exponential backoff is applied
- Eventually succeeds

### 3. Retry Exhaustion → Proper Failure

- All retry attempts fail
- `UpstreamError` is raised
- Error message is meaningful
- No infinite loops

### 4. Non-Retriable Errors (Fail Fast)

- 4xx errors (400, 401, 403, 404) do not retry
- Authentication failures fail immediately
- Clear error messages

### 5. Circuit Breaker Opens After Threshold

- Repeated failures open the circuit
- Fail-fast behavior once open
- No outbound calls while open

### 6. Circuit Breaker Recovery

- Circuit enters HALF_OPEN after timeout
- Success closes circuit
- Failure reopens circuit

### 7. Rate Limiting

- Requests exceeding rate are delayed/rejected
- Throughput stays within limits

### 8. Timeout Handling

- Slow operations trigger timeouts
- Requests don't hang indefinitely
- Resources cleaned up

### 9. Resource Cleanup

- Connections properly closed
- No leaked file handles
- Concurrent operations managed

### 10. Observability & Logging

- Errors logged with context
- Retry attempts logged
- Structured, actionable logs

## Infrastructure

Tests use [testcontainers-python](https://github.com/testcontainers/testcontainers-python)
to manage Docker containers. Containers are:

- **Session-scoped**: Started once per test run for efficiency
- **Self-contained**: No external docker-compose required
- **Deterministic**: Same behavior locally and in CI

### MinIO Container

Provides S3-compatible object storage:

```python
@pytest.fixture(scope="session")
def minio_container():
    container = MinioContainer(
        image="minio/minio:RELEASE.2024-01-01T16-36-33Z",
        access_key="minioadmin",
        secret_key="minioadmin",
    )
    container.start()
    yield container
    container.stop()
```

## Adding New Integration Tests

1. Create fixture in `tests/integration/clients/conftest.py`
2. Create test file `tests/integration/clients/test_{client}.py`
3. Cover all 10 test categories where applicable
4. Use unique bucket/key names per test to avoid collisions
5. Clean up resources in fixture teardown

## CI Compatibility

These tests require Docker to run. Ensure your CI environment:

- Has Docker installed and running
- Has sufficient memory for containers (2GB minimum)
- Allows network access for container image pulls (first run only)

## Parallel Execution & Port Conflicts

Testcontainers is designed for parallel execution:

| Concern | How It's Handled |
|---------|------------------|
| **Port conflicts** | Random ephemeral host ports (never hardcoded) |
| **Container names** | Unique random suffixes per container |
| **pytest-xdist** | Each worker creates independent containers |
| **Orphan cleanup** | Ryuk sidecar auto-removes crashed containers |

### Running with pytest-xdist

```bash
# Parallel execution (each worker gets its own containers)
pytest tests/integration/ -n auto

# Limit workers if memory-constrained
pytest tests/integration/ -n 2
```

**Note**: With session-scoped fixtures, each xdist worker creates its own
container. This uses more resources but ensures complete isolation.

## Troubleshooting

### Container startup timeout

Increase the wait timeout in `_wait_for_*_health()` functions.

### Port conflicts

Testcontainers automatically assigns random ports. If issues persist:

- Stop local services on ports 9000, 5432, 6379, 6333, 8080
- Check for orphan Docker containers: `docker ps`

### Orphan containers

Testcontainers uses Ryuk to clean up orphans automatically. If containers
remain after a crash:

```bash
# Manual cleanup
docker ps --filter "name=testcontainers" -q | xargs -r docker rm -f
```

To disable Ryuk (not recommended):

```bash
export TESTCONTAINERS_RYUK_DISABLED=true
```

### Image pull failures

Ensure Docker has network access. Images are cached after first pull.
