# Middleware Stack Diagram

Shows the ASGI middleware pipeline order: how requests flow inward through each layer and responses flow back out (onion model).

```mermaid
flowchart TB
    subgraph External["Client Request"]
        A["HTTP Request<br/>GET /search/images?q=bird"]
    end

    subgraph MW1["Layer 1: CORSMiddleware (Starlette)"]
        B1["Preflight: handle OPTIONS<br/>Set Access-Control-* headers"]
    end

    subgraph MW2["Layer 2: LoggerMiddleware"]
        C1["Request: log path, method, remoteAddr<br/>to pipeline.access logger"]
        C2["Response: log statuscode, duration<br/>via send_wrapper"]
    end

    subgraph MW3["Layer 3: HealthzFilterMiddleware"]
        D1{Path == /healthz?}
        D2["Suppress uvicorn.access logger<br/>(setLevel CRITICAL+1)"]
        D3[Pass through]
    end

    subgraph MW4["Layer 4: ExceptionHandlerMiddleware"]
        E1["try: await app(scope, receive, send)"]
        E2["except: map exception → JSONResponse"]
    end

    subgraph App["FastAPI Application"]
        F["Route Handler<br/>(async def with asyncio.to_thread)"]
    end

    subgraph Response["HTTP Response"]
        G["JSON Response<br/>200 / 4xx / 5xx"]
    end

    A --> B1
    B1 --> C1
    C1 --> D1
    D1 -->|Yes| D2 --> F
    D1 -->|No| D3 --> E1
    E1 --> F

    F --> G
    E2 --> G

    F -.->|Exception| E2
    G --> C2
    C2 --> B1
    B1 --> Response

    style A fill:#e1f5fe
    style B1 fill:#fff3e0
    style C1 fill:#e8f5e9
    style C2 fill:#e8f5e9
    style D2 fill:#f3e5f5
    style E1 fill:#ffcdd2
    style E2 fill:#ffcdd2
    style F fill:#c8e6c9
    style G fill:#e1f5fe
```

## Middleware Registration Order

Middleware is registered in `src/api/app.py`. Starlette processes middleware in **reverse registration order** for inbound requests (last registered = closest to the route handler):

| Registration Order | Class | Inbound Behavior | Outbound Behavior |
|---|---|---|---|
| 1st | `CORSMiddleware` | Handle CORS preflight, set headers | Add `Access-Control-*` headers |
| 2nd | `LoggerMiddleware` | Log request start (path, method, IP) | Log response (status, duration) via `send` wrapper |
| 3rd | `HealthzFilterMiddleware` | Suppress `uvicorn.access` for `/healthz` | Restore logger level |
| 4th | `ExceptionHandlerMiddleware` | Wrap inner app in `try/except` | Map exceptions to JSON responses |

## Exception Mapping

The `ExceptionHandlerMiddleware` converts Python exceptions to HTTP status codes:

```mermaid
flowchart LR
    subgraph Exceptions["Exception Type"]
        EX1[BadRequestError]
        EX2[PipelineTimeoutError]
        EX3[UpstreamError]
        EX4[PipelineError]
        EX5[HTTPException]
        EX6["ValueError / TypeError<br/>AttributeError / KeyError"]
        EX7["RuntimeError / OSError"]
        EX8["Exception (catch-all)"]
    end

    subgraph Status["HTTP Status"]
        S400[400 Bad Request]
        S504[504 Gateway Timeout]
        S502[502 Bad Gateway]
        S500[500 Internal Server Error]
        SHTTP["N (from HTTPException)"]
    end

    EX1 --> S400
    EX2 --> S504
    EX3 --> S502
    EX4 --> S500
    EX5 --> SHTTP
    EX6 --> S500
    EX7 --> S500
    EX8 --> S500

    style S400 fill:#fff3e0
    style S504 fill:#fff3e0
    style S502 fill:#ffcdd2
    style S500 fill:#ffcdd2
```

## Key Design Decisions

- **Pure ASGI classes** (not `BaseHTTPMiddleware`) — avoids per-request thread overhead and memory stream buffering
- **`send` wrapper** in `LoggerMiddleware` captures status code from the `http.response.start` ASGI message without buffering the full response
- **Logger suppression** in `HealthzFilterMiddleware` uses `setLevel(CRITICAL+1)` to silence health probe noise at the source, restored in a `finally` block
- **Prometheus metrics** (`/metrics`) are added separately via `prometheus-fastapi-instrumentator`, not as middleware
