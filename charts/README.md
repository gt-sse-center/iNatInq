# Architecture Charts

Mermaid diagrams documenting the Query Engine, Ingestion Engine, and CDC architectures.

## Diagrams

| Diagram | Mermaid Source | Description |
| ------- | -------------- | ----------- |
| Query Flow | [query-engine-flowchart.md](./query-engine-flowchart.md) | Flowchart of search request through semantic cache, embedding, Qdrant |
| Query Sequence | [query-engine-sequence.md](./query-engine-sequence.md) | Sequence diagram for image search with cache hit/miss |
| Ingestion Flow | [ingestion-engine-flowchart.md](./ingestion-engine-flowchart.md) | Flowchart of S3 → Ray/Databricks → Qdrant ingestion |
| Ingestion Sequence | [ingestion-engine-sequence.md](./ingestion-engine-sequence.md) | Sequence diagram for job submission and batch processing |
| CDC Architecture | [cdc_architecture.md](./cdc_architecture.md) | Change Data Capture pipeline architecture |
| Resilience Flow | [resilience-flow.md](./resilience-flow.md) | Circuit breaker, retry, DLQ, and checkpoint layers |
| Infrastructure | [infrastructure-deployment.md](./infrastructure-deployment.md) | Docker Compose services, ports, and data flows |
| Provider Abstraction | [provider-abstraction.md](./provider-abstraction.md) | ABC pattern for embedding and vector DB providers |
| Config Layering | [config-layering.md](./config-layering.md) | YAML + env var merge order into Pydantic settings |
| DLQ Recovery | [dlq-recovery-flow.md](./dlq-recovery-flow.md) | Dead Letter Queue capture, storage, and reprocessing |
| Middleware Stack | [middleware-stack.md](./middleware-stack.md) | ASGI middleware pipeline and exception mapping |
| Strategy Pattern | [strategy-pattern.md](./strategy-pattern.md) | ClusterStrategy protocol for ingestion pipeline |

## Rendering

Mermaid diagrams render natively on GitHub. To render locally:

- **VS Code**: Install the "Markdown Preview Mermaid Support" extension
- **CLI**: `npx @mermaid-js/mermaid-cli mmdc -i chart.md -o chart.png`
- **Web**: Paste into [mermaid.live](https://mermaid.live)

## Editing Diagrams

1. Edit the `.md` source files (Mermaid syntax)
2. Verify rendering on GitHub or mermaid.live
3. Diagrams are source-of-truth; PNGs are not checked in
