*INQUIRE Benchmark Results*

Evaluation of our semantic image search pipeline against the INQUIRE benchmark (NeurIPS 2024), a text-to-image retrieval benchmark built on iNaturalist.

*Pipeline Under Test*
```
Embedding model   OpenAI CLIP ViT-B/32 (hosted, Azure ML)
Vector database   Qdrant Cloud (Azure East US)
Collection size   1,000,000 iNaturalist images
Evaluation k      50 (INQUIRE default)
```

*Dataset Preparation*

The INQUIRE benchmark provides human-annotated text-to-image retrieval queries over the full iNaturalist corpus (~3.6M images). Our Qdrant collection indexes 1M of those images, so not all relevant documents exist in our collection.

We generated subset datasets by trimming each query's relevant document list to only those present in our collection, and dropping queries with zero remaining relevant docs.

```
                   Full INQUIRE        Our Subset          Coverage
Val split          50 queries          47 queries          94% queries
                   6,226 relevant      1,443 relevant      23% docs
Test split         200 queries         186 queries         93% queries
                   26,470 relevant     5,782 relevant      22% docs
```

*Results (Val Split, k=50)*

```
Metric           Score     What it means
─────────────────────────────────────────────────────────────────
Precision@50     0.0834    8.3% of top-50 results are relevant
Recall@50        0.1567    15.7% of relevant docs found in top-50
MAP              0.0679    Avg precision across relevant hits
NDCG@50          0.1518    Ranking quality (position-weighted)
MRR              0.3224    First relevant result ≈ rank 3
```

```
Latency    Value
───────────────────
p50        154.5 ms
p95        240.8 ms
p99        316.7 ms
Mean       173.3 ms
QPS        5.77
```

*How these compare to published numbers*

The INQUIRE paper (Vendrow et al., NeurIPS 2024) reports ViT-B/32 over the full corpus:

```
Metric      INQUIRE Paper    Our Result
────────────────────────────────────────
MAP         ~0.07–0.10       0.068
NDCG@50     ~0.15–0.20       0.152
```

Our results fall within the expected range. Two factors roughly cancel out: smaller corpus (1M vs 3.6M, fewer distractors) and trimmed relevant lists (23% of full set, fewer targets).

*Key takeaways*

• *MRR 0.32* — a relevant result typically appears by rank 3
• *Precision@50 of 0.08* — ~4 relevant results in top 50 (27x above random baseline in a 1M collection)
• *Recall@50 of 0.16* — finds ~1 in 6 relevant docs (many queries have 100+ relevant images, so 50 slots can't capture all)
• *MAP 0.07* — relevant results spread across top-50 rather than concentrated at the top, characteristic of ViT-B/32
• *Latency p50 ~155ms* — end-to-end: CLIP embed (Azure ML) + Qdrant search (Azure Cloud), sequential from local client
