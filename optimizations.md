# Image Ingestion Pipeline -- Remaining Optimizations

Findings from the Feb 20 2026 Databricks run (`run_ingest_image.py`, 100k-image config).
Issues 1, 5, and 7 from the original analysis have been fixed in code. The items
below require infrastructure or operational changes.

---

## 1. Pre-create the Qdrant collection before ingestion

**Problem**: `QDRANT_DISABLE_INDEXING_DURING_INGEST=true` is set, but the
`disable_indexing` call fails with a 404 because the target collection does not
exist yet. The pipeline catches the error and continues, so every upsert runs
with indexing **enabled** -- defeating the purpose of deferred indexing.

**Impact**: Each upsert triggers incremental re-indexing inside Qdrant, adding
significant write latency at scale.

**Steps**:

1. Before launching the Databricks job, ensure the collection exists:

   ```bash
   # Using the Qdrant REST API
   curl -X PUT "$QDRANT_URL/collections/$COLLECTION" \
     -H "api-key: $QDRANT_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "vectors": { "size": 512, "distance": "Cosine" }
     }'
   ```

2. Alternatively, add a `create_collection_if_missing()` step at the top of
   `process_s3_images.main()`, before the `disable_indexing` call.

3. After ingestion completes the `finally` block already calls
   `enable_indexing`, which will trigger a full index rebuild.

---

## 2. Scale the Databricks cluster to match `RAY_NUM_WORKERS`

**Problem**: The job requested `RAY_NUM_WORKERS=32` but the Spark cluster only
had enough executor slots for **2 Ray workers**. The pipeline ran at roughly
1/16th of intended parallelism.

**Impact**: Throughput was ~10 images/sec instead of the ~160 images/sec
achievable with 32 workers.

**Steps**:

1. In the Databricks cluster configuration, increase the number of worker nodes
   (or use autoscaling with a high max) so that at least 32 Spark executor
   slots are available.
2. Verify after startup with `ray.cluster_resources()` that the expected CPU
   count appears.
3. If the cluster cannot support 32 workers, lower `RAY_NUM_WORKERS` to match
   reality so that `max_inflight_batches` is sized correctly.

---

## 3. Use larger instance types to meet the 10 GB-per-worker memory threshold

**Problem**: Ray warned that each worker starts with ~6.5 GB heap, below the
recommended 10 GB minimum. Image pipelines download, decode, and embed images
in memory, so memory pressure can cause spilling and GC pauses.

**Steps**:

1. Switch to memory-optimized VM sizes (e.g., Azure `Standard_E8s_v3` or
   `Standard_E16s_v3`) for Spark executor nodes.
2. Alternatively, reduce `RAY_NUM_WORKERS` or increase
   `RAY_WORKER_CPUS` so that fewer Ray workers share each executor, giving
   each worker a larger memory share.
3. Monitor the Ray dashboard's object store usage during ingestion to confirm
   spilling is not occurring.

---

## 4. Increase `IMAGE_PAGE_SIZE` for faster S3 listing

**Problem**: `IMAGE_PAGE_SIZE=1000` means the pipeline makes one S3 `ListObjects`
call per 1000 keys. For 100k images this is 100 sequential API calls, each
adding latency between batch submissions.

**Steps**:

1. Set `IMAGE_PAGE_SIZE=5000` (or up to 10000) in the job environment:

   ```
   IMAGE_PAGE_SIZE=5000
   ```

2. This reduces the number of S3 list calls from 100 to 20, cutting pagination
   overhead significantly.
3. Verify that the S3/MinIO endpoint supports the requested page size (the S3
   API maximum is 1000 per call, but `iter_objects` handles transparent
   re-pagination internally if needed).

---

## 5. Monitor and scale the hosted CLIP endpoint

**Problem**: Throughput showed a bursty pattern -- some 64-image batches
completed in 1-2 seconds while others took 20-25 seconds. This variance
suggests the hosted CLIP endpoint
(`inatclipdeployment-deyxo.eastus.inference.ml.azure.com`) is intermittently
saturated.

**Steps**:

1. Check Azure ML endpoint metrics (request latency P50/P95/P99, queue depth,
   throttling counts) during a run.
2. If latency spikes correlate with high concurrency, scale out the endpoint
   (increase instance count or enable autoscaling).
3. Consider increasing `CLIP_TIMEOUT` beyond 120s if occasional long-tail
   requests are timing out and causing retries.
4. If the endpoint cannot be scaled further, reduce per-task concurrency
   (`RAY_PIPELINE_CONCURRENCY`) to avoid overwhelming it.
