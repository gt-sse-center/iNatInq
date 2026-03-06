"""Unit tests for core.ingestion.databricks.process_inat_images module."""

from unittest.mock import MagicMock, patch

import pytest


class TestIterTaskPayloadBatches:
    """Tests for _iter_task_payload_batches helper."""

    def test_respects_max_items(self) -> None:
        """Batching should stop after max_items records are yielded."""
        from core.ingestion.databricks.process_inat_images import _iter_task_payload_batches

        records = [
            MagicMock(photo_id="1", extension="jpg", photo_url="https://example.com/photos/1/medium.jpg"),
            MagicMock(photo_id="2", extension="png", photo_url="https://example.com/photos/2/medium.png"),
            MagicMock(photo_id="3", extension="jpg", photo_url="https://example.com/photos/3/medium.jpg"),
            MagicMock(photo_id="4", extension="jpg", photo_url="https://example.com/photos/4/medium.jpg"),
        ]

        batches = list(_iter_task_payload_batches(records, image_size="medium", batch_size=2, max_items=2))

        assert len(batches) == 1, "Only 1 batch should be yielded when max_items equals batch_size"
        assert len(batches[0]) == 2, "The single batch should contain exactly 2 items"
        assert batches[0][0]["photo_id"] == "1"
        assert batches[0][1]["photo_id"] == "2"


class TestDatabricksINatImageJobMain:
    """Tests for main() in process_inat_images."""

    @pytest.fixture
    def mock_dependencies(self, mock_ray):
        """Set up common dependency mocks for main() tests."""
        mock_strategy = MagicMock()
        mock_strategy.init = MagicMock()
        mock_strategy.shutdown = MagicMock()

        mock_inat_client = MagicMock()
        mock_inat_client.iter_photo_records.return_value = iter(())
        mock_inat_client.close = MagicMock()

        with (
            patch("core.ingestion.databricks.process_inat_images.INatConfig.from_env") as mock_inat_cfg,
            patch("core.ingestion.databricks.process_inat_images.RayJobConfig.from_env") as mock_ray_cfg,
            patch("core.ingestion.databricks.process_inat_images.EmbeddingConfig.from_env") as mock_embed_cfg,
            patch("core.ingestion.databricks.process_inat_images.VectorDBConfig.from_env") as mock_vector_cfg,
            patch("core.ingestion.databricks.process_inat_images.DatabricksStrategy") as mock_strat_cls,
            patch("core.ingestion.databricks.process_inat_images.INaturalistOpenDataClient") as mock_inat_cls,
            patch(
                "core.ingestion.databricks.process_inat_images.qdrant_indexing_disabled"
            ) as mock_indexing_disabled,
            patch("core.ingestion.databricks.process_inat_images.QdrantClientWrapper") as mock_qdrant_cls,
        ):
            ray_cfg = MagicMock(
                num_workers=4,
                image_batch_size=50,
                image_embed_batch_size=8,
                task_num_cpus=1,
                task_max_retries=3,
                wait_batch_size=10,
                wait_timeout=1.0,
                pipeline_concurrency=10,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=30,
                embedding_timeout=120,
                upsert_timeout=60,
                retry_max_attempts=3,
                retry_min_wait=1.0,
                retry_max_wait=10.0,
                disable_indexing_during_ingest=False,
            )
            inat_cfg = MagicMock(
                image_size="medium",
                max_rows=50,
                metadata_url="",
                photo_base_url="https://inaturalist-open-data.s3.amazonaws.com/photos",
                timeout_s=120,
                cb_failure_threshold=5,
                cb_recovery_timeout_s=30,
                image_max_items=None,
            )
            mock_inat_cfg.return_value = inat_cfg
            mock_ray_cfg.return_value = ray_cfg
            embed_cfg = MagicMock()
            mock_embed_cfg.return_value = embed_cfg
            mock_vector_cfg.return_value = MagicMock(
                collection="documents",
                ingestion_targets=frozenset({"qdrant", "weaviate"}),
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
            )
            mock_strat_cls.from_env.return_value = mock_strategy
            mock_inat_cls.return_value = mock_inat_client

            yield {
                "strategy": mock_strategy,
                "inat_client": mock_inat_client,
                "inat_cls": mock_inat_cls,
                "inat_cfg": inat_cfg,
                "ray_cfg": ray_cfg,
                "embed_cfg": embed_cfg,
                "vector_cfg": mock_vector_cfg.return_value,
                "qdrant_indexing_disabled": mock_indexing_disabled,
                "qdrant_cls": mock_qdrant_cls,
            }

    def test_main_requires_inat_max_rows(self, mock_ray) -> None:
        """main() should fail fast when INAT_MAX_ROWS is missing."""
        from config import INatConfig

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="INAT_MAX_ROWS is required"),
        ):
            INatConfig.from_env()

    def test_main_reads_metadata_and_submits_batches(self, mock_dependencies, mock_ray) -> None:
        """main() reads iNat records and submits batch processing tasks."""
        from core.ingestion.databricks.process_inat_images import main

        mock_dependencies["inat_cfg"].metadata_url = "https://example.com/photos.tsv"
        mock_dependencies["inat_cfg"].max_rows = 50

        record1 = MagicMock(
            photo_id="1", extension="jpg", photo_url="https://example.com/photos/1/medium.jpg"
        )
        record2 = MagicMock(
            photo_id="2", extension="png", photo_url="https://example.com/photos/2/medium.png"
        )
        mock_dependencies["inat_client"].iter_photo_records.return_value = iter([record1, record2])

        future_mock = MagicMock()
        mock_ray.wait.return_value = ([future_mock], [])
        mock_ray.get.return_value = [[("photos/1/medium.jpg", True, ""), ("photos/2/medium.png", True, "")]]

        with patch("core.ingestion.databricks.process_inat_images.process_inat_photo_batch_ray") as mock_task:
            mock_task.options.return_value.remote.return_value = future_mock
            main()

        mock_dependencies["inat_client"].iter_photo_records.assert_called_once_with(
            metadata_url="https://example.com/photos.tsv",
            size="medium",
            max_rows=50,
        )
        mock_dependencies["inat_client"].read_photo_records.assert_not_called()
        mock_task.options.assert_called_once()
        remote_call = mock_task.options.return_value.remote.call_args.kwargs
        assert remote_call["ingestion_targets"] == frozenset({"qdrant", "weaviate"})
        mock_dependencies["strategy"].init.assert_called_once()
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_returns_early_when_no_records(self, mock_dependencies, mock_ray) -> None:
        """main() should return gracefully when metadata yields zero records."""
        from core.ingestion.databricks.process_inat_images import main

        mock_dependencies["inat_cfg"].metadata_url = "https://example.com/photos.tsv"
        mock_dependencies["inat_cfg"].max_rows = 25
        mock_dependencies["inat_client"].iter_photo_records.return_value = iter(())

        main()

        mock_dependencies["strategy"].shutdown.assert_called_once()
        mock_ray.wait.assert_not_called()

    def test_main_defaults_metadata_url_from_client(self, mock_dependencies, mock_ray) -> None:
        """main() should use client default metadata URL when config has empty string."""
        from core.ingestion.databricks.process_inat_images import main

        mock_dependencies["inat_cfg"].metadata_url = ""
        mock_dependencies["inat_cfg"].max_rows = 15
        mock_dependencies[
            "inat_client"
        ].build_metadata_s3_uri.return_value = "s3://inaturalist-open-data/photos.csv.gz"
        mock_dependencies["inat_client"].iter_photo_records.return_value = iter(())

        main()

        mock_dependencies["inat_client"].build_metadata_s3_uri.assert_called_once_with(
            dataset="photos",
            compressed=True,
        )
        mock_dependencies["inat_client"].iter_photo_records.assert_called_once_with(
            metadata_url="s3://inaturalist-open-data/photos.csv.gz",
            size="medium",
            max_rows=15,
        )

    def test_main_applies_python_params(self, mock_dependencies, mock_ray) -> None:
        """main() should apply sys.argv KEY=VALUE params before config parsing."""
        from core.ingestion.databricks.process_inat_images import main

        mock_dependencies["inat_cfg"].metadata_url = "https://example.com/photos.tsv"
        mock_dependencies["inat_cfg"].max_rows = 10

        with (
            patch("sys.argv", ["script.py", "INAT_MAX_ROWS=10"]),
            patch.dict("os.environ", {}, clear=False),
        ):
            main()

        mock_dependencies["inat_client"].iter_photo_records.assert_called_once_with(
            metadata_url="https://example.com/photos.tsv",
            size="medium",
            max_rows=10,
        )

    def test_main_streams_with_bounded_inflight_batches(self, mock_dependencies, mock_ray) -> None:
        """main() should throttle submitted futures to configured worker count."""
        from core.ingestion.databricks.process_inat_images import main

        mock_dependencies["ray_cfg"].num_workers = 1
        mock_dependencies["ray_cfg"].image_batch_size = 1
        mock_dependencies["inat_cfg"].metadata_url = "https://example.com/photos.tsv"
        mock_dependencies["inat_cfg"].max_rows = 3

        records = [
            MagicMock(photo_id="1", extension="jpg", photo_url="https://example.com/photos/1/medium.jpg"),
            MagicMock(photo_id="2", extension="jpg", photo_url="https://example.com/photos/2/medium.jpg"),
            MagicMock(photo_id="3", extension="jpg", photo_url="https://example.com/photos/3/medium.jpg"),
        ]
        mock_dependencies["inat_client"].iter_photo_records.return_value = iter(records)

        futures = [MagicMock(name="f1"), MagicMock(name="f2"), MagicMock(name="f3")]

        def wait_side_effect(current_futures, **_kwargs):
            return [current_futures[0]], current_futures[1:]

        mock_ray.wait.side_effect = wait_side_effect
        mock_ray.get.side_effect = [
            [[("photos/1/medium.jpg", True, "")]],
            [[("photos/2/medium.jpg", True, "")]],
            [[("photos/3/medium.jpg", True, "")]],
        ]

        with patch("core.ingestion.databricks.process_inat_images.process_inat_photo_batch_ray") as mock_task:
            mock_task.options.return_value.remote.side_effect = futures
            main()

        assert mock_task.options.return_value.remote.call_count == 3
        assert mock_ray.wait.call_count >= 3

    def test_main_disables_and_reenables_qdrant_indexing(self, mock_dependencies, mock_ray) -> None:
        """main() uses qdrant_indexing_disabled context manager when flag is set.

        **Why this test is important:**

        - Disabling HNSW indexing during bulk uploads improves throughput
        - Indexing must be re-enabled after processing completes
        - Both operations must use the correct collection name and credentials

        **What it tests:**

        - qdrant_indexing_disabled context manager is entered with correct args
        - Context manager is only used when disable_indexing_during_ingest is True
        """
        from core.ingestion.databricks.process_inat_images import main

        mock_dependencies["ray_cfg"].disable_indexing_during_ingest = True
        mock_dependencies["inat_cfg"].metadata_url = "https://example.com/photos.tsv"
        mock_dependencies["inat_cfg"].max_rows = 10
        mock_dependencies["inat_client"].iter_photo_records.return_value = iter(())

        main()

        mock_dependencies["qdrant_indexing_disabled"].assert_called_once_with(
            client=mock_dependencies["qdrant_cls"].from_config.return_value,
            collection="documents",
            vector_size=mock_dependencies["embed_cfg"].clip_vector_size,
        )
