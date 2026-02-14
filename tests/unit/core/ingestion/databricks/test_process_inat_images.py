"""Unit tests for core.ingestion.databricks.process_inat_images module."""

from unittest.mock import MagicMock, patch

import pytest


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
            patch("core.ingestion.databricks.process_inat_images.RayJobConfig.from_env") as mock_ray_cfg,
            patch(
                "core.ingestion.databricks.process_inat_images.ImageEmbeddingConfig.from_env"
            ) as mock_embed_cfg,
            patch("core.ingestion.databricks.process_inat_images.VectorDBConfig.from_env") as mock_vector_cfg,
            patch("core.ingestion.databricks.process_inat_images.DatabricksStrategy") as mock_strat_cls,
            patch("core.ingestion.databricks.process_inat_images.INaturalistOpenDataClient") as mock_inat_cls,
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
            )
            mock_ray_cfg.return_value = ray_cfg
            mock_embed_cfg.return_value = MagicMock()
            mock_vector_cfg.return_value = MagicMock(ingestion_targets=frozenset({"qdrant", "weaviate"}))
            mock_strat_cls.from_env.return_value = mock_strategy
            mock_inat_cls.return_value = mock_inat_client

            yield {
                "strategy": mock_strategy,
                "inat_client": mock_inat_client,
                "inat_cls": mock_inat_cls,
                "ray_cfg": ray_cfg,
                "vector_cfg": mock_vector_cfg.return_value,
            }

    def test_main_allows_missing_image_max_items(self, mock_dependencies, mock_ray) -> None:
        """main() should process without a cap when IMAGE_MAX_ITEMS is unset."""
        from core.ingestion.databricks.process_inat_images import main

        mock_dependencies[
            "inat_client"
        ].build_metadata_s3_uri.return_value = "s3://inaturalist-open-data/photos.csv.gz"

        with patch.dict("os.environ", {}, clear=True):
            main()

        mock_dependencies["inat_client"].iter_photo_records.assert_called_once_with(
            metadata_url="s3://inaturalist-open-data/photos.csv.gz",
            size="medium",
            max_rows=None,
        )

    def test_main_rejects_invalid_image_max_items(self, mock_dependencies, mock_ray) -> None:
        """main() should fail fast when IMAGE_MAX_ITEMS is invalid."""
        from core.ingestion.databricks.process_inat_images import main

        with (
            patch.dict("os.environ", {"IMAGE_MAX_ITEMS": "0"}, clear=False),
            pytest.raises(RuntimeError, match="IMAGE_MAX_ITEMS must be a positive integer"),
        ):
            main()

    def test_main_reads_metadata_and_submits_batches(self, mock_dependencies, mock_ray) -> None:
        """main() reads iNat records and submits batch processing tasks."""
        from core.ingestion.databricks.process_inat_images import main

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
            with patch.dict(
                "os.environ",
                {"INAT_METADATA_URL": "https://example.com/photos.tsv", "IMAGE_MAX_ITEMS": "50"},
                clear=False,
            ):
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

        mock_dependencies["inat_client"].iter_photo_records.return_value = iter(())

        with patch.dict(
            "os.environ",
            {"INAT_METADATA_URL": "https://example.com/photos.tsv", "IMAGE_MAX_ITEMS": "25"},
            clear=False,
        ):
            main()

        mock_dependencies["strategy"].shutdown.assert_called_once()
        mock_ray.wait.assert_not_called()

    def test_main_defaults_metadata_url_from_client(self, mock_dependencies, mock_ray) -> None:
        """main() should use client default metadata URL when env var is missing."""
        from core.ingestion.databricks.process_inat_images import main

        mock_dependencies[
            "inat_client"
        ].build_metadata_s3_uri.return_value = "s3://inaturalist-open-data/photos.csv.gz"
        mock_dependencies["inat_client"].iter_photo_records.return_value = iter(())

        with patch.dict("os.environ", {"IMAGE_MAX_ITEMS": "15"}, clear=False):
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
        """main() should apply INAT_METADATA_URL from sys.argv KEY=VALUE params."""
        from core.ingestion.databricks.process_inat_images import main

        with (
            patch(
                "sys.argv",
                ["script.py", "INAT_METADATA_URL=https://example.com/photos.tsv", "IMAGE_MAX_ITEMS=10"],
            ),
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
            with patch.dict(
                "os.environ",
                {"INAT_METADATA_URL": "https://example.com/photos.tsv", "IMAGE_MAX_ITEMS": "3"},
                clear=False,
            ):
                main()

        assert mock_task.options.return_value.remote.call_count == 3
        assert mock_ray.wait.call_count >= 3
