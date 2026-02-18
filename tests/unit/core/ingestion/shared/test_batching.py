"""Unit tests for core.ingestion.shared.batching module.

Tests the iter_image_batches generator that streams S3 pages through
filtering, checkpoint-skipping, and batching.

Run with: uv run pytest tests/unit/core/ingestion/shared/test_batching.py -v
"""

from unittest.mock import MagicMock, patch

from core.ingestion.shared.batching import iter_image_batches


class TestIterImageBatches:
    """Test suite for iter_image_batches generator."""

    def _make_s3(self, pages: list[list[str]]) -> MagicMock:
        """Create a mock S3 client whose iter_objects yields *pages*."""
        s3 = MagicMock()
        s3.iter_objects.return_value = iter(pages)
        return s3

    def test_yields_batches_from_single_page(self):
        """Single page with keys smaller than batch_size yields one batch.

        **What it tests:**
          - Keys from a single page are collected and yielded
          - Batch contains exactly the filtered keys
        """
        s3 = self._make_s3([["img1.jpg", "img2.png"]])

        with patch("core.ingestion.shared.batching.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = lambda k: k
            batches = list(
                iter_image_batches(
                    s3=s3,
                    bucket="b",
                    prefix="p/",
                    processed=set(),
                    batch_size=10,
                    max_items=None,
                )
            )

        assert batches == [["img1.jpg", "img2.png"]]

    def test_splits_across_batch_size(self):
        """Keys exceeding batch_size are split into multiple batches.

        **What it tests:**
          - 5 keys with batch_size=2 yields 3 batches (2, 2, 1)
        """
        keys = [f"img{i}.jpg" for i in range(5)]
        s3 = self._make_s3([keys])

        with patch("core.ingestion.shared.batching.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = lambda k: k
            batches = list(
                iter_image_batches(
                    s3=s3,
                    bucket="b",
                    prefix="",
                    processed=set(),
                    batch_size=2,
                    max_items=None,
                )
            )

        assert len(batches) == 3
        assert batches[0] == ["img0.jpg", "img1.jpg"]
        assert batches[1] == ["img2.jpg", "img3.jpg"]
        assert batches[2] == ["img4.jpg"]

    def test_skips_processed_keys(self):
        """Already-processed keys are excluded from batches.

        **What it tests:**
          - Keys in the processed set are skipped
          - Only unprocessed keys appear in output
        """
        s3 = self._make_s3([["a.jpg", "b.jpg", "c.jpg"]])

        with patch("core.ingestion.shared.batching.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = lambda k: k
            batches = list(
                iter_image_batches(
                    s3=s3,
                    bucket="b",
                    prefix="",
                    processed={"a.jpg", "c.jpg"},
                    batch_size=10,
                    max_items=None,
                )
            )

        assert batches == [["b.jpg"]]

    def test_respects_max_items(self):
        """Generator stops after yielding max_items keys.

        **What it tests:**
          - Only the first max_items keys are yielded
          - Subsequent pages are not consumed
        """
        page1 = [f"img{i}.jpg" for i in range(5)]
        page2 = [f"img{i}.jpg" for i in range(5, 10)]
        s3 = self._make_s3([page1, page2])

        with patch("core.ingestion.shared.batching.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = lambda k: k
            batches = list(
                iter_image_batches(
                    s3=s3,
                    bucket="b",
                    prefix="",
                    processed=set(),
                    batch_size=3,
                    max_items=3,
                )
            )

        # First batch of 3 hits max_items limit
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_filters_non_image_keys(self):
        """Non-image keys are removed by ImageContentFetcher.filter_image_keys.

        **What it tests:**
          - filter_image_keys is called per page
          - Only filtered (image) keys enter the buffer
        """
        s3 = self._make_s3([["img.jpg", "doc.pdf", "photo.png"]])

        with patch("core.ingestion.shared.batching.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = ["img.jpg", "photo.png"]
            batches = list(
                iter_image_batches(
                    s3=s3,
                    bucket="b",
                    prefix="",
                    processed=set(),
                    batch_size=10,
                    max_items=None,
                )
            )

        assert batches == [["img.jpg", "photo.png"]]

    def test_empty_pages_yield_nothing(self):
        """No pages or all-empty pages yield no batches.

        **What it tests:**
          - Generator exhausts without yielding when there are no keys
        """
        s3 = self._make_s3([])

        with patch("core.ingestion.shared.batching.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = lambda k: k
            batches = list(
                iter_image_batches(
                    s3=s3,
                    bucket="b",
                    prefix="",
                    processed=set(),
                    batch_size=10,
                    max_items=None,
                )
            )

        assert batches == []

    def test_multiple_pages_accumulate_into_batches(self):
        """Keys from multiple small pages accumulate until batch_size is reached.

        **What it tests:**
          - Pages smaller than batch_size accumulate in the buffer
          - Batch is yielded once buffer reaches batch_size
          - Remaining buffer is yielded at the end
        """
        s3 = self._make_s3([["a.jpg"], ["b.jpg"], ["c.jpg"]])

        with patch("core.ingestion.shared.batching.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = lambda k: k
            batches = list(
                iter_image_batches(
                    s3=s3,
                    bucket="b",
                    prefix="",
                    processed=set(),
                    batch_size=2,
                    max_items=None,
                )
            )

        assert batches == [["a.jpg", "b.jpg"], ["c.jpg"]]

    def test_passes_page_size_to_iter_objects(self):
        """Custom page_size is forwarded to s3.iter_objects.

        **What it tests:**
          - The page_size parameter is passed through to the S3 client
          - iter_objects is called with the correct bucket, prefix, and page_size
        """
        s3 = self._make_s3([["img.jpg"]])

        with patch("core.ingestion.shared.batching.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = lambda k: k
            list(
                iter_image_batches(
                    s3=s3,
                    bucket="b",
                    prefix="p/",
                    processed=set(),
                    batch_size=10,
                    max_items=None,
                    page_size=500,
                )
            )

        s3.iter_objects.assert_called_once_with(bucket="b", prefix="p/", page_size=500)
