#!/usr/bin/env python3
"""Synthetic data generation and upload for pipeline testing.

This module provides tools to:
1. Generate test documents from public domain books (e.g., Moby Dick)
2. Upload generated documents to MinIO/S3

Usage:
    # Generate documents
    uv run python synthetic_data.py generate --count 1000 --chunk-size 500

    # Upload to MinIO
    uv run python synthetic_data.py upload --endpoint http://localhost:9000

    # Generate and upload in one step
    uv run python synthetic_data.py setup --count 1000
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from typing import Any

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:
    print("❌ boto3 not installed. Install with: uv add boto3", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:
    print("❌ tqdm not installed. Install with: uv add tqdm", file=sys.stderr)
    sys.exit(1)

# Import retry logic from foundation
try:
    from foundation.retry import RetryWithBackoff
except ImportError:
    # Fallback if running outside package context
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    from foundation.retry import RetryWithBackoff


# =============================================================================
# Document Generator
# =============================================================================


@attrs.define
class DocumentGenerator:
    """Generates test documents from a source text file.

    Splits source text into chunks and saves them as individual files.
    Useful for creating test data for ML pipelines.

    Note:
        This implementation is tailored for Project Gutenberg texts (e.g., Moby Dick)
        and includes logic to strip Gutenberg-specific headers/footers. This is
        intended for demo purposes; production use will likely require different
        data sources (e.g., iNaturalist images).

    Attributes:
        source_file: Path to source text file (e.g., Moby Dick).
        output_dir: Directory to write generated documents.
        chunk_size: Target size for each chunk in characters.
    """

    source_file: Path = attrs.field(converter=Path)
    output_dir: Path = attrs.field(converter=Path)
    chunk_size: int = attrs.field(default=500)

    def _clean_text(self, text: str) -> str:
        """Clean Project Gutenberg text by removing headers and footers.

        Args:
            text: Raw text from Project Gutenberg.

        Returns:
            Cleaned text with headers/footers removed.
        """
        # Remove Project Gutenberg header (everything before "CHAPTER 1")
        match = re.search(r"(?i)(?:^|\n)(?:chapter\s+1|chapter\s+i)\b", text)
        if match:
            text = text[match.start() :]

        # Remove Project Gutenberg footer
        match = re.search(r"(?i)\n\s*end\s+of\s+(?:the\s+)?project\s+gutenberg", text)
        if match:
            text = text[: match.start()]

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks of approximately chunk_size characters.

        Tries to break at sentence boundaries when possible.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.

        Note:
            TODO: Consider using nltk.sent_tokenize() for more robust sentence
            boundary detection. Skipped for now since we're moving to image data.
        """
        chunks = []
        current_pos = 0
        text_len = len(text)

        while current_pos < text_len:
            end_pos = min(current_pos + self.chunk_size, text_len)

            # Try to break at a sentence boundary
            if end_pos < text_len:
                search_start = max(current_pos, end_pos - self.chunk_size // 5)
                match = re.search(r"[.!?]\s+[A-Z]|[.!?]\n", text[search_start:end_pos])
                if match:
                    end_pos = search_start + match.end()

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            current_pos = end_pos

        return chunks

    def generate(self, count: int) -> int:
        """Generate test documents from source text.

        Args:
            count: Number of documents to generate.

        Returns:
            Number of documents generated.

        Raises:
            FileNotFoundError: If source file does not exist.
        """
        if not self.source_file.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_file}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Reading source file: {self.source_file}")
        with open(self.source_file, encoding="utf-8") as f:
            raw_text = f.read()

        print("Cleaning text...")
        cleaned_text = self._clean_text(raw_text)

        print(f"Source text length: {len(cleaned_text):,} characters")
        print(f"Generating {count} documents with ~{self.chunk_size} characters each...")

        # Split into chunks
        chunks = self._split_into_chunks(cleaned_text)

        # Generate documents (cycle through chunks if needed)
        documents_generated = 0
        chunk_index = 0

        while documents_generated < count:
            chunk = chunks[chunk_index % len(chunks)]
            output_file = self.output_dir / f"moby-dick-{documents_generated:05d}.txt"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(chunk)

            documents_generated += 1
            chunk_index += 1

            if documents_generated % 100 == 0:
                print(f"  Generated {documents_generated}/{count} documents...")

        print(f"✅ Generated {documents_generated} documents in {self.output_dir}")
        return documents_generated


# =============================================================================
# MinIO Uploader
# =============================================================================


@attrs.define
class MinIOUploader:
    """Uploads files to MinIO/S3-compatible storage.

    Uses concurrent uploads with ThreadPoolExecutor for better performance.
    Includes retry logic and progress tracking.

    Attributes:
        endpoint: MinIO endpoint URL.
        access_key: MinIO access key.
        secret_key: MinIO secret key.
        bucket: Target bucket name.
        max_workers: Maximum concurrent upload threads.
        max_retries: Maximum retry attempts per file.
    """

    endpoint: str = attrs.field(default="http://localhost:9000")
    access_key: str = attrs.field(default="minioadmin")
    secret_key: str = attrs.field(default="minioadmin")
    bucket: str = attrs.field(default="pipeline")
    max_workers: int = attrs.field(default=50)
    max_retries: int = attrs.field(default=3)

    _client: Any = attrs.field(init=False, default=None)
    _retry: RetryWithBackoff = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        """Initialize S3 client and retry utility."""
        s3_config = Config(max_pool_connections=self.max_workers * 2)
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name="us-east-1",
            config=s3_config,
        )
        self._retry = RetryWithBackoff(
            max_attempts=self.max_retries,
            wait_min=0.1,
            wait_max=2.0,
            multiplier=1.0,
        )

    def _upload_single_file(self, file_path: Path, s3_key: str) -> tuple[str, bool]:
        """Upload a single file with retry logic.

        Args:
            file_path: Path to file to upload.
            s3_key: S3 key (path) for the object.

        Returns:
            Tuple of (filename, success).
        """
        filename = file_path.name

        def _upload() -> None:
            self._client.upload_file(str(file_path), self.bucket, s3_key)

        try:
            self._retry.call(_upload, retry_exceptions=(ClientError, OSError))
            return (filename, True)
        except (ClientError, OSError):
            return (filename, False)

    def upload_directory(
        self,
        input_dir: Path,
        prefix: str = "inputs/",
        pattern: str = "*.txt",
    ) -> tuple[int, int]:
        """Upload all matching files from a directory to MinIO.

        Args:
            input_dir: Directory containing files to upload.
            prefix: S3 prefix (directory) to upload to.
            pattern: Glob pattern for files to upload.

        Returns:
            Tuple of (successful_count, failed_count).

        Raises:
            FileNotFoundError: If input directory does not exist.
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        files = sorted(input_dir.glob(pattern))
        if not files:
            print(f"❌ No {pattern} files found in {input_dir}")
            return (0, 0)

        total_files = len(files)
        print(f"Found {total_files} files to upload...")
        print(f"Uploading to s3://{self.bucket}/{prefix}")
        print(f"Using {self.max_workers} concurrent workers...")

        progress_bar = tqdm(
            total=total_files,
            desc="Uploading",
            unit="files",
            mininterval=1.0,
            maxinterval=10.0,
            file=sys.stdout,
            ncols=100,
        )

        successful = 0
        failed = 0
        start_time = time.time()

        def upload_with_progress(file_path: Path) -> tuple[str, bool]:
            s3_key = f"{prefix.rstrip('/')}/{file_path.name}"
            result = self._upload_single_file(file_path, s3_key)
            progress_bar.update(1)
            return result

        try:
            with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {executor.submit(upload_with_progress, fp): fp for fp in files}

                for future in futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        filename, success = future.result()
                        if success:
                            successful += 1
                        else:
                            failed += 1
                            print(f"  ❌ Failed: {filename}", file=sys.stderr)
                    except (ClientError, OSError, RuntimeError) as e:
                        failed += 1
                        print(f"  ❌ Exception: {file_path.name}: {e}", file=sys.stderr)
        finally:
            progress_bar.close()

        elapsed = time.time() - start_time
        rate = total_files / elapsed if elapsed > 0 else 0

        print(f"\n✅ Upload complete: {successful} successful, {failed} failed ")
        print(f"   ({elapsed:.1f}s, {rate:.1f} files/s)")

        return (successful, failed)


# =============================================================================
# CLI
# =============================================================================


def cmd_generate(args: argparse.Namespace) -> int:
    """Handle 'generate' subcommand."""
    generator = DocumentGenerator(
        source_file=args.source,
        output_dir=args.output,
        chunk_size=args.chunk_size,
    )
    try:
        generator.generate(count=args.count)
        return 0
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cmd_upload(args: argparse.Namespace) -> int:
    """Handle 'upload' subcommand."""
    uploader = MinIOUploader(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=args.bucket,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=args.input_dir,
            prefix=args.prefix,
        )
        return 1 if failed > 0 else 0
    except (FileNotFoundError, ClientError, OSError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cmd_setup(args: argparse.Namespace) -> int:
    """Handle 'setup' subcommand (generate + upload)."""
    # Generate
    generator = DocumentGenerator(
        source_file=args.source,
        output_dir=args.output,
        chunk_size=args.chunk_size,
    )
    try:
        generator.generate(count=args.count)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

    print()  # Blank line between steps

    # Upload
    uploader = MinIOUploader(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=args.bucket,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=args.output,
            prefix=args.prefix,
        )
        return 1 if failed > 0 else 0
    except (FileNotFoundError, ClientError, OSError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """Main CLI entry point."""
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Synthetic data generation and upload for pipeline testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Generate subcommand ---
    gen_parser = subparsers.add_parser("generate", help="Generate test documents")
    gen_parser.add_argument(
        "--source",
        type=Path,
        default=base_dir / "moby-dick.txt",
        help="Source text file (default: moby-dick.txt)",
    )
    gen_parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "inputs",
        help="Output directory (default: inputs/)",
    )
    gen_parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of documents to generate (default: 1000)",
    )
    gen_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    gen_parser.set_defaults(func=cmd_generate)

    # --- Upload subcommand ---
    up_parser = subparsers.add_parser("upload", help="Upload documents to MinIO")
    up_parser.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "inputs",
        help="Directory with files to upload (default: inputs/)",
    )
    up_parser.add_argument(
        "--bucket",
        type=str,
        default="pipeline",
        help="MinIO bucket name (default: pipeline)",
    )
    up_parser.add_argument(
        "--prefix",
        type=str,
        default="inputs/",
        help="S3 prefix/directory (default: inputs/)",
    )
    up_parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:9000",
        help="MinIO endpoint (default: http://localhost:9000)",
    )
    up_parser.add_argument(
        "--access-key",
        type=str,
        default="minioadmin",
        help="MinIO access key (default: minioadmin)",
    )
    up_parser.add_argument(
        "--secret-key",
        type=str,
        default="minioadmin",
        help="MinIO secret key (default: minioadmin)",
    )
    up_parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="Concurrent upload threads (default: 50)",
    )
    up_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per file (default: 3)",
    )
    up_parser.set_defaults(func=cmd_upload)

    # --- Setup subcommand (generate + upload) ---
    setup_parser = subparsers.add_parser("setup", help="Generate and upload (all-in-one)")
    setup_parser.add_argument(
        "--source",
        type=Path,
        default=base_dir / "moby-dick.txt",
        help="Source text file (default: moby-dick.txt)",
    )
    setup_parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "inputs",
        help="Output directory (default: inputs/)",
    )
    setup_parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of documents to generate (default: 1000)",
    )
    setup_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    setup_parser.add_argument(
        "--bucket",
        type=str,
        default="pipeline",
        help="MinIO bucket name (default: pipeline)",
    )
    setup_parser.add_argument(
        "--prefix",
        type=str,
        default="inputs/",
        help="S3 prefix/directory (default: inputs/)",
    )
    setup_parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:9000",
        help="MinIO endpoint (default: http://localhost:9000)",
    )
    setup_parser.add_argument(
        "--access-key",
        type=str,
        default="minioadmin",
        help="MinIO access key (default: minioadmin)",
    )
    setup_parser.add_argument(
        "--secret-key",
        type=str,
        default="minioadmin",
        help="MinIO secret key (default: minioadmin)",
    )
    setup_parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="Concurrent upload threads (default: 50)",
    )
    setup_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per file (default: 3)",
    )
    setup_parser.set_defaults(func=cmd_setup)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
