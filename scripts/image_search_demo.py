#!/usr/bin/env python3
"""Image search demo script.

Searches for images using text queries and provides easy access to results.

Usage:
    # Search and display results
    uv run python scripts/image_search_demo.py "red circle"

    # Search with more results
    uv run python scripts/image_search_demo.py "blue square" --limit 10

    # Download results to a folder
    uv run python scripts/image_search_demo.py "green triangle" --download ./results

    # Open in browser (generates presigned URLs)
    uv run python scripts/image_search_demo.py "yellow" --open
"""
# ruff: noqa: T201

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

import boto3
import requests
from botocore.config import Config


def search_images(
    query: str,
    limit: int = 5,
    provider: str = "qdrant",
    api_url: str = "http://localhost:8000",
) -> list[dict]:
    """Search for images using the API."""
    response = requests.get(
        f"{api_url}/search/images",
        params={"q": query, "provider": provider, "limit": limit},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["results"]


def get_presigned_url(
    s3_key: str,
    endpoint: str = "http://localhost:9000",
    bucket: str = "pipeline",
    expires: int = 3600,
) -> str:
    """Generate a presigned URL for an S3 object."""
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",  # noqa: S106
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=expires,
    )


def download_image(
    s3_key: str,
    output_dir: Path,
    endpoint: str = "http://localhost:9000",
    bucket: str = "pipeline",
) -> Path:
    """Download an image from S3."""
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",  # noqa: S106
        region_name="us-east-1",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(s3_key).name
    output_path = output_dir / filename
    s3.download_file(bucket, s3_key, str(output_path))
    return output_path


def display_results(results: list[dict], *, show_urls: bool = False) -> None:
    """Display search results in a formatted table."""
    print()
    print("=" * 80)
    print(f"{'Rank':<6} {'Score':<10} {'Image':<50}")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        s3_key = result["s3_key"]
        score = result["score"]
        filename = Path(s3_key).name
        print(f"{i:<6} {score:<10.4f} {filename:<50}")

        if show_urls:
            url = get_presigned_url(s3_key)
            print(f"       URL: {url}")
            print()

    print("=" * 80)
    print()


def main() -> None:
    """Run the image search demo CLI."""
    parser = argparse.ArgumentParser(
        description="Search for images using text queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="Text query to search for")
    parser.add_argument("--limit", "-n", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--provider",
        "-p",
        choices=["qdrant"],
        default="qdrant",
        help="Vector DB provider (default: qdrant)",
    )
    parser.add_argument(
        "--download",
        "-d",
        type=Path,
        metavar="DIR",
        help="Download images to specified directory",
    )
    parser.add_argument(
        "--open",
        "-o",
        action="store_true",
        help="Open top result in browser",
    )
    parser.add_argument(
        "--urls",
        "-u",
        action="store_true",
        help="Show presigned URLs for each result",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )

    args = parser.parse_args()

    print(f"\nSearching for: '{args.query}'")
    print(f"Provider: {args.provider}")

    try:
        results = search_images(
            query=args.query,
            limit=args.limit,
            provider=args.provider,
            api_url=args.api_url,
        )
    except requests.RequestException as e:
        print(f"Error: Failed to search - {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No results found.")
        sys.exit(0)

    display_results(results, show_urls=args.urls)

    if args.download:
        print(f"Downloading {len(results)} images to {args.download}...")
        for result in results:
            path = download_image(result["s3_key"], args.download)
            print(f"  Downloaded: {path}")
        print()

        # Try to open folder on macOS
        if sys.platform == "darwin":
            subprocess.run(  # noqa: S603
                ["/usr/bin/open", str(args.download)], check=False
            )

    if args.open and results:
        url = get_presigned_url(results[0]["s3_key"])
        print(f"Opening top result in browser: {results[0]['s3_key']}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
