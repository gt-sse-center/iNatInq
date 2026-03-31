"""Search command group for semantic search operations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from cli._util import open_browser

if TYPE_CHECKING:
    from config import MinIOConfig

# Type alias for boto3 S3 client (boto3 is untyped)
S3Client = Any

app = typer.Typer(help="Semantic search operations.")


def _search_images(query: str, limit: int, api_url: str) -> list[dict[str, Any]]:
    """Call Pipeline API /search/images and return results.

    Args:
        query: Text search query.
        limit: Maximum number of results.
        api_url: Base URL of Pipeline API.

    Returns:
        List of search result dictionaries.

    Raises:
        requests.RequestException: On API call failure.
    """
    import requests

    response = requests.get(
        f"{api_url}/search/images",
        params={"q": query, "limit": limit},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["results"]


def _make_s3_client(config: MinIOConfig, *, use_s3v4: bool = False) -> S3Client:
    """Create boto3 S3 client from MinIOConfig.

    Args:
        config: MinIOConfig instance with S3 connection details.
        use_s3v4: If True, use s3v4 signature (required for presigned URLs).

    Returns:
        Configured boto3 S3 client.
    """
    import boto3
    from botocore.config import Config

    client_config = Config(s3={"addressing_style": "path"})
    if use_s3v4:
        client_config = Config(signature_version="s3v4", s3={"addressing_style": "path"})

    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name=config.region,
        config=client_config,
    )


def _get_presigned_url(s3_key: str, s3_client: S3Client, config: MinIOConfig) -> str:
    """Generate presigned URL for an S3 object.

    Args:
        s3_key: S3 object key.
        s3_client: Configured boto3 S3 client (must have s3v4 signature).
        config: MinIOConfig instance.

    Returns:
        Presigned URL string.
    """
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": config.bucket, "Key": s3_key},
        ExpiresIn=3600,
    )


def _download_image(s3_key: str, output_dir: Path | str, s3_client: S3Client, config: MinIOConfig) -> Path:
    """Download image from S3 to local directory.

    Args:
        s3_key: S3 object key.
        output_dir: Local directory for downloaded file.
        s3_client: Configured boto3 S3 client.
        config: MinIOConfig instance.

    Returns:
        Path to downloaded file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(s3_key).name
    output_path = output_dir / filename
    s3_client.download_file(config.bucket, s3_key, str(output_path))
    return output_path


def _display_results(
    results: list[dict[str, Any]], *, show_urls: bool, s3_client: S3Client | None, config: MinIOConfig
) -> None:
    """Display search results in formatted table.

    Args:
        results: List of search result dictionaries.
        show_urls: If True, generate and display presigned URLs.
        s3_client: Configured boto3 S3 client (needed if show_urls=True).
        config: MinIOConfig instance (needed if show_urls=True).
    """
    from pathlib import Path

    print()  # noqa: T201
    print("=" * 80)  # noqa: T201
    print(f"{'Rank':<6} {'Score':<10} {'Image':<50}")  # noqa: T201
    print("=" * 80)  # noqa: T201

    for i, result in enumerate(results, 1):
        s3_key = result["s3_key"]
        score = result["score"]
        filename = Path(s3_key).name
        print(f"{i:<6} {score:<10.4f} {filename:<50}")  # noqa: T201

        if show_urls:
            url = _get_presigned_url(s3_key, s3_client, config)
            print(f"       URL: {url}")  # noqa: T201
            print()  # noqa: T201

    print("=" * 80)  # noqa: T201
    print()  # noqa: T201


@app.command()
def images(
    query: Annotated[str, typer.Option("--query", "-q", help="Search query.")] = "What is the story about?",
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of results.")] = 3,
    api_url: Annotated[str, typer.Option("--api-url", help="API base URL.")] = "http://localhost:8000",
) -> None:
    """Search for images using semantic search."""
    typer.echo(f"Searching images in Qdrant for: '{query}'")

    import requests

    try:
        results = _search_images(query, limit, api_url)
        typer.echo(f"Results: {len(results)}")
        for result in results[:limit]:
            s3_key = result.get("s3_key", "N/A")
            score = result.get("score", 0.0)
            typer.echo(f"  - {s3_key} (score: {score:.3f})")
    except requests.RequestException as e:
        typer.echo(f"Search failed: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def demo(
    query: Annotated[str, typer.Option("--query", "-q", help="Search query.")] = "red circle",
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of results.")] = 5,
    urls: Annotated[bool, typer.Option("--urls", help="Show presigned URLs.")] = True,
    api_url: Annotated[str, typer.Option("--api-url", help="API base URL.")] = "http://localhost:8000",
) -> None:
    """Run image search demo with presigned URLs."""
    import requests
    from botocore.exceptions import ClientError, EndpointConnectionError
    from config import MinIOConfig

    try:
        typer.echo(f"\nSearching for: '{query}'")
        results = _search_images(query, limit, api_url)

        if not results:
            typer.echo("No results found.")
            return

        config = MinIOConfig.from_env()
        s3_client = _make_s3_client(config, use_s3v4=True) if urls else None

        _display_results(results, show_urls=urls, s3_client=s3_client, config=config)

    except requests.RequestException as e:
        typer.echo(f"Search failed: {e}", err=True)
        raise typer.Exit(code=1) from e
    except (ClientError, EndpointConnectionError) as e:
        typer.echo(f"S3 error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def download(
    query: Annotated[str, typer.Option("--query", "-q", help="Search query.")] = "red circle",
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of results.")] = 5,
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory.")] = "./image-results",
    api_url: Annotated[str, typer.Option("--api-url", help="API base URL.")] = "http://localhost:8000",
) -> None:
    """Search and download images to local directory."""
    import requests
    import subprocess
    from botocore.exceptions import ClientError, EndpointConnectionError
    from config import MinIOConfig

    try:
        typer.echo(f"\nSearching for: '{query}'")
        results = _search_images(query, limit, api_url)

        if not results:
            typer.echo("No results found.")
            return

        config = MinIOConfig.from_env()
        s3_client = _make_s3_client(config)

        output_dir = Path(output)
        typer.echo(f"Downloading {len(results)} images to {output_dir}...")

        for result in results:
            path = _download_image(result["s3_key"], output_dir, s3_client, config)
            typer.echo(f"  Downloaded: {path}")

        typer.echo()

        # Try to open folder on macOS
        if sys.platform == "darwin":
            subprocess.run(["/usr/bin/open", str(output_dir)], check=False)

    except requests.RequestException as e:
        typer.echo(f"Search failed: {e}", err=True)
        raise typer.Exit(code=1) from e
    except (ClientError, EndpointConnectionError) as e:
        typer.echo(f"S3 error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command(name="open")
def open_image(
    query: Annotated[str, typer.Option("--query", "-q", help="Search query.")] = "red circle",
    api_url: Annotated[str, typer.Option("--api-url", help="API base URL.")] = "http://localhost:8000",
) -> None:
    """Search and open the top result in a browser."""
    import requests
    from botocore.exceptions import ClientError, EndpointConnectionError
    from config import MinIOConfig

    try:
        results = _search_images(query, 1, api_url)

        if not results:
            typer.echo("No results found.")
            return

        config = MinIOConfig.from_env()
        s3_client = _make_s3_client(config, use_s3v4=True)

        url = _get_presigned_url(results[0]["s3_key"], s3_client, config)
        typer.echo(f"Opening top result in browser: {results[0]['s3_key']}")
        open_browser(url)

    except requests.RequestException as e:
        typer.echo(f"Search failed: {e}", err=True)
        raise typer.Exit(code=1) from e
    except (ClientError, EndpointConnectionError) as e:
        typer.echo(f"S3 error: {e}", err=True)
        raise typer.Exit(code=1) from e
