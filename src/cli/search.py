"""Search command group for semantic search operations."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import quote_plus

import typer

from cli._util import run

app = typer.Typer(help="Semantic search operations.")


@app.command()
def images(
    query: Annotated[str, typer.Option("--query", "-q", help="Search query.")] = "What is the story about?",
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of results.")] = 3,
) -> None:
    """Search for images using semantic search."""
    typer.echo(f"Searching images in Qdrant for: '{query}'")

    import requests

    try:
        encoded_query = quote_plus(query)
        response = requests.get(
            f"http://localhost:8000/search/images?query={encoded_query}&limit={limit}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

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
) -> None:
    """Run image search demo with presigned URLs."""
    cmd = ["uv", "run", "python", "scripts/image_search_demo.py", query, "--limit", str(limit)]
    if urls:
        cmd.append("--urls")
    run(cmd)


@app.command()
def download(
    query: Annotated[str, typer.Option("--query", "-q", help="Search query.")] = "red circle",
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of results.")] = 5,
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory.")] = "./image-results",
) -> None:
    """Search and download images to local directory."""
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/image_search_demo.py",
            query,
            "--limit",
            str(limit),
            "--download",
            output,
        ]
    )


@app.command(name="open")
def open_image(
    query: Annotated[str, typer.Option("--query", "-q", help="Search query.")] = "red circle",
) -> None:
    """Search and open the top result in a browser."""
    run(["uv", "run", "python", "scripts/image_search_demo.py", query, "--limit", "1", "--open"])
