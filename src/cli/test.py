"""Test command group for running tests."""

from __future__ import annotations

from typing import Annotated

import typer

from cli._util import run

app = typer.Typer(help="Test commands.")


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def unit(ctx: typer.Context) -> None:
    """Run unit tests.

    Extra arguments after -- are passed to pytest.
    Example: inq test unit -- -k test_name
    """
    typer.echo("Running unit tests...")
    cmd = ["uv", "run", "pytest", "tests/unit/", "-v"]
    if ctx.args:
        cmd.extend(ctx.args)
    run(cmd)


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def integration(
    ctx: typer.Context,
    parallel: Annotated[bool, typer.Option("--parallel", "-p", help="Run tests in parallel.")] = False,
) -> None:
    """Run integration tests (requires Docker).

    Extra arguments after -- are passed to pytest.
    Example: inq test integration --parallel -- -k test_qdrant
    """
    typer.echo("Running integration tests..." + (" in parallel" if parallel else ""))
    cmd = ["uv", "run", "pytest", "tests/integration/", "-v", "-m", "integration"]
    if parallel:
        cmd.extend(["-n", "auto"])
    if ctx.args:
        cmd.extend(ctx.args)
    run(cmd)


@app.command(name="all", context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def all_tests(ctx: typer.Context) -> None:
    """Run all tests (unit + integration).

    Extra arguments after -- are passed to pytest.
    Example: inq test all -- --maxfail=1
    """
    typer.echo("Running all tests (unit + integration)...")
    cmd = ["uv", "run", "pytest", "tests/", "-v"]
    if ctx.args:
        cmd.extend(ctx.args)
    run(cmd)


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def cov(
    ctx: typer.Context,
    all_tests: Annotated[bool, typer.Option("--all", "-a", help="Run all tests (not just unit).")] = False,
) -> None:
    """Run tests with coverage.

    Extra arguments after -- are passed to pytest.
    Example: inq test cov --all -- --cov-report=xml
    """
    typer.echo("Running " + ("all" if all_tests else "unit") + " tests with coverage...")
    test_path = "tests/" if all_tests else "tests/unit/"
    cmd = ["uv", "run", "pytest", test_path, "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]
    if ctx.args:
        cmd.extend(ctx.args)
    run(cmd)


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def e2e(ctx: typer.Context) -> None:
    """Run E2E tests (requires Docker Compose stack running).

    Extra arguments after -- are passed to pytest.
    Example: inq test e2e -- -k test_ingestion

    Note: E2E tests use --no-cov to disable coverage (tests exercise
    running containers, not importable source code).
    """
    typer.echo("Running E2E tests...")
    cmd = ["uv", "run", "pytest", "tests/e2e/", "-v", "-m", "e2e", "--no-header", "--no-cov"]
    if ctx.args:
        cmd.extend(ctx.args)
    run(cmd)
