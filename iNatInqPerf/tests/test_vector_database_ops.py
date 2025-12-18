"""Tests for the vector_database_ops CLI."""

from importlib import util
from pathlib import Path

from typer.testing import CliRunner


def _load_vector_database_ops():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "vector_database_ops.py"
    spec = util.spec_from_file_location("vector_database_ops", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load vector_database_ops module")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_search_parallel_cli_invokes_benchmarker(monkeypatch, tmp_path):
    """Ensure the search_parallel command wires through to Benchmarker.search_parallel."""
    calls: dict[str, object] = {}

    vector_database_ops = _load_vector_database_ops()

    class DummyBenchmarker:
        def __init__(self, config_file, base_path):
            calls["init"] = (config_file, base_path)

        def get_vector_db(self):
            calls["get_vector_db"] = True
            return "db"

        def search_parallel(self, db, baseline_results_path=None, processes=None):
            calls["search_parallel"] = (db, baseline_results_path, processes)

    monkeypatch.setattr(vector_database_ops, "Benchmarker", DummyBenchmarker)

    config = tmp_path / "config.yaml"
    config.write_text("dummy")

    result = CliRunner().invoke(
        vector_database_ops.app,
        ["search-parallel", str(config), "--processes", "3"],
    )

    assert result.exit_code == 0
    expected_base = Path(vector_database_ops.__file__).parent.parent
    assert calls["init"] == (config, expected_base)
    assert calls["search_parallel"] == ("db", None, 3)
