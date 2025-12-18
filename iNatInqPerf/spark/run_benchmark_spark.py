#!/usr/bin/env python
"""CLI entrypoint for running the Spark-based benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spark.benchmark_spark import SparkBenchmarker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Spark-based benchmark")
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Base path for dataset and artifact storage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SparkBenchmarker(args.config_file, base_path=args.base_path) as benchmarker:
        benchmarker.run()


if __name__ == "__main__":
    main()
