#!/usr/bin/env python3
"""Validate configuration files against JSON Schema.

Usage:
    python configs/validate.py
    python configs/validate.py --config configs/environments/prod.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from jsonschema import ValidationError, validate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate iNatInq configuration files")
    parser.add_argument(
        "--config",
        type=Path,
        help="Specific config file to validate (default: all configs)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).parent / "schemas" / "config.schema.json",
        help="Path to JSON Schema file",
    )
    args = parser.parse_args()

    # Load schema
    if not args.schema.exists():
        print(f"❌ Schema not found: {args.schema}")
        return 1

    schema = json.loads(args.schema.read_text())

    # Determine which configs to validate
    configs_dir = Path(__file__).parent
    if args.config:
        config_files = [args.config]
    else:
        config_files = list(configs_dir.glob("*.yaml")) + list((configs_dir / "environments").glob("*.yaml"))

    errors = 0
    for cfg_path in config_files:
        # Skip secrets files
        if "secrets" in cfg_path.name and "example" not in cfg_path.name:
            print(f"⏭️  {cfg_path} (skipped - secrets file)")
            continue

        try:
            content = cfg_path.read_text()
            config = yaml.safe_load(content)

            if config is None:
                print(f"⚠️  {cfg_path} (empty file)")
                continue

            validate(config, schema)
            print(f"✅ {cfg_path}")

        except ValidationError as e:
            print(f"❌ {cfg_path}")
            print(f"   Error: {e.message}")
            if e.absolute_path:
                print(f"   Path: {'.'.join(str(p) for p in e.absolute_path)}")
            errors += 1

        except yaml.YAMLError as e:
            print(f"❌ {cfg_path}")
            print(f"   YAML Error: {e}")
            errors += 1

        except Exception as e:
            print(f"❌ {cfg_path}")
            print(f"   Error: {e}")
            errors += 1

    print()
    if errors:
        print(f"❌ {errors} validation error(s)")
        return 1

    print("✅ All configurations valid!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
