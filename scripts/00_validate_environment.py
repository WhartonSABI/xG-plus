#!/usr/bin/env python3
"""Validate local dependencies and expected project paths for the xG+ pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path


REQUIRED_MODULES = ["numpy", "pandas", "sklearn", "joblib"]
OPTIONAL_MODULES = ["boto3", "tqdm", "PIL", "matplotlib"]
XGBOOST_MODULE = "xgboost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local xG+ pipeline environment.")
    parser.add_argument("--tracking-root", type=Path, default=Path("pff-tracking"))
    parser.add_argument("--event-root", type=Path, default=Path("pff-events"))
    parser.add_argument("--require-xgboost", action="store_true")
    parser.add_argument("--skip-secret-scan", action="store_true")
    return parser.parse_args()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def scan_for_literal_aws_keys(paths: list[Path]) -> list[Path]:
    pattern = re.compile(r"AKIA[0-9A-Z]{16}")
    hits: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ipynb"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern.search(text):
                hits.append(path)
    return hits


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    print(f"Python: {sys.version.split()[0]}")
    for module in REQUIRED_MODULES:
        if module_available(module):
            print(f"ok: import {module}")
        else:
            failures.append(f"Missing required Python module: {module}")

    if module_available(XGBOOST_MODULE):
        print(f"ok: import {XGBOOST_MODULE}")
    elif args.require_xgboost:
        failures.append("Missing required Python module: xgboost")
    else:
        print("warn: xgboost is not installed; training/prediction steps will fail until it is installed")

    for module in OPTIONAL_MODULES:
        status = "ok" if module_available(module) else "warn"
        print(f"{status}: import {module}")

    for env_name in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_PROFILE"]:
        if os.environ.get(env_name):
            print(f"ok: {env_name} is set")

    for root in [args.tracking_root, args.event_root]:
        if root.exists():
            print(f"ok: found {root}")
        else:
            print(f"warn: missing {root}")

    if not args.skip_secret_scan:
        hits = scan_for_literal_aws_keys([Path("archived/code"), Path("scripts")])
        if hits:
            print("warn: literal AWS access key ids found in archived/local files:")
            for path in hits[:20]:
                print(f"  {path}")
            if len(hits) > 20:
                print(f"  ... {len(hits) - 20} more")

    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
