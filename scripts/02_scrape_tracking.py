#!/usr/bin/env python3
"""Mirror PFF tracking files from the shot-probability S3 bucket.

Credentials are read from the standard AWS environment variables:
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION.
"""

from __future__ import annotations

import argparse
import bz2
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


def list_keys(client, bucket: str, prefixes: list[str]) -> list[dict]:
    objects: list[dict] = []
    for prefix in prefixes:
        token = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = client.list_objects_v2(**kwargs)
            objects.extend(
                obj
                for obj in response.get("Contents", [])
                if not obj["Key"].endswith("/") and not Path(obj["Key"]).name.startswith(".")
            )
            if not response.get("IsTruncated"):
                break
            token = response["NextContinuationToken"]
    return objects


def local_path_for_key(key: str, output_root: Path) -> Path:
    prefix = "pff-data/tracking/"
    if not key.startswith(prefix):
        raise ValueError(f"Unexpected key outside {prefix}: {key}")
    relative = key[len(prefix) :]
    return output_root / relative


def validate_bz2(path: Path) -> None:
    with bz2.open(path, "rb") as fh:
        while fh.read(1024 * 1024):
            pass


def load_archived_credentials(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    assignments = dict(
        re.findall(
            r'os\.environ\["(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_REGION)"\]\s*=\s*["\']([^"\']+)["\']',
            text,
        )
    )
    missing = [key for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] if not assignments.get(key)]
    if missing:
        raise SystemExit(f"Missing archived AWS assignments in {path}: {', '.join(missing)}")
    for key, value in assignments.items():
        if value == "REDACTED":
            raise SystemExit(f"Archived AWS assignment for {key} is redacted in {path}")
        os.environ[key] = value


def download_one(client, bucket: str, key: str, output_path: Path, expected_size: int, force: bool) -> str:
    if output_path.exists() and output_path.stat().st_size == expected_size and not force:
        return "skipped"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        client.download_file(bucket, key, str(tmp_path))
        if output_path.suffix == ".bz2":
            validate_bz2(tmp_path)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror PFF tracking from shot-probability S3.")
    parser.add_argument("--bucket", default="shot-probability")
    parser.add_argument("--output-root", type=Path, default=Path("pff-tracking"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--seasons", nargs="+", default=["2022-2023", "2023-2024", "2024-2025"])
    parser.add_argument("--games", nargs="+", default=None, help="Optional game ids to mirror within each season.")
    parser.add_argument(
        "--credentials-from",
        type=Path,
        default=None,
        help="Optional archived Python file containing AWS os.environ assignments.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.credentials_from is not None:
        load_archived_credentials(args.credentials_from)
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("s3", region_name=region)
    if args.games:
        prefixes = [
            f"pff-data/tracking/{args.competition}/{season}/{game}/"
            for season in args.seasons
            for game in args.games
        ]
    else:
        prefixes = [f"pff-data/tracking/{args.competition}/{season}/" for season in args.seasons]
    objects = list_keys(client, args.bucket, prefixes)
    if not objects:
        raise SystemExit(f"No tracking files found for prefixes: {prefixes}")
    total_size = sum(obj["Size"] for obj in objects)
    print(f"Found {len(objects)} files under {', '.join(prefixes)} ({total_size / 1e9:.2f} GB)")

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                client,
                args.bucket,
                obj["Key"],
                local_path_for_key(obj["Key"], args.output_root),
                obj["Size"],
                args.force,
            ): obj["Key"]
            for obj in objects
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mirroring tracking"):
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"Failed {futures[future]}: {exc}")

    print(
        "Done: "
        f"{counts['downloaded']} downloaded, {counts['skipped']} skipped, {counts['failed']} failed"
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
