#!/usr/bin/env python3
"""List match numbers present in the PFF S3 tracking prefix."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError as exc:
    raise SystemExit(
        "Missing boto3. Run this with the project venv, for example:\n"
        "  .venv/bin/python scripts/check_s3_match_numbers.py "
        "--credentials-from archived/local/sagemaker/features.py"
    ) from exc


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


def list_game_dirs(client, bucket: str, prefix: str) -> list[str]:
    games: list[str] = []
    token = None
    while True:
        kwargs = {
            "Bucket": bucket,
            "Prefix": prefix,
            "Delimiter": "/",
        }
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("CommonPrefixes", []):
            game = item["Prefix"][len(prefix) :].strip("/")
            if game:
                games.append(game)
        if not response.get("IsTruncated"):
            break
        token = response["NextContinuationToken"]
    return sorted(set(games), key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List PFF match numbers in S3 tracking prefixes.")
    parser.add_argument("--bucket", default="shot-probability")
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--seasons", nargs="+", default=["2022-2023", "2023-2024", "2024-2025"])
    parser.add_argument("--target-game", default="13472")
    parser.add_argument(
        "--credentials-from",
        type=Path,
        default=None,
        help="Optional archived Python file containing AWS os.environ assignments.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/s3_match_numbers_pl.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.credentials_from is not None:
        load_archived_credentials(args.credentials_from)

    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("s3", region_name=region)

    rows: list[dict[str, str]] = []
    found_in: list[str] = []
    for season in args.seasons:
        prefix = f"pff-data/tracking/{args.competition}/{season}/"
        try:
            games = list_game_dirs(client, args.bucket, prefix)
        except (BotoCoreError, ClientError) as exc:
            raise SystemExit(f"Failed to list s3://{args.bucket}/{prefix}: {exc}") from exc
        if args.target_game in games:
            found_in.append(season)
        print(f"{season}: {len(games)} games")
        print("  " + " ".join(games))
        rows.extend({"competition": args.competition, "season": season, "game": game} for game in games)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["competition", "season", "game"])
        writer.writeheader()
        writer.writerows(rows)

    if found_in:
        print(f"\nFOUND {args.target_game} in: {', '.join(found_in)}")
    else:
        print(f"\nMISSING {args.target_game} from all listed seasons")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
