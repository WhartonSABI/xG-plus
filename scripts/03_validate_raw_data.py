#!/usr/bin/env python3
"""Validate local raw PFF event and tracking mirrors before feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw event/tracking data mirrors.")
    parser.add_argument("--tracking-root", type=Path, default=Path("pff-tracking"))
    parser.add_argument("--event-root", type=Path, default=Path("pff-events"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--seasons", nargs="+", default=["2022-2023", "2023-2024", "2024-2025"])
    parser.add_argument("--expected-games", type=int, default=380)
    parser.add_argument("--allow-temp-files", action="store_true", help="Warn on *.tmp files instead of failing.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def numeric_game_dirs(season_dir: Path) -> dict[str, Path]:
    if not season_dir.exists():
        return {}
    return {
        path.name: path
        for path in sorted(season_dir.iterdir())
        if path.is_dir() and path.name.isdigit()
    }


def event_files(season_dir: Path) -> dict[str, Path]:
    if not season_dir.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(season_dir.glob("*.csv"))
        if path.stem.isdigit()
    }


def validate_season(args: argparse.Namespace, season: str) -> dict[str, Any]:
    tracking_dir = args.tracking_root / args.competition / season
    event_dir = args.event_root / args.competition / season
    tracking_games = numeric_game_dirs(tracking_dir)
    events = event_files(event_dir)

    missing_tracking_parts: dict[str, list[str]] = {}
    for game, game_dir in tracking_games.items():
        missing = []
        if not (game_dir / f"{game}.jsonl.bz2").exists():
            missing.append("tracking_jsonl")
        if not (game_dir / "metadata.json").exists():
            missing.append("metadata")
        if not (game_dir / "rosters.json").exists():
            missing.append("rosters")
        if missing:
            missing_tracking_parts[game] = missing

    zero_byte_files = [
        str(path)
        for root in [tracking_dir, event_dir]
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.stat().st_size == 0
    ]
    tmp_files = [
        str(path)
        for root in [tracking_dir, event_dir]
        if root.exists()
        for path in root.rglob("*.tmp")
        if path.is_file()
    ]

    event_ids = set(events)
    tracking_ids = set(tracking_games)
    problems: list[str] = []
    if len(tracking_games) != args.expected_games:
        problems.append(f"expected {args.expected_games} tracking games, found {len(tracking_games)}")
    if len(events) != args.expected_games:
        problems.append(f"expected {args.expected_games} event files, found {len(events)}")
    if missing_tracking_parts:
        problems.append(f"{len(missing_tracking_parts)} tracking game dirs are missing required files")
    if event_ids - tracking_ids:
        problems.append(f"{len(event_ids - tracking_ids)} event ids have no tracking dir")
    if tracking_ids - event_ids:
        problems.append(f"{len(tracking_ids - event_ids)} tracking ids have no event csv")
    if zero_byte_files:
        problems.append(f"{len(zero_byte_files)} zero-byte files")
    if tmp_files and not args.allow_temp_files:
        problems.append(f"{len(tmp_files)} temporary download files")

    return {
        "season": season,
        "tracking_games": len(tracking_games),
        "event_files": len(events),
        "event_not_tracking": sorted(event_ids - tracking_ids),
        "tracking_not_event": sorted(tracking_ids - event_ids),
        "missing_tracking_parts": missing_tracking_parts,
        "zero_byte_files": zero_byte_files,
        "tmp_files": tmp_files,
        "warnings": [f"{len(tmp_files)} temporary download files"] if tmp_files and args.allow_temp_files else [],
        "problems": problems,
    }


def main() -> None:
    args = parse_args()
    results = [validate_season(args, season) for season in args.seasons]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            status = "ok" if not result["problems"] else "fail"
            print(
                f"{status}: {result['season']} "
                f"tracking_games={result['tracking_games']} event_files={result['event_files']}"
            )
            for problem in result["problems"]:
                print(f"  {problem}")
            for warning in result["warnings"]:
                print(f"  warn: {warning}")

    if any(result["problems"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
