#!/usr/bin/env python3
"""Attach event-derived labels and metadata to extracted xG+ feature tables."""

from __future__ import annotations

import argparse
import ast
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


FEATURES = ["r", "theta", "z", "speed", "GK_r", "GK_theta", "openGoal"]
SHOT_WINDOWS = [
    ("hasShotsIn10s", 10.0),
    ("hasShotsIn5s", 5.0),
    ("hasShotsIn3s", 3.0),
    ("hasShotsIn1s", 1.0),
    ("hasShotsIn0.5s", 0.5),
]
TARGETS = [name for name, _ in SHOT_WINDOWS] + ["is_shot", "is_goal"]
METAS = [
    "game",
    "date",
    "home_id",
    "home_name",
    "away_id",
    "away_name",
    "competition",
    "season",
    "attack",
    "attack_merged",
    "period",
    "is_home",
    "frameNum",
    "periodGameClockTime",
    "videoTimeMs",
    "attack_team_id",
    "player_name",
    "player_id",
]

for i in range(5):
    FEATURES.append(f"DefDist{i}")
    FEATURES.append(f"DefAngle{i}")
    FEATURES.append(f"OffDist{i}")
    FEATURES.append(f"OffAngle{i}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label extracted xG+ features with event outcomes.")
    parser.add_argument("--process-dir", type=Path, default=Path("data/process_games"))
    parser.add_argument("--tracking-root", type=Path, default=Path("pff-tracking"))
    parser.add_argument("--event-root", type=Path, default=Path("pff-events"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/merged_games"))
    parser.add_argument("--chunk-dir", type=Path, default=Path("data/merged_data"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-missing-events", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_identifier(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def merge_attacks(atk: pd.DataFrame) -> pd.DataFrame:
    tmp = (
        atk.groupby(["game", "attack", "is_home"], dropna=False)
        .agg(start_time=("videoTimeMs", "min"), end_time=("videoTimeMs", "max"))
        .reset_index()
        .sort_values(["game", "is_home", "start_time"])
        .reset_index(drop=True)
    )
    tmp["attack_merged"] = tmp["attack"]
    for i in range(1, len(tmp)):
        same_game = tmp.at[i, "game"] == tmp.at[i - 1, "game"]
        same_side = bool(tmp.at[i, "is_home"]) == bool(tmp.at[i - 1, "is_home"])
        close_gap = tmp.at[i, "start_time"] - tmp.at[i - 1, "end_time"] <= 5000
        if same_game and same_side and close_gap:
            tmp.at[i, "attack_merged"] = tmp.at[i - 1, "attack_merged"]
    return atk.merge(tmp[["game", "attack", "attack_merged"]], on=["game", "attack"], how="left")


def build_roster_lookup(rosters: list[dict[str, Any]], team_id: str) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for row in rosters:
        team = row.get("team") or {}
        if str(team.get("id")) != str(team_id):
            continue
        shirt = safe_int(row.get("shirtNumber"))
        if shirt is None:
            continue
        player = row.get("player") or {}
        lookup[shirt] = {
            "id": safe_int(player.get("id")),
            "name": player.get("nickname"),
        }
    return lookup


def parse_possession_events(text: Any) -> list[dict[str, Any]]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    if isinstance(text, list):
        return text
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def label_with_events(atk: pd.DataFrame, event_csv: Path, allow_missing: bool) -> pd.DataFrame:
    for target in TARGETS:
        atk[target] = False
    if "game_event_id" not in atk.columns:
        if allow_missing:
            return atk
        raise ValueError("Extracted attack data is missing game_event_id; cannot join events")
    if not event_csv.exists():
        if allow_missing:
            return atk
        raise FileNotFoundError(f"Missing event file: {event_csv}")

    event = pd.read_csv(event_csv)
    if "possessionEvents" not in event.columns or "id" not in event.columns:
        if allow_missing:
            return atk
        raise ValueError(f"Unexpected schema in {event_csv}")

    attack_event_ids = atk["game_event_id"].map(normalize_identifier)
    attack_times = pd.to_numeric(atk["periodGameClockTime"], errors="coerce")
    attack_sides = atk["is_home"].map(lambda value: bool(value) if not pd.isna(value) else pd.NA)

    for _, row in event.iterrows():
        pos_events = parse_possession_events(row["possessionEvents"])
        if not pos_events:
            continue
        first = pos_events[0]
        if first.get("possessionEventType") != "SH":
            continue

        game_event_id = normalize_identifier(row["id"])
        shot_rows = atk[attack_event_ids == game_event_id]
        if shot_rows.empty:
            continue
        idx = shot_rows.index[0]
        atk.at[idx, "is_shot"] = True
        outcome = ((first.get("shootingEvent") or {}).get("shotOutcomeType") == "G")
        atk.at[idx, "is_goal"] = bool(outcome)

        shot_time = float(atk.at[idx, "periodGameClockTime"])
        shot_period = atk.at[idx, "period"]
        shot_side = bool(atk.at[idx, "is_home"])
        attack_col = "attack_merged" if "attack_merged" in atk.columns else "attack"
        shot_attack = atk.at[idx, attack_col]
        base_mask = (
            (atk["period"] == shot_period)
            & (attack_sides == shot_side)
            & (atk[attack_col] == shot_attack)
            & (attack_times <= shot_time)
        )
        for target, seconds in SHOT_WINDOWS:
            atk.loc[base_mask & (attack_times >= shot_time - seconds), target] = True

    return atk


def add_player_and_match_meta(
    atk: pd.DataFrame,
    metadata: dict[str, Any],
    rosters: list[dict[str, Any]],
) -> pd.DataFrame:
    home_team = metadata.get("homeTeam", {}) or {}
    away_team = metadata.get("awayTeam", {}) or {}
    home_lookup = build_roster_lookup(rosters, str(home_team.get("id")))
    away_lookup = build_roster_lookup(rosters, str(away_team.get("id")))

    def player_meta(row: pd.Series) -> tuple[int | None, str | None]:
        attacker_list = row["homePlayersRelative"] if bool(row["is_home"]) else row["awayPlayersRelative"]
        if not isinstance(attacker_list, list) or not attacker_list:
            return None, None
        shirt = safe_int(attacker_list[0].get("jerseyNum"))
        if shirt is None:
            return None, None
        lookup = home_lookup if bool(row["is_home"]) else away_lookup
        player = lookup.get(shirt, {})
        return player.get("id"), player.get("name")

    player_rows = atk.apply(player_meta, axis=1)
    atk["player_id"] = player_rows.apply(lambda x: x[0])
    atk["player_name"] = player_rows.apply(lambda x: x[1])
    atk["attack_team_id"] = atk["is_home"].apply(lambda x: str(home_team.get("id")) if bool(x) else str(away_team.get("id")))

    atk["date"] = metadata.get("date")
    atk["home_id"] = str(home_team.get("id"))
    atk["home_name"] = home_team.get("name")
    atk["away_id"] = str(away_team.get("id"))
    atk["away_name"] = away_team.get("name")
    return atk


def process_one_file(
    process_file: Path,
    tracking_root: Path,
    event_root: Path,
    output_dir: Path,
    allow_missing_events: bool,
) -> tuple[str, int, str]:
    atk = pd.read_pickle(process_file, compression="bz2")
    if atk.empty:
        return process_file.name, 0, "empty"

    competition = str(atk["competition"].iloc[0])
    season = str(atk["season"].iloc[0])
    game = str(atk["game"].iloc[0])

    atk = merge_attacks(atk)
    event_csv = event_root / competition / season / f"{game}.csv"
    atk = label_with_events(atk, event_csv, allow_missing_events)

    metadata_path = tracking_root / competition / season / game / "metadata.json"
    rosters_path = tracking_root / competition / season / game / "rosters.json"
    metadata = load_json(metadata_path)
    rosters = load_json(rosters_path)
    if not isinstance(metadata, dict) or not isinstance(rosters, list):
        return process_file.name, 0, "bad_meta"

    atk = add_player_and_match_meta(atk, metadata, rosters)

    keep = FEATURES + TARGETS + METAS
    for col in keep:
        if col not in atk.columns:
            atk[col] = np.nan
    out = atk[keep].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"train_{competition}_{season}_{game}.csv"
    out.to_csv(out_path, index=False)
    return process_file.name, len(out), "ok"


def build_chunks(source_dir: Path, chunk_dir: Path, competition: str, season: str, chunk_size: int) -> list[Path]:
    files = sorted(source_dir.glob(f"train_{competition}_{season}_*.csv"))
    if not files:
        return []
    chunk_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for chunk_idx, start in enumerate(range(0, len(files), chunk_size)):
        part = files[start : start + chunk_size]
        df = pd.concat((pd.read_csv(path) for path in part), ignore_index=True)
        out_path = chunk_dir / f"train_{competition}_{season}_{chunk_idx}.csv"
        df.to_csv(out_path, index=False)
        outputs.append(out_path)
    return outputs


def main() -> None:
    args = parse_args()
    process_files = sorted(args.process_dir.glob(f"atk_{args.competition}_{args.season}_*.pkl.bz2"))
    if not process_files:
        raise SystemExit(f"No process files found in {args.process_dir} for {args.competition} {args.season}")

    tasks: list[Path] = []
    for path in process_files:
        game = path.name.removesuffix(".pkl.bz2").split("_")[-1]
        out_path = args.output_dir / f"train_{args.competition}_{args.season}_{game}.csv"
        if args.skip_existing and out_path.exists():
            continue
        tasks.append(path)

    if tasks:
        results: list[tuple[str, int, str]] = []
        if args.workers <= 1:
            for process_file in tqdm(tasks, total=len(tasks), desc="Label/export games"):
                results.append(
                    process_one_file(
                        process_file,
                        args.tracking_root,
                        args.event_root,
                        args.output_dir,
                        args.allow_missing_events,
                    )
                )
        else:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(
                            process_one_file,
                            process_file,
                            args.tracking_root,
                            args.event_root,
                            args.output_dir,
                            args.allow_missing_events,
                        )
                        for process_file in tasks
                    ]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Label/export games"):
                        results.append(future.result())
            except PermissionError:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(
                            process_one_file,
                            process_file,
                            args.tracking_root,
                            args.event_root,
                            args.output_dir,
                            args.allow_missing_events,
                        )
                        for process_file in tasks
                    ]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Label/export games (threads)"):
                        results.append(future.result())
        ok = [row for row in results if row[2] == "ok"]
        bad = [row for row in results if row[2] != "ok"]
        print(f"Labeled {len(ok)} games.")
        if bad:
            print(f"Non-ok {len(bad)} games:")
            for row in sorted(bad):
                print(f"  {row[0]}: {row[2]} ({row[1]} rows)")
    else:
        print("All labeled files already exist; skipping per-game export.")

    chunks = build_chunks(args.output_dir, args.chunk_dir, args.competition, args.season, args.chunk_size)
    if not chunks:
        raise SystemExit("No chunked outputs were created.")
    print(f"Wrote {len(chunks)} chunk files to {args.chunk_dir}")


if __name__ == "__main__":
    main()
