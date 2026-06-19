#!/usr/bin/env python3
"""Build per-game xG+ tracking features from local PFF tracking files."""

from __future__ import annotations

import argparse
import bz2
import json
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pitch_geometry import GOAL_AREA_WIDTH, PitchDimensions, active_pitch_dimensions
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


OPEN_GOAL_HALF_WIDTH = GOAL_AREA_WIDTH / 2.0
PLAYER_BLOCK_RADIUS = 0.375
# The existing openGoal feature measures the visible six-yard-box mouth,
# so this remains a fixed soccer dimension rather than pitch-width metadata.
EXTRACTION_VERSION = "pitch-possession-v2"


def tangent_inter(x0: float, y0: float, a: float, b: float, r: float, x_line: float) -> list[float]:
    dx = x0 - a
    dy = y0 - b
    d = float(np.hypot(dx, dy))
    if d < r or np.isclose(d, r):
        return [-10.0, 10.0]
    angle_to_point = math.atan2(dy, dx)
    angle_offset = math.asin(r / d)
    angles = [angle_to_point + angle_offset, angle_to_point - angle_offset]
    points: list[float] = []
    for theta in angles:
        dir_x = math.cos(theta)
        dir_y = math.sin(theta)
        if np.isclose(dir_x, 0.0):
            continue
        t = (x_line - x0) / dir_x
        y = y0 + t * dir_y
        points.append(float(y))
    if len(points) < 2:
        return [-10.0, 10.0]
    return sorted(points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract local tracking features for xG+.")
    parser.add_argument("--tracking-root", type=Path, default=Path("pff-tracking"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("data/process_games"))
    parser.add_argument("--limit-games", type=int, default=0, help="0 means all games.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.metadata.json")


def output_is_current(path: Path, version_key: str, version: str) -> bool:
    meta_path = sidecar_path(path)
    if not path.exists() or not meta_path.exists():
        return False
    try:
        metadata = load_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(metadata, dict) and metadata.get(version_key) == version


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_roster_lookup(rows: pd.DataFrame) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        shirt = safe_int(row.get("shirtNumber"))
        if shirt is None:
            continue
        player = row.get("player") or {}
        lookup[shirt] = {
            "id": safe_int(player.get("id")),
            "name": player.get("nickname"),
            "position": row.get("positionGroupType"),
        }
    return lookup


def parse_tracking_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compute_in_play(df: pd.DataFrame) -> pd.Series:
    in_play_flags: list[bool] = []
    in_play = False
    for event in df["game_event"]:
        if event is not None:
            event_type = event.get("game_event_type")
            if event_type in {"OUT", "END"}:
                in_play = False
            elif event_type in {"FIRSTKICKOFF", "SECONDKICKOFF", "OTB", "G"}:
                in_play = True
        in_play_flags.append(in_play)
    return pd.Series(in_play_flags, index=df.index, dtype=bool)


def compute_possession_side(df: pd.DataFrame) -> pd.Series:
    sides: list[bool | None] = []
    current_side: bool | None = None
    for event in df["game_event"]:
        if event is not None:
            event_type = event.get("game_event_type")
            if event_type in {"OUT", "END"}:
                current_side = None
            elif event.get("home_ball") is not None:
                current_side = bool(event.get("home_ball"))
        sides.append(current_side)
    return pd.Series(sides, index=df.index, dtype=object)


def compute_ball_xyz(ball: Any, axis: str) -> float | None:
    if ball is None or not isinstance(ball, dict):
        return None
    value = ball.get(axis)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def game_paths(tracking_root: Path, competition: str, season: str) -> list[tuple[str, Path, Path, Path]]:
    season_dir = tracking_root / competition / season
    if not season_dir.exists():
        return []
    entries: list[tuple[str, Path, Path, Path]] = []
    for game_dir in sorted(season_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        game = game_dir.name
        tracking_path = game_dir / f"{game}.jsonl.bz2"
        metadata_path = game_dir / "metadata.json"
        rosters_path = game_dir / "rosters.json"
        if tracking_path.exists() and metadata_path.exists() and rosters_path.exists():
            entries.append((game, tracking_path, metadata_path, rosters_path))
    return entries


def compute_frame_features(
    attacks: pd.DataFrame,
    home_lookup: dict[int, dict[str, Any]],
    away_lookup: dict[int, dict[str, Any]],
    pitch: PitchDimensions,
) -> pd.DataFrame:
    goal_x = pitch.length / 2.0
    home_relative_rows: list[list[dict[str, Any]]] = []
    away_relative_rows: list[list[dict[str, Any]]] = []
    gk_r_values: list[float] = []
    gk_theta_values: list[float] = []
    open_goal_values: list[float] = []

    frame_values = zip(
        attacks["is_home"].tolist(),
        attacks["x_flipped"].astype(float).tolist(),
        attacks["y_flipped"].astype(float).tolist(),
        attacks["attack_direction"].tolist(),
        attacks["homePlayersSmoothed"].tolist(),
        attacks["awayPlayersSmoothed"].tolist(),
    )

    for atk_home_raw, ball_x, ball_y, atk_dir_raw, home_players, away_players in frame_values:
        cover_l: list[float] = []
        cover_r: list[float] = []
        atk_home = bool(atk_home_raw)
        atk_dir = 0 if pd.isna(atk_dir_raw) else int(atk_dir_raw)
        flip = 1 - 2 * atk_dir
        gk_r = np.nan
        gk_theta = np.nan

        def player_relatives(
            players: list[dict[str, Any]] | None,
            lookup: dict[int, dict[str, Any]],
            defender_side: bool,
        ) -> list[dict[str, Any]]:
            nonlocal gk_r, gk_theta
            relatives: list[dict[str, Any]] = []
            for player in players or []:
                jersey = safe_int(player.get("jerseyNum"))
                x_raw = player.get("x")
                y_raw = player.get("y")
                if jersey is None or x_raw is None or y_raw is None:
                    continue
                x_flipped = float(x_raw) * flip
                y_flipped = float(y_raw) * flip
                r = math.sqrt((goal_x - x_flipped) ** 2 + y_flipped**2)
                theta = math.atan2(y_flipped, (goal_x - x_flipped))
                delta_x = x_flipped - ball_x
                delta_y = y_flipped - ball_y
                dist_ball = math.sqrt(delta_x**2 + delta_y**2)
                angle_ball = math.atan2(delta_x, delta_y)
                roster = lookup.get(jersey, {})
                rel = {
                    "jerseyNum": jersey,
                    "r": r,
                    "theta": theta,
                    "dist_ball": dist_ball,
                    "angle_ball": angle_ball,
                    "position": roster.get("position"),
                }

                if defender_side:
                    if rel["position"] == "GK":
                        gk_r = r
                        gk_theta = theta
                        continue
                    if ball_x <= x_flipped:
                        left, right = tangent_inter(
                            x_flipped,
                            y_flipped,
                            ball_x,
                            ball_y,
                            PLAYER_BLOCK_RADIUS,
                            goal_x,
                        )
                        lo = max(left, -OPEN_GOAL_HALF_WIDTH)
                        hi = min(right, OPEN_GOAL_HALF_WIDTH)
                        if lo <= hi:
                            cover_l.append(lo)
                            cover_r.append(hi)

                relatives.append(rel)
            return sorted(relatives, key=lambda x: x["dist_ball"])

        home_relatives = player_relatives(home_players, home_lookup, defender_side=not atk_home)
        away_relatives = player_relatives(away_players, away_lookup, defender_side=atk_home)

        marks = [-OPEN_GOAL_HALF_WIDTH, OPEN_GOAL_HALF_WIDTH]
        for i in range(len(cover_l)):
            marks.extend([cover_l[i], cover_r[i]])
        marks = sorted(set(marks))
        idx = {x: i for i, x in enumerate(marks)}
        intervals = sorted((idx[cover_l[i]], idx[cover_r[i]]) for i in range(len(cover_l)))

        seg = 0
        pos = 0
        covered = 0.0
        while seg < len(intervals):
            covered += marks[intervals[seg][0]] - marks[pos]
            pos = intervals[seg][1]
            seg += 1
            while seg < len(intervals) and intervals[seg][0] <= pos:
                pos = max(pos, intervals[seg][1])
                seg += 1
        open_goal = (covered + (marks[-1] - marks[pos])) / GOAL_AREA_WIDTH

        home_relative_rows.append(home_relatives)
        away_relative_rows.append(away_relatives)
        gk_r_values.append(gk_r)
        gk_theta_values.append(gk_theta)
        open_goal_values.append(open_goal)

    attacks["homePlayersRelative"] = home_relative_rows
    attacks["awayPlayersRelative"] = away_relative_rows
    attacks["GK_r"] = gk_r_values
    attacks["GK_theta"] = gk_theta_values
    attacks["openGoal"] = open_goal_values

    def get_metric(players: list[dict[str, Any]], idx: int, key: str) -> float:
        if idx < 0 or idx >= len(players):
            return np.nan
        value = players[idx].get(key)
        return float(value) if value is not None else np.nan

    is_home_values = [bool(value) for value in attacks["is_home"].tolist()]
    for i in range(5):
        defenders = [
            away_relatives if is_home else home_relatives
            for is_home, home_relatives, away_relatives in zip(is_home_values, home_relative_rows, away_relative_rows)
        ]
        attackers = [
            home_relatives if is_home else away_relatives
            for is_home, home_relatives, away_relatives in zip(is_home_values, home_relative_rows, away_relative_rows)
        ]
        attacks[f"DefDist{i}"] = [get_metric(players, i, "dist_ball") for players in defenders]
        attacks[f"DefAngle{i}"] = [get_metric(players, i, "angle_ball") for players in defenders]
        attacks[f"OffDist{i}"] = [get_metric(players, i + 1, "dist_ball") for players in attackers]
        attacks[f"OffAngle{i}"] = [get_metric(players, i + 1, "angle_ball") for players in attackers]
    return attacks


def process_one_game(
    competition: str,
    season: str,
    game: str,
    tracking_path: Path,
    metadata_path: Path,
    rosters_path: Path,
    output_dir: Path,
) -> tuple[str, int, str]:
    df = parse_tracking_jsonl(tracking_path)
    meta = load_json(metadata_path)
    pitch = active_pitch_dimensions(meta if isinstance(meta, dict) else {})
    rosters = pd.DataFrame(load_json(rosters_path))

    if df.empty:
        return game, 0, "empty"

    df = df.dropna(subset=["homePlayersSmoothed", "awayPlayersSmoothed"]).reset_index(drop=True)
    home_count = df["homePlayersSmoothed"].apply(lambda x: len(x or []))
    away_count = df["awayPlayersSmoothed"].apply(lambda x: len(x or []))
    df = df[(home_count <= 11) & (away_count <= 11)].reset_index(drop=True)
    df["in_play"] = compute_in_play(df)
    df = df[df["in_play"]].reset_index(drop=True)
    if df.empty:
        return game, 0, "no_in_play"

    df["is_home"] = compute_possession_side(df)
    df["period"] = df["period"].astype(float) - 1

    home_start_left = bool(meta.get("homeTeamStartLeft", False))
    attack_direction: list[float] = []
    for is_home, period in zip(df["is_home"], df["period"]):
        if is_home is None or pd.isna(is_home):
            attack_direction.append(np.nan)
            continue
        value = bool(is_home) ^ home_start_left ^ bool(int(period) % 2)
        attack_direction.append(int(value))
    df["attack_direction"] = attack_direction

    for axis in ["x", "y", "z"]:
        df[axis] = df["ballsSmoothed"].apply(lambda ball, k=axis: compute_ball_xyz(ball, k))
        df[axis] = df[axis].interpolate(method="linear")
        df[axis] = df[axis].ffill().bfill()

    third_x = pitch.length / 2.0 - pitch.length / 3.0
    attack_counter = 0
    current_side = -1
    attack_ids = [0] * len(df)

    def cleared(ball_x: float, atk_dir: Any) -> bool:
        if pd.isna(atk_dir):
            return False
        return ball_x > -third_x if int(atk_dir) == 1 else ball_x < third_x

    def side_value(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        return int(bool(value))

    for i in range(len(df)):
        bx = float(df.at[i, "x"])
        side = side_value(df.at[i, "is_home"])
        atk_dir = df.at[i, "attack_direction"]
        if current_side == 1:
            if side == 0 or cleared(bx, atk_dir):
                current_side = -1
        elif current_side == 0:
            if side == 1 or cleared(bx, atk_dir):
                current_side = -1
        else:
            if side is not None and not pd.isna(atk_dir):
                in_zone = (int(atk_dir) == 1 and bx <= -third_x) or (int(atk_dir) == 0 and bx >= third_x)
                if in_zone:
                    current_side = side
                    attack_counter += 1
        if current_side >= 0:
            attack_ids[i] = attack_counter
    df["attack"] = attack_ids

    attacks = df[df["attack"] > 0].reset_index(drop=True)
    if attacks.empty:
        return game, 0, "no_attacks"

    attacks["is_shot"] = attacks["possession_event"].apply(
        lambda x: bool(x and x.get("possession_event_type") == "SH")
    )
    for window in [10000, 5000, 3000, 1000]:
        attacks[f"hasShotsIn{int(window/1000)}s"] = False
    shot_times = attacks.loc[attacks["is_shot"], "videoTimeMs"].tolist()
    for shot_time in shot_times:
        attacks.loc[
            (attacks["videoTimeMs"] >= shot_time - 10000) & (attacks["videoTimeMs"] <= shot_time),
            "hasShotsIn10s",
        ] = True
        attacks.loc[
            (attacks["videoTimeMs"] >= shot_time - 5000) & (attacks["videoTimeMs"] <= shot_time),
            "hasShotsIn5s",
        ] = True
        attacks.loc[
            (attacks["videoTimeMs"] >= shot_time - 3000) & (attacks["videoTimeMs"] <= shot_time),
            "hasShotsIn3s",
        ] = True
        attacks.loc[
            (attacks["videoTimeMs"] >= shot_time - 1000) & (attacks["videoTimeMs"] <= shot_time),
            "hasShotsIn1s",
        ] = True

    attacks["x_flipped"] = attacks["x"] * (1 - 2 * attacks["attack_direction"].fillna(0).astype(int))
    attacks["y_flipped"] = attacks["y"] * (1 - 2 * attacks["attack_direction"].fillna(0).astype(int))
    goal_x = pitch.length / 2.0
    attacks["r"] = np.sqrt((goal_x - attacks["x_flipped"]) ** 2 + attacks["y_flipped"] ** 2)
    attacks["theta"] = np.arctan2(attacks["y_flipped"], (goal_x - attacks["x_flipped"]))
    attacks["time_s"] = attacks["videoTimeMs"] / 1000.0
    grouped_attack = attacks.groupby(["period", "attack"], dropna=False)
    dx = grouped_attack["x"].diff()
    dy = grouped_attack["y"].diff()
    dt = grouped_attack["time_s"].diff()
    speed = np.sqrt(dx**2 + dy**2) / dt
    attacks["speed"] = speed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    home_team = meta.get("homeTeam", {}) if isinstance(meta, dict) else {}
    away_team = meta.get("awayTeam", {}) if isinstance(meta, dict) else {}
    home_lookup = build_roster_lookup(rosters[rosters["team"].apply(lambda x: str(x.get("id")) == str(home_team.get("id")))])
    away_lookup = build_roster_lookup(rosters[rosters["team"].apply(lambda x: str(x.get("id")) == str(away_team.get("id")))])
    attacks = compute_frame_features(attacks, home_lookup, away_lookup, pitch)

    attacks["competition"] = competition
    attacks["season"] = season
    attacks["game"] = game
    attacks["pitch_id"] = pitch.pitch_id
    attacks["pitch_length"] = pitch.length
    attacks["pitch_width"] = pitch.width

    drop_cols = [
        "version",
        "gameRefId",
        "generatedTime",
        "smoothedTime",
        "homePlayers",
        "homePlayersSmoothed",
        "awayPlayers",
        "awayPlayersSmoothed",
        "balls",
        "ballsSmoothed",
        "game_event",
        "possession_event",
        "in_play",
        "x",
        "y",
        "time_s",
    ]
    kept = attacks.drop(columns=[col for col in drop_cols if col in attacks.columns])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"atk_{competition}_{season}_{game}.pkl.bz2"
    kept.to_pickle(out_path, compression="bz2")
    sidecar_path(out_path).write_text(
        json.dumps(
            {
                "extraction_version": EXTRACTION_VERSION,
                "competition": competition,
                "season": season,
                "game": game,
                "rows": int(len(kept)),
                "pitch_id": pitch.pitch_id,
                "pitch_length": pitch.length,
                "pitch_width": pitch.width,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return game, len(kept), "ok"


def main() -> None:
    args = parse_args()
    entries = game_paths(args.tracking_root, args.competition, args.season)
    if args.limit_games > 0:
        entries = entries[: args.limit_games]
    if not entries:
        raise SystemExit(f"No games found under {args.tracking_root / args.competition / args.season}")

    tasks: list[tuple[str, Path, Path, Path]] = []
    for game, tracking_path, metadata_path, rosters_path in entries:
        out_path = args.output_dir / f"atk_{args.competition}_{args.season}_{game}.pkl.bz2"
        if args.skip_existing and output_is_current(out_path, "extraction_version", EXTRACTION_VERSION):
            continue
        tasks.append((game, tracking_path, metadata_path, rosters_path))

    if not tasks:
        print("All outputs already exist; nothing to do.")
        return

    results: list[tuple[str, int, str]] = []
    if args.workers <= 1:
        for game, tracking_path, metadata_path, rosters_path in tqdm(tasks, total=len(tasks), desc="Extract tracking"):
            results.append(
                process_one_game(
                    args.competition,
                    args.season,
                    game,
                    tracking_path,
                    metadata_path,
                    rosters_path,
                    args.output_dir,
                )
            )
    else:
        executor_cls = ProcessPoolExecutor
        try:
            with executor_cls(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        process_one_game,
                        args.competition,
                        args.season,
                        game,
                        tracking_path,
                        metadata_path,
                        rosters_path,
                        args.output_dir,
                    )
                    for game, tracking_path, metadata_path, rosters_path in tasks
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extract tracking"):
                    results.append(future.result())
        except PermissionError:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        process_one_game,
                        args.competition,
                        args.season,
                        game,
                        tracking_path,
                        metadata_path,
                        rosters_path,
                        args.output_dir,
                    )
                    for game, tracking_path, metadata_path, rosters_path in tasks
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extract tracking (threads)"):
                    results.append(future.result())

    ok = [row for row in results if row[2] == "ok"]
    skipped = [row for row in results if row[2] != "ok"]
    print(f"Processed {len(ok)} games successfully.")
    if skipped:
        print(f"Skipped/empty {len(skipped)} games:")
        for game, rows, status in sorted(skipped):
            print(f"  {game}: {status} ({rows} rows)")


if __name__ == "__main__":
    main()
