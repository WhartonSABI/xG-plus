#!/usr/bin/env python3
"""Export selected final attacks in the sample-data CSV shape."""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pitch_geometry import GOAL_HALF_WIDTH, active_pitch_dimensions


REQUESTED_ATTACKS = {212, 226, 268, 285, 340, 408}


def tracking_paths(root: Path, game: int) -> tuple[Path, Path, Path]:
    matches = sorted(root.glob(f"**/{game}/{game}.jsonl.bz2"))
    if not matches:
        raise FileNotFoundError(f"Could not find tracking for game {game} under {root}")
    tracking_path = matches[0]
    game_dir = tracking_path.parent
    return tracking_path, game_dir / "metadata.json", game_dir / "rosters.json"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def side_team_ids(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "home": str(metadata.get("homeTeam", {}).get("id", "")),
        "away": str(metadata.get("awayTeam", {}).get("id", "")),
    }


def roster_name_map(rosters: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[tuple[str, str], str]:
    team_ids = side_team_ids(metadata)
    id_to_side = {team_id: side for side, team_id in team_ids.items()}
    names: dict[tuple[str, str], str] = {}
    for row in rosters:
        team_id = str(row.get("team", {}).get("id", ""))
        side = id_to_side.get(team_id)
        shirt = row.get("shirtNumber")
        name = row.get("player", {}).get("nickname")
        if side and shirt and name:
            names[(side, str(shirt))] = str(name)
    return names


def attack_team_name(metric_row: pd.Series) -> str:
    if int(metric_row["attack_team_id"]) == int(metric_row["home_id"]):
        return str(metric_row["home_name"])
    if int(metric_row["attack_team_id"]) == int(metric_row["away_id"]):
        return str(metric_row["away_name"])
    return f"Team {metric_row['attack_team_id']}"


def player_rows(
    players: list[dict[str, Any]] | None,
    team: str,
    frame_num: int,
    video_time_s: float,
    names: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for player in players or []:
        x = player.get("x")
        y = player.get("y")
        if x is None or y is None:
            continue
        jersey = player.get("jerseyNum")
        jersey_key = str(jersey) if jersey is not None else ""
        rows.append(
            {
                "frameNum": frame_num,
                "video_time_s": video_time_s,
                "team": team,
                "jerseyNum": float(jersey) if jersey is not None else None,
                "player_name": names.get((team, jersey_key)),
                "x": x,
                "y": y,
            }
        )
    return rows


def ball_row(frame: dict[str, Any], frame_num: int, video_time_s: float) -> dict[str, Any] | None:
    ball = frame.get("ballsSmoothed")
    if not isinstance(ball, dict) or ball.get("x") is None or ball.get("y") is None:
        balls = frame.get("balls")
        ball = balls[0] if isinstance(balls, list) and balls else None
    if not isinstance(ball, dict) or ball.get("x") is None or ball.get("y") is None:
        return None
    return {
        "frameNum": frame_num,
        "video_time_s": video_time_s,
        "team": "ball",
        "jerseyNum": None,
        "player_name": None,
        "x": ball.get("x"),
        "y": ball.get("y"),
    }


def ball_xy(frame: dict[str, Any]) -> tuple[float, float] | None:
    ball = frame.get("ballsSmoothed")
    if not isinstance(ball, dict) or ball.get("x") is None or ball.get("y") is None:
        balls = frame.get("balls")
        ball = balls[0] if isinstance(balls, list) and balls else None
    if not isinstance(ball, dict) or ball.get("x") is None or ball.get("y") is None:
        return None
    return float(ball["x"]), float(ball["y"])


def goal_line_cross_time(
    tracking_path: Path,
    pff_period: int,
    shot_time: float,
    scan_seconds: float,
    pitch_length: float,
) -> float | None:
    """Return first clock where ball appears across either goal line after shot."""
    end_scan = shot_time + scan_seconds
    goal_x = pitch_length / 2.0
    with bz2.open(tracking_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            frame = json.loads(line)
            period = int(frame.get("period", -1))
            clock = float(frame.get("periodGameClockTime", -1))
            if period < pff_period:
                continue
            if period > pff_period or clock > end_scan:
                break
            if clock < shot_time:
                continue
            xy = ball_xy(frame)
            if xy is None:
                continue
            x, y = xy
            if abs(x) >= goal_x - 0.1 and abs(y) <= GOAL_HALF_WIDTH + 0.15:
                return clock
    return None


def export_tracking(
    tracking_path: Path,
    metadata_path: Path,
    roster_path: Path,
    pff_period: int,
    start_time: float,
    end_time: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = load_json(metadata_path)
    rosters = load_json(roster_path)
    pitch = active_pitch_dimensions(metadata)
    names = roster_name_map(rosters, metadata)
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    with bz2.open(tracking_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            frame = json.loads(line)
            period = int(frame.get("period", -1))
            clock = float(frame.get("periodGameClockTime", -1))
            if period > pff_period:
                break
            if period == pff_period and clock > end_time:
                break
            if period != pff_period or clock < start_time or clock > end_time:
                continue
            frame_num = int(frame["frameNum"])
            video_time_s = float(frame.get("videoTimeMs", 0)) / 1000.0
            frames.append({"frameNum": frame_num, "periodGameClockTime": clock})
            rows.extend(player_rows(frame.get("homePlayersSmoothed"), "home", frame_num, video_time_s, names))
            rows.extend(player_rows(frame.get("awayPlayersSmoothed"), "away", frame_num, video_time_s, names))
            ball = ball_row(frame, frame_num, video_time_s)
            if ball:
                rows.append(ball)
    if not rows:
        raise ValueError(f"No tracking rows in {tracking_path} for period {pff_period} {start_time}-{end_time}")
    tracking = pd.DataFrame(rows)
    tracking["pitch_id"] = pitch.pitch_id
    tracking["pitch_length"] = pitch.length
    tracking["pitch_width"] = pitch.width
    return tracking, pd.DataFrame(frames).drop_duplicates().sort_values("periodGameClockTime")


def add_nearest_frame(metric: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    metric = metric.sort_values("periodGameClockTime").copy()
    forward = pd.merge_asof(
        metric,
        frames.sort_values("periodGameClockTime"),
        on="periodGameClockTime",
        direction="forward",
    )
    backward = pd.merge_asof(
        metric,
        frames.sort_values("periodGameClockTime"),
        on="periodGameClockTime",
        direction="backward",
    )
    # Prefer the first tracking frame at/after the model timestamp for
    # event alignment, with backward fill only when forward is unavailable.
    forward["frameNum"] = forward["frameNum"].fillna(backward["frameNum"])
    return forward


def final_rows(final_dir: Path, attacks: set[int]) -> pd.DataFrame:
    rows = []
    for path in sorted(final_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df = df[df["attack_merged"].astype(int).isin(attacks)].copy()
        if not df.empty:
            df["source_file"] = path.name
            rows.append(df)
    if not rows:
        raise ValueError(f"No requested attacks found in {final_dir}")
    return pd.concat(rows, ignore_index=True)


def selected_attack_rows(predictions: pd.DataFrame, attacks: set[int], final_dir: Path | None) -> pd.DataFrame:
    if final_dir is not None and final_dir.exists():
        try:
            return final_rows(final_dir, attacks)
        except ValueError:
            pass
    if "attack_merged" not in predictions.columns or "game" not in predictions.columns:
        raise ValueError("Predictions must include game and attack_merged columns")
    rows = predictions[predictions["attack_merged"].astype(int).isin(attacks)][["game", "attack_merged"]].drop_duplicates()
    if rows.empty:
        raise ValueError("No requested attacks found in predictions")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected attacks to sample-data-like CSVs.")
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions/predictions_pl_2024-2025.csv"))
    parser.add_argument("--shots", type=Path, default=Path("data/predictions/shots_pl_2024-2025.csv"))
    parser.add_argument("--final-dir", type=Path, default=None)
    parser.add_argument("--tracking-root", type=Path, default=Path("pff-tracking"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/attack_samples"))
    parser.add_argument("--attacks", type=int, nargs="+", default=sorted(REQUESTED_ATTACKS))
    parser.add_argument("--post-shot-seconds", type=float, default=1.0)
    parser.add_argument("--goal-line-scan-seconds", type=float, default=8.0)
    parser.add_argument("--post-goal-line-seconds", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.attacks)
    predictions = pd.read_csv(args.predictions)
    final = selected_attack_rows(predictions, requested, args.final_dir)
    if args.shots.exists():
        shots = pd.read_csv(args.shots)
    else:
        shots = predictions[predictions["is_shot"].astype(str).str.lower().isin({"true", "1"})].copy()
    written: list[str] = []

    for row in final.sort_values(["attack_merged", "game"]).itertuples(index=False):
        game = int(row.game)
        attack = int(row.attack_merged)
        metric = predictions[
            (predictions["game"].astype(int) == game)
            & (predictions["attack_merged"].astype(int) == attack)
        ].copy()
        if metric.empty:
            raise ValueError(f"No prediction rows for game {game}, attack {attack}")

        shot_rows = shots[
            (shots["game"].astype(int) == game)
            & (shots["attack_merged"].astype(int) == attack)
        ].copy()
        if not shot_rows.empty:
            metric = pd.concat([metric, shot_rows], ignore_index=True)
            metric["is_shot_bool"] = metric["is_shot"].astype(str).str.lower().isin({"true", "1"})
            metric = (
                metric.sort_values(["periodGameClockTime", "is_shot_bool"], ascending=[True, False])
                .drop_duplicates(subset=["periodGameClockTime"], keep="first")
                .drop(columns=["is_shot_bool"])
                .reset_index(drop=True)
            )

        tracking_path, metadata_path, roster_path = tracking_paths(args.tracking_root, game)
        metadata = load_json(metadata_path)
        pitch = active_pitch_dimensions(metadata)
        pff_period = int(metric["period"].iloc[0]) + 1

        goal_rows = metric[metric["is_goal"].astype(str).str.lower().isin({"true", "1"})]
        if not goal_rows.empty:
            shot_time = float(goal_rows["periodGameClockTime"].min())
            end_time = shot_time + args.post_shot_seconds
            cross_time = goal_line_cross_time(
                tracking_path=tracking_path,
                pff_period=pff_period,
                shot_time=shot_time,
                scan_seconds=args.goal_line_scan_seconds,
                pitch_length=pitch.length,
            )
            if cross_time is not None:
                end_time = max(end_time, cross_time + args.post_goal_line_seconds)
        else:
            end_time = float(metric["periodGameClockTime"].max())

        start_time = float(metric["periodGameClockTime"].min())
        metric = metric[(metric["periodGameClockTime"] >= start_time) & (metric["periodGameClockTime"] <= end_time)].copy()

        tracking, frames = export_tracking(
            tracking_path,
            metadata_path,
            roster_path,
            pff_period,
            start_time=start_time,
            end_time=end_time,
        )
        metric = add_nearest_frame(metric, frames)
        metric["is_shot_bool"] = metric["is_shot"].astype(str).str.lower().isin({"true", "1"})
        metric["is_goal_bool"] = metric["is_goal"].astype(str).str.lower().isin({"true", "1"})
        metric = (
            metric.sort_values(["frameNum", "is_shot_bool", "is_goal_bool", "periodGameClockTime"])
            .drop_duplicates(subset=["frameNum"], keep="last")
            .drop(columns=["is_shot_bool", "is_goal_bool"])
            .sort_values("periodGameClockTime")
            .reset_index(drop=True)
        )
        metric["xG_plus"] = metric["goal_proba"]
        metric["int_sec"] = metric["periodGameClockTime"].astype(int)
        metric["gameId"] = metric["game"]
        if "pitch_id" not in metric.columns:
            metric["pitch_id"] = pitch.pitch_id
        elif pitch.pitch_id is not None:
            metric["pitch_id"] = metric["pitch_id"].fillna(pitch.pitch_id)
        for col, value in [("pitch_length", pitch.length), ("pitch_width", pitch.width)]:
            if col not in metric.columns:
                metric[col] = value
            else:
                metric[col] = pd.to_numeric(metric[col], errors="coerce").fillna(value)
        if "competition" not in metric.columns:
            metric["competition"] = "pl"
        if "season" not in metric.columns:
            metric["season"] = tracking_path.parts[-3]
        metric["attack_team_name"] = metric.apply(attack_team_name, axis=1)
        metric["is_home"] = metric["attack_team_id"].astype(int) == metric["home_id"].astype(int)

        metric_path = args.output_dir / f"atk{attack}.csv"
        tracking_out_path = args.output_dir / f"atk{attack}_tracking.csv"
        metric.to_csv(metric_path, index=False)
        tracking.to_csv(tracking_out_path, index=False)
        written.extend([str(metric_path), str(tracking_out_path)])
        print(f"wrote atk{attack}: {len(metric)} metric rows, {len(tracking)} tracking rows")

    print("written")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
