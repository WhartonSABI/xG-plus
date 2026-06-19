#!/usr/bin/env python3
"""Create an xG+ GIF by joining model predictions to PFF tracking.

This keeps the visual style of ``visualize_attack_gif.py`` but uses richer PFF
tracking JSONL files:

* ``homePlayersSmoothed`` / ``awayPlayersSmoothed`` for player dots
* ``ballsSmoothed`` for the ball, falling back to raw ``balls``
* ``data/predictions/predictions_pl_2024-2025.csv`` for xS, xG, and xG+ model predictions

The prediction table uses zero-indexed periods; PFF tracking uses one-indexed
periods. The join therefore maps ``prediction.period + 1`` to
``tracking.period`` and aligns by ``periodGameClockTime``.
"""

from __future__ import annotations

import argparse
import bz2
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw
from pitch_geometry import active_pitch_dimensions


def load_render_helpers():
    helper_path = Path(__file__).with_name("11_render_sample_attack_gif.py")
    spec = importlib.util.spec_from_file_location("render_sample_attack_gif", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load render helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_render_helpers = load_render_helpers()
COLORS = _render_helpers.COLORS
draw_cumulative_panel = _render_helpers.draw_cumulative_panel
draw_metric_box = _render_helpers.draw_metric_box
draw_pitch = _render_helpers.draw_pitch
draw_players = _render_helpers.draw_players
draw_sparkline_panel = _render_helpers.draw_sparkline_panel
draw_text = _render_helpers.draw_text
load_font = _render_helpers.load_font
pitch_from_data = _render_helpers.pitch_from_data


def prediction_rows(predictions_path: Path, game: int, attack: int) -> pd.DataFrame:
    usecols = [
        "game",
        "date",
        "home_id",
        "home_name",
        "away_id",
        "away_name",
        "attack_team_id",
        "attack_merged",
        "period",
        "periodGameClockTime",
        "player_id",
        "player_name",
        "is_shot",
        "shot_proba",
        "xG",
        "goal_proba",
        "is_goal",
    ]
    optional = ["pitch_id", "pitch_length", "pitch_width"]
    available = set(pd.read_csv(predictions_path, nrows=0).columns)
    missing = [col for col in usecols if col not in available]
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {', '.join(missing)}")
    df = pd.read_csv(predictions_path, usecols=[col for col in usecols + optional if col in available])
    df = df[(df["game"].astype(int) == game) & (df["attack_merged"].astype(int) == attack)].copy()
    if df.empty:
        raise ValueError(f"No predictions found for game={game}, attack_merged={attack} in {predictions_path}")
    if df["period"].nunique() != 1:
        raise ValueError("This visualizer expects a single-period attack.")
    df = df.sort_values("periodGameClockTime").reset_index(drop=True)
    df["xG_plus"] = df["goal_proba"]
    df["cum_xG_plus"] = df["xG_plus"].cumsum()
    df["int_sec"] = df["periodGameClockTime"].astype(int)
    return df


def tracking_paths(root: Path, game: int) -> tuple[Path, Path, Path]:
    matches = sorted(root.glob(f"**/{game}/{game}.jsonl.bz2"))
    if not matches:
        raise FileNotFoundError(f"Could not find {game}.jsonl.bz2 under {root}")
    tracking_path = matches[0]
    game_dir = tracking_path.parent
    return tracking_path, game_dir / "metadata.json", game_dir / "rosters.json"


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def hex_to_rgb(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def readable_text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (20, 24, 23) if luminance(rgb) > 150 else (255, 255, 255)


def kit_fill_color(kit: dict[str, Any] | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    kit = kit or {}
    primary = hex_to_rgb(kit.get("primaryColor"))
    secondary = hex_to_rgb(kit.get("secondaryColor"))
    if primary and luminance(primary) <= 225:
        return primary
    if secondary:
        return secondary
    return primary or fallback


def team_styles_from_metadata(metadata: dict[str, Any]) -> dict[str, dict[str, tuple[int, int, int]]]:
    styles: dict[str, dict[str, tuple[int, int, int]]] = {}
    for side, fallback in [("home", COLORS["home"]), ("away", COLORS["away"])]:
        fill = kit_fill_color(metadata.get(f"{side}TeamKit"), fallback)
        styles[side] = {
            "fill": fill,
            "text": readable_text_color(fill),
            "outline": (255, 255, 255) if luminance(fill) < 225 else (35, 35, 35),
        }
    return styles


def player_rows(players: list[dict[str, Any]], team: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for player in players or []:
        x = player.get("x")
        y = player.get("y")
        if x is None or y is None:
            continue
        rows.append(
            {
                "team": team,
                "jerseyNum": player.get("jerseyNum"),
                "player_name": None,
                "x": x,
                "y": y,
            }
        )
    return rows


def ball_row(frame: dict[str, Any]) -> dict[str, Any] | None:
    ball = frame.get("ballsSmoothed")
    if not isinstance(ball, dict) or ball.get("x") is None or ball.get("y") is None:
        balls = frame.get("balls")
        ball = balls[0] if isinstance(balls, list) and balls else None
    if not isinstance(ball, dict) or ball.get("x") is None or ball.get("y") is None:
        return None
    return {
        "team": "ball",
        "jerseyNum": None,
        "player_name": None,
        "x": ball.get("x"),
        "y": ball.get("y"),
    }


def tracking_rows_for_attack(
    tracking_path: Path,
    pff_period: int,
    start_time: float,
    end_time: float,
    pad_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lo = start_time - pad_seconds
    hi = end_time + pad_seconds
    with bz2.open(tracking_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            frame = json.loads(line)
            period = int(frame.get("period", -1))
            clock = float(frame.get("periodGameClockTime", -1))
            if period != pff_period or clock < lo or clock > hi:
                continue
            frame_num = int(frame["frameNum"])
            rows.extend(
                {
                    **row,
                    "frameNum": frame_num,
                    "period": period,
                    "periodGameClockTime": clock,
                }
                for row in player_rows(frame.get("homePlayersSmoothed"), "home")
            )
            rows.extend(
                {
                    **row,
                    "frameNum": frame_num,
                    "period": period,
                    "periodGameClockTime": clock,
                }
                for row in player_rows(frame.get("awayPlayersSmoothed"), "away")
            )
            ball = ball_row(frame)
            if ball:
                rows.append(
                    {
                        **ball,
                        "frameNum": frame_num,
                        "period": period,
                        "periodGameClockTime": clock,
                    }
                )
    if not rows:
        raise ValueError(f"No tracking rows found in {tracking_path} for period={pff_period}, {lo:.3f}-{hi:.3f}s")
    return pd.DataFrame(rows).sort_values(["periodGameClockTime", "team", "jerseyNum"]).reset_index(drop=True)


def selected_frame_times(tracking: pd.DataFrame, max_frames: int | None, stride: int) -> list[tuple[int, float]]:
    frames = (
        tracking[["frameNum", "periodGameClockTime"]]
        .drop_duplicates()
        .sort_values("periodGameClockTime")
        .reset_index(drop=True)
    )
    if stride > 1:
        frames = frames.iloc[::stride].reset_index(drop=True)
    if max_frames and len(frames) > max_frames:
        positions = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        frames = frames.iloc[positions].reset_index(drop=True)
    return [(int(row.frameNum), float(row.periodGameClockTime)) for row in frames.itertuples(index=False)]


def metric_index_for_time(metric: pd.DataFrame, clock: float) -> int:
    times = metric["periodGameClockTime"].to_list()
    index = pd.Series(times).searchsorted(clock, side="right") - 1
    if index < 0:
        return 0
    return min(int(index), len(metric) - 1)


def metric_row_for_time(metric: pd.DataFrame, clock: float, interpolate: bool) -> tuple[pd.Series, int]:
    metric_index = metric_index_for_time(metric, clock)
    row = metric.iloc[metric_index]
    if not interpolate:
        return row, metric_index

    times = metric["periodGameClockTime"].to_list()
    if clock <= times[0] or metric_index >= len(metric) - 1:
        return row, metric_index

    next_row = metric.iloc[metric_index + 1]
    t0 = float(row["periodGameClockTime"])
    t1 = float(next_row["periodGameClockTime"])
    if t1 <= t0:
        return row, metric_index

    interpolated = row.copy()
    weight = (clock - t0) / (t1 - t0)
    for column in ["shot_proba", "xG", "xG_plus"]:
        interpolated[column] = float(row[column]) + weight * (float(next_row[column]) - float(row[column]))
    return interpolated, metric_index


def attack_team_name(row: pd.Series) -> str:
    if int(row["attack_team_id"]) == int(row["home_id"]):
        return str(row["home_name"])
    if int(row["attack_team_id"]) == int(row["away_id"]):
        return str(row["away_name"])
    return f"Team {row['attack_team_id']}"


def render_frame(
    metric: pd.DataFrame,
    tracking_by_frame: dict[int, pd.DataFrame],
    frame_num: int,
    clock: float,
    rendered_index: int,
    args: argparse.Namespace,
    fonts: dict[str, Any],
    team_styles: dict[str, dict[str, tuple[int, int, int]]],
) -> Image.Image:
    width, height = args.width, args.height
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)

    row, metric_index = metric_row_for_time(metric, clock, args.interpolate_metrics)
    title = f"{attack_team_name(row)} attack {int(row['attack_merged'])}"
    subtitle = f"{row['away_name']} at {row['home_name']} | period {int(row['period']) + 1} | {clock:.1f}s"
    draw_text(draw, (34, 22), title, fonts["title"], fill=COLORS["text"])
    draw_text(draw, (36, 58), subtitle, fonts["small"], fill=COLORS["muted"])
    draw_text(draw, (36, 85), f"frame {int(frame_num)}", fonts["small"], fill=COLORS["muted"])

    ticker_y = 18
    ticker_x = width - 34 - (124 * 4 + 12 * 3)
    draw_metric_box(draw, ticker_x, ticker_y, "xS", float(row["shot_proba"]), COLORS["xS"], fonts["tiny"], fonts["metric"])
    draw_metric_box(draw, ticker_x + 136, ticker_y, "xG", float(row["xG"]), COLORS["xG"], fonts["tiny"], fonts["metric"])
    draw_metric_box(draw, ticker_x + 272, ticker_y, "xG+", float(row["xG_plus"]), COLORS["xG+"], fonts["tiny"], fonts["metric"])
    draw_metric_box(draw, ticker_x + 408, ticker_y, "cum xG+", float(row["cum_xG_plus"]), COLORS["cumulative"], fonts["tiny"], fonts["metric"])

    pitch_box = (34, 104, width - 34, 610)
    draw_pitch(draw, pitch_box, args.pitch)
    draw_players(draw, tracking_by_frame.get(frame_num, pd.DataFrame()), pitch_box, fonts["jersey"], team_styles, pitch=args.pitch)

    if not pd.isna(row.get("player_name")):
        draw.rounded_rectangle([48, 622, 330, 650], radius=8, fill=(255, 255, 255), outline=(226, 230, 224), width=1)
        draw_text(draw, (60, 628), f"modeled player: {row['player_name']}", fonts["small"], fill=COLORS["text"])

    draw_sparkline_panel(draw, (34, 620, width - 34, 760), metric, metric_index, fonts["small_bold"])
    draw_cumulative_panel(draw, (34, 768, width - 34, height - 34), metric, metric_index, fonts["small_bold"])

    progress_left, progress_right = 34, width - 34
    progress_y = height - 24
    draw.line([(progress_left, progress_y), (progress_right, progress_y)], fill=(207, 214, 205), width=4)
    progress = rendered_index / max(args.rendered_frame_count - 1, 1)
    draw.line(
        [(progress_left, progress_y), (progress_left + progress * (progress_right - progress_left), progress_y)],
        fill=COLORS["cumulative"],
        width=4,
    )
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an xG+ GIF from PFF tracking and model predictions.")
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions/predictions_pl_2024-2025.csv"))
    parser.add_argument("--tracking-root", type=Path, default=Path("pff-tracking"))
    parser.add_argument("--game", type=int, default=13419)
    parser.add_argument("--attack", type=int, default=178)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pad-seconds", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=240, help="Use 0 for every tracking frame in the attack window.")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--duration-ms", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--interpolate-metrics",
        action="store_true",
        help="Linearly interpolate instantaneous xS, xG, and xG+ between prediction timestamps for display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames == 0:
        args.max_frames = None
    if args.output is None:
        args.output = Path("outputs") / f"pff_game{args.game}_atk{args.attack}.gif"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metric = prediction_rows(args.predictions, args.game, args.attack)
    tracking_path, metadata_path, roster_path = tracking_paths(args.tracking_root, args.game)
    metadata = load_metadata(metadata_path)
    team_styles = team_styles_from_metadata(metadata)
    args.pitch = active_pitch_dimensions(metadata) if metadata else pitch_from_data(metric, pd.DataFrame(), None, None)
    if metadata.get("stadium", {}).get("pitches"):
        print(
            "Using active tracking pitch dimensions from metadata: "
            f"{args.pitch.length} x {args.pitch.width}"
        )
    if metadata.get("homeTeam") and metadata.get("awayTeam"):
        print(
            "Team colors: "
            f"{metadata['homeTeam'].get('shortName', 'home')}={team_styles['home']['fill']} "
            f"{metadata['awayTeam'].get('shortName', 'away')}={team_styles['away']['fill']}"
        )
    print(f"Roster file: {roster_path}" if roster_path.exists() else "Roster file not found.")

    pff_period = int(metric["period"].iloc[0]) + 1
    tracking = tracking_rows_for_attack(
        tracking_path,
        pff_period=pff_period,
        start_time=float(metric["periodGameClockTime"].min()),
        end_time=float(metric["periodGameClockTime"].max()),
        pad_seconds=args.pad_seconds,
    )
    frame_times = selected_frame_times(tracking, args.max_frames, max(args.stride, 1))
    args.rendered_frame_count = len(frame_times)

    tracking_by_frame = {int(frame): rows for frame, rows in tracking.groupby("frameNum")}
    fonts = {
        "title": load_font(26, bold=True),
        "metric": load_font(24, bold=True),
        "small_bold": load_font(15, bold=True),
        "small": load_font(14),
        "tiny": load_font(12),
        "jersey": load_font(10, bold=True),
    }

    images = [
        render_frame(metric, tracking_by_frame, frame_num, clock, rendered_index, args, fonts, team_styles)
        for rendered_index, (frame_num, clock) in enumerate(frame_times)
    ]
    images[0].save(
        args.output,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    seconds = len(images) * args.duration_ms / 1000
    print(f"Wrote {args.output} ({len(images)} frames, {seconds:.1f}s playback)")
    print(f"Predictions: {args.predictions}")
    print(f"Tracking: {tracking_path}")
    print(f"Game {args.game}, attack {args.attack}, period {pff_period}")


if __name__ == "__main__":
    main()
