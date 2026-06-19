#!/usr/bin/env python3
"""Create a GIF visualization for one xG+ sample attack.

The script expects the sample-data pairing used in this repository:

    sample-data/_______atk486.csv
    sample-data/_______atk486_tracking.csv

The metric CSV contains one row per modeled frame. The tracking CSV contains
player/ball x-y locations by frame. By default, cumulative xG+ uses a running
sum of the maximum xG+ value within each integer second, which avoids counting
near-identical 30 fps predictions 30 times.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pitch_geometry import (
    CENTER_CIRCLE_RADIUS,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    PENALTY_AREA_DEPTH,
    PENALTY_AREA_WIDTH,
    PENALTY_SPOT_DISTANCE,
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
    PitchDimensions,
)


DEFAULT_PITCH = PitchDimensions(DEFAULT_PITCH_LENGTH, DEFAULT_PITCH_WIDTH)

COLORS = {
    "bg": (244, 246, 241),
    "panel": (253, 253, 249),
    "pitch": (58, 126, 88),
    "pitch_alt": (52, 116, 82),
    "line": (236, 244, 235),
    "text": (28, 34, 32),
    "muted": (104, 113, 108),
    "grid": (218, 224, 216),
    "home": (30, 74, 176),
    "away": (218, 69, 55),
    "ball": (250, 250, 250),
    "xS": (24, 126, 188),
    "xG": (220, 134, 34),
    "xG+": (82, 160, 91),
    "cumulative": (45, 92, 78),
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def metric_file_from_attack(data_dir: Path, attack: str) -> Path:
    path = data_dir / f"{attack}.csv"
    if path.exists():
        return path
    matches = sorted(data_dir.glob(f"*{attack}*.csv"))
    matches = [match for match in matches if not match.name.endswith("_tracking.csv")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No metric CSV found for attack '{attack}' in {data_dir}")
    names = ", ".join(match.name for match in matches)
    raise ValueError(f"Attack '{attack}' is ambiguous. Matches: {names}")


def tracking_file_for(metric_file: Path) -> Path:
    path = metric_file.with_name(f"{metric_file.stem}_tracking.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing tracking CSV next to {metric_file}: {path}")
    return path


def select_frames(metric: pd.DataFrame, max_frames: int | None, stride: int) -> list[int]:
    frames = metric["frameNum"].astype(int).tolist()
    if stride > 1:
        frames = frames[::stride]
    if max_frames and len(frames) > max_frames:
        positions = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        frames = [frames[pos] for pos in positions]
    return frames


def build_cumulative(metric: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "frame-sum":
        return metric["xG_plus"].cumsum()

    per_second = metric.loc[metric.groupby("int_sec")["xG_plus"].idxmax(), ["int_sec", "xG_plus"]]
    second_to_cum = per_second.sort_values("int_sec").set_index("int_sec")["xG_plus"].cumsum()
    return metric["int_sec"].map(second_to_cum).ffill().fillna(0.0)


def first_numeric_value(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def pitch_from_data(
    metric: pd.DataFrame,
    tracking: pd.DataFrame,
    length_override: float | None,
    width_override: float | None,
) -> PitchDimensions:
    length = length_override or first_numeric_value(metric, "pitch_length") or first_numeric_value(tracking, "pitch_length")
    width = width_override or first_numeric_value(metric, "pitch_width") or first_numeric_value(tracking, "pitch_width")
    return PitchDimensions(length or DEFAULT_PITCH_LENGTH, width or DEFAULT_PITCH_WIDTH)


def xy_to_pitch_px(x: float, y: float, box: tuple[int, int, int, int], pitch: PitchDimensions = DEFAULT_PITCH) -> tuple[int, int]:
    left, top, right, bottom = box
    px = left + (x - pitch.x_min) / pitch.length * (right - left)
    py = bottom - (y - pitch.y_min) / pitch.width * (bottom - top)
    return int(round(px)), int(round(py))


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=COLORS["text"]) -> None:
    draw.text(xy, text, font=font, fill=fill)


def draw_pitch(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    pitch: PitchDimensions = DEFAULT_PITCH,
) -> None:
    left, top, right, bottom = box
    stripe_count = 10
    stripe_w = (right - left) / stripe_count
    for i in range(stripe_count):
        x0 = left + i * stripe_w
        x1 = left + (i + 1) * stripe_w
        draw.rectangle([x0, top, x1, bottom], fill=COLORS["pitch"] if i % 2 == 0 else COLORS["pitch_alt"])

    line = COLORS["line"]
    draw.rectangle(box, outline=line, width=3)
    mid_x = (left + right) // 2
    mid_y = (top + bottom) // 2
    draw.line([(mid_x, top), (mid_x, bottom)], fill=line, width=2)
    circle_rx = CENTER_CIRCLE_RADIUS / pitch.length * (right - left)
    circle_ry = CENTER_CIRCLE_RADIUS / pitch.width * (bottom - top)
    draw.ellipse([mid_x - circle_rx, mid_y - circle_ry, mid_x + circle_rx, mid_y + circle_ry], outline=line, width=2)
    draw.ellipse([mid_x - 4, mid_y - 4, mid_x + 4, mid_y + 4], fill=line)

    # Penalty boxes, six-yard boxes, and spots.
    for side in ["left", "right"]:
        goal_x = left if side == "left" else right
        sign = 1 if side == "left" else -1
        pen_x = goal_x + sign * (PENALTY_AREA_DEPTH / pitch.length) * (right - left)
        six_x = goal_x + sign * (GOAL_AREA_DEPTH / pitch.length) * (right - left)
        pen_y0 = mid_y - (PENALTY_AREA_WIDTH / pitch.width) * (bottom - top) / 2
        pen_y1 = mid_y + (PENALTY_AREA_WIDTH / pitch.width) * (bottom - top) / 2
        six_y0 = mid_y - (GOAL_AREA_WIDTH / pitch.width) * (bottom - top) / 2
        six_y1 = mid_y + (GOAL_AREA_WIDTH / pitch.width) * (bottom - top) / 2
        draw.rectangle([min(goal_x, pen_x), pen_y0, max(goal_x, pen_x), pen_y1], outline=line, width=2)
        draw.rectangle([min(goal_x, six_x), six_y0, max(goal_x, six_x), six_y1], outline=line, width=2)
        spot_x = goal_x + sign * (PENALTY_SPOT_DISTANCE / pitch.length) * (right - left)
        draw.ellipse([spot_x - 3, mid_y - 3, spot_x + 3, mid_y + 3], fill=line)


def draw_players(
    draw: ImageDraw.ImageDraw,
    tracking_rows: pd.DataFrame,
    pitch_box: tuple[int, int, int, int],
    font_small,
    team_styles: dict[str, dict[str, tuple[int, int, int]]] | None = None,
    pitch: PitchDimensions = DEFAULT_PITCH,
) -> None:
    team_styles = team_styles or {}
    for _, row in tracking_rows.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            continue
        x, y = xy_to_pitch_px(float(row["x"]), float(row["y"]), pitch_box, pitch)
        team = str(row["team"])
        if team == "ball":
            radius = 3
            draw.ellipse([x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1], fill=(35, 35, 35))
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=COLORS["ball"])
            continue

        style = team_styles.get(team, {})
        color = style.get("fill", COLORS["home"] if team == "home" else COLORS["away"])
        text_color = style.get("text", (255, 255, 255))
        outline_color = style.get("outline", (255, 255, 255))
        radius = 8
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=outline_color, width=2)
        jersey = row.get("jerseyNum")
        if not pd.isna(jersey):
            label = str(int(float(jersey)))
            tw = draw.textlength(label, font=font_small)
            draw.text((x - tw / 2, y - 5), label, font=font_small, fill=text_color)


def draw_metric_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    value: float,
    color: tuple[int, int, int],
    font_label,
    font_value,
) -> None:
    draw.rounded_rectangle([x, y, x + 124, y + 66], radius=8, fill=COLORS["panel"], outline=(226, 230, 224), width=1)
    draw.rectangle([x, y, x + 7, y + 66], fill=color)
    draw_text(draw, (x + 16, y + 9), label, font_label, fill=COLORS["muted"])
    draw_text(draw, (x + 16, y + 30), f"{value:.3f}", font_value, fill=COLORS["text"])


def line_points(
    values: Iterable[float],
    box: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    total_count: int | None = None,
) -> list[tuple[int, int]]:
    vals = list(values)
    left, top, right, bottom = box
    if not vals:
        return []
    total_count = total_count or len(vals)
    denom = max(y_max - y_min, 1e-9)
    points = []
    for i, value in enumerate(vals):
        x = left if total_count <= 1 else left + i * (right - left) / (total_count - 1)
        y = bottom - (value - y_min) / denom * (bottom - top)
        points.append((int(round(x)), int(round(y))))
    return points


def draw_sparkline_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    metric: pd.DataFrame,
    index: int,
    font_label,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=8, fill=COLORS["panel"], outline=(226, 230, 224), width=1)
    draw_text(draw, (left + 16, top + 12), "Instantaneous", font_label, fill=COLORS["text"])

    rows = [
        ("xS", "shot_proba", COLORS["xS"]),
        ("xG", "xG", COLORS["xG"]),
        ("xG+", "xG_plus", COLORS["xG+"]),
    ]
    y_max = max(max(float(metric[column].max()), 0.0) for _, column, _ in rows)
    tick_step = 0.05
    y_max = max(tick_step, math.ceil(y_max / tick_step) * tick_step)
    plot_box = (left + 54, top + 58, right - 18, bottom - 50)
    tick_count = int(round(y_max / tick_step))
    for tick in range(tick_count + 1):
        value = tick * tick_step
        y = int(plot_box[3] - (value / y_max) * (plot_box[3] - plot_box[1]))
        draw.line([(plot_box[0], y), (plot_box[2], y)], fill=COLORS["grid"], width=1)
        label = "0" if tick == 0 else f"{value:.2f}"
        draw_text(draw, (left + 15, y - 7), label, font_label, fill=COLORS["muted"])
    draw.line([(plot_box[0], plot_box[1]), (plot_box[0], plot_box[3])], fill=COLORS["grid"], width=1)

    for label, column, color in rows:
        pts = line_points(metric[column].iloc[: index + 1], plot_box, 0, y_max, total_count=len(metric))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
        elif pts:
            x, y = pts[0]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)

    legend_y = bottom - 34
    legend_x = left + 18
    for label, _, color in rows:
        draw.line([(legend_x, legend_y + 7), (legend_x + 20, legend_y + 7)], fill=color, width=4)
        draw_text(draw, (legend_x + 26, legend_y), label, font_label, fill=COLORS["text"])
        legend_x += 78


def draw_cumulative_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    metric: pd.DataFrame,
    index: int,
    font_label,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=8, fill=COLORS["panel"], outline=(226, 230, 224), width=1)
    draw_text(draw, (left + 16, top + 12), "Cumulative xG+", font_label, fill=COLORS["text"])
    plot_box = (left + 54, top + 46, right - 18, bottom - 34)
    tick_step = 0.05
    max_value = max(float(metric["cum_xG_plus"].max()), 0.0)
    y_max = max(tick_step, math.ceil(max_value / tick_step) * tick_step)
    tick_count = int(round(y_max / tick_step))
    for tick in range(tick_count + 1):
        value = tick * tick_step
        y = int(plot_box[3] - (value / y_max) * (plot_box[3] - plot_box[1]))
        draw.line([(plot_box[0], y), (plot_box[2], y)], fill=COLORS["grid"], width=1)
        label = "0" if tick == 0 else f"{value:.2f}"
        draw_text(draw, (left + 15, y - 7), label, font_label, fill=COLORS["muted"])
    draw.line([(plot_box[0], plot_box[1]), (plot_box[0], plot_box[3])], fill=COLORS["grid"], width=1)
    pts = line_points(metric["cum_xG_plus"].iloc[: index + 1], plot_box, 0, y_max, total_count=len(metric))
    if len(pts) > 1:
        draw.line(pts, fill=COLORS["cumulative"], width=4)
    elif pts:
        x, y = pts[0]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=COLORS["cumulative"])
    draw_text(draw, (left + 16, bottom - 24), f"final {max_value:.3f}", font_label, fill=COLORS["muted"])


def render_frame(
    metric: pd.DataFrame,
    tracking_by_frame: dict[int, pd.DataFrame],
    frame_num: int,
    rendered_index: int,
    frame_to_index: dict[int, int],
    args: argparse.Namespace,
    fonts: dict[str, ImageFont.ImageFont],
) -> Image.Image:
    width, height = args.width, args.height
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)

    metric_index = frame_to_index[frame_num]
    row = metric.iloc[metric_index]
    title = f"{row['attack_team_name']} attack {int(row['attack_merged'])}"
    subtitle = f"{row['away_name']} at {row['home_name']} | period {int(row['period'])} | {row['periodGameClockTime']:.1f}s"
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
    pitch = getattr(args, "pitch", DEFAULT_PITCH)
    draw_pitch(draw, pitch_box, pitch)
    tracking_rows = tracking_by_frame.get(frame_num, pd.DataFrame())
    draw_players(draw, tracking_rows, pitch_box, fonts["jersey"], pitch=pitch)

    if not pd.isna(row.get("player_name")):
        draw.rounded_rectangle([48, 622, 330, 650], radius=8, fill=(255, 255, 255), outline=(226, 230, 224), width=1)
        draw_text(draw, (60, 628), f"modeled player: {row['player_name']}", fonts["small"], fill=COLORS["text"])

    draw_sparkline_panel(draw, (34, 620, width - 34, 760), metric, metric_index, fonts["small_bold"])
    draw_cumulative_panel(draw, (34, 768, width - 34, height - 34), metric, metric_index, fonts["small_bold"])

    progress_left, progress_right = 34, width - 34
    progress_y = height - 24
    draw.line([(progress_left, progress_y), (progress_right, progress_y)], fill=(207, 214, 205), width=4)
    if args.max_frames == 1:
        progress = 1.0
    else:
        progress = rendered_index / max(args.rendered_frame_count - 1, 1)
    draw.line([(progress_left, progress_y), (progress_left + progress * (progress_right - progress_left), progress_y)], fill=COLORS["cumulative"], width=4)

    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an xG+ tracking GIF from sample-data CSVs.")
    parser.add_argument("--data-dir", type=Path, default=Path("sample-data"), help="Directory containing paired sample CSVs.")
    parser.add_argument("--attack", default="atk486", help="Attack id or substring, e.g. atk486.")
    parser.add_argument("--output", type=Path, default=None, help="Output GIF path.")
    parser.add_argument("--max-frames", type=int, default=240, help="Evenly sample down to this many GIF frames. Use 0 for all frames.")
    parser.add_argument("--stride", type=int, default=1, help="Take every Nth modeled frame before max-frame sampling.")
    parser.add_argument("--duration-ms", type=int, default=55, help="GIF duration per rendered frame.")
    parser.add_argument("--cumulative-mode", choices=["second-max", "frame-sum"], default="second-max")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--pitch-length", type=float, default=None)
    parser.add_argument("--pitch-width", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames == 0:
        args.max_frames = None

    metric_path = metric_file_from_attack(args.data_dir, args.attack)
    tracking_path = tracking_file_for(metric_path)
    if args.output is None:
        args.output = Path("outputs") / f"{metric_path.stem}.gif"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metric = pd.read_csv(metric_path).sort_values("frameNum").reset_index(drop=True)
    tracking = pd.read_csv(tracking_path).sort_values("frameNum")
    args.pitch = pitch_from_data(metric, tracking, args.pitch_length, args.pitch_width)
    metric["frameNum"] = metric["frameNum"].astype(int)
    tracking["frameNum"] = tracking["frameNum"].astype(int)
    metric["int_sec"] = metric.get("int_sec", metric["periodGameClockTime"].astype(int)).astype(int)
    metric["xG_plus"] = metric.get("goal_proba", metric["xG"] * metric["shot_proba"])
    metric["cum_xG_plus"] = build_cumulative(metric, args.cumulative_mode)

    frames = select_frames(metric, args.max_frames, max(args.stride, 1))
    if not frames:
        raise ValueError(f"No frames selected from {metric_path}")
    args.rendered_frame_count = len(frames)

    tracking_by_frame = {int(frame): rows for frame, rows in tracking.groupby("frameNum")}
    frame_to_index = {int(frame): i for i, frame in enumerate(metric["frameNum"])}
    fonts = {
        "title": load_font(26, bold=True),
        "metric": load_font(24, bold=True),
        "small_bold": load_font(15, bold=True),
        "small": load_font(14),
        "tiny": load_font(12),
        "jersey": load_font(10, bold=True),
    }

    images: list[Image.Image] = []
    for rendered_index, frame_num in enumerate(frames):
        images.append(render_frame(metric, tracking_by_frame, frame_num, rendered_index, frame_to_index, args, fonts))

    first, rest = images[0], images[1:]
    first.save(
        args.output,
        save_all=True,
        append_images=rest,
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    seconds = len(frames) * args.duration_ms / 1000
    print(f"Wrote {args.output} ({len(frames)} frames, {seconds:.1f}s playback)")
    print(f"Metric CSV: {metric_path}")
    print(f"Tracking CSV: {tracking_path}")
    print(f"Cumulative mode: {args.cumulative_mode}")


if __name__ == "__main__":
    main()
