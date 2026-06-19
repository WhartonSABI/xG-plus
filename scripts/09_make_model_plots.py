#!/usr/bin/env python3
"""Create model diagnostic plots from xG+ predictions and XGBoost models."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


SHOT_TARGET_CANDIDATES = ["hasShotsIn1s", "hasShotsIn0.5s", "hasShotsIn3s", "hasShotsIn5s", "hasShotsIn10s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create xG+ model diagnostics.")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/plots"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025")
    parser.add_argument("--model-id", default=None, help="Model label used for metadata and importance plots; defaults to --season.")
    parser.add_argument("--shot-target", default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--top-features", type=int, default=20)
    return parser.parse_args()


def import_matplotlib():
    try:
        cache_dir = Path(tempfile.gettempdir()) / "xgplus-matplotlib-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required for plot generation.") from exc
    return plt


def bool_target(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    truthy = {"true", "1", "yes", "y", "t"}
    return series.fillna("").astype(str).str.lower().isin(truthy)


def resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions is not None:
        return args.predictions
    return Path("data/predictions") / f"predictions_{args.competition}_{args.season}.csv"


def read_metadata(args: argparse.Namespace) -> dict[str, Any]:
    model_label = args.model_id or args.season
    path = args.models_dir / f"model_metadata_{args.competition}_{model_label}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pick_shot_target(args: argparse.Namespace, predictions: pd.DataFrame, metadata: dict[str, Any]) -> str | None:
    if args.shot_target:
        return args.shot_target
    metadata_target = metadata.get("shot_target")
    if metadata_target and metadata_target in predictions.columns:
        return str(metadata_target)
    for candidate in SHOT_TARGET_CANDIDATES:
        if candidate in predictions.columns:
            return candidate
    return None


def calibration_plot(plt, data: pd.DataFrame, target: str, prob: str, title: str, output: Path, bins: int) -> None:
    frame = pd.DataFrame(
        {
            "target": bool_target(data[target]).astype(int),
            "prob": pd.to_numeric(data[prob], errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return
    frame["bin"] = pd.cut(frame["prob"].clip(0.0, 1.0), bins=bins, labels=False, include_lowest=True)
    grouped = frame.groupby("bin", dropna=True).agg(actual=("target", "mean"), predicted=("prob", "mean"), rows=("target", "size"))
    if grouped.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=1, linestyle="--")
    ax.plot(grouped["predicted"], grouped["actual"], marker="o", color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_xlim(0, max(0.01, min(1.0, frame["prob"].quantile(0.995) * 1.2)))
    ax.set_ylim(0, max(0.01, min(1.0, grouped["actual"].max() * 1.2)))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def distribution_plot(plt, predictions: pd.DataFrame, output: Path) -> None:
    columns = [col for col in ["shot_proba", "xG", "goal_proba"] if col in predictions.columns]
    if not columns:
        return
    fig, axes = plt.subplots(len(columns), 1, figsize=(7, 2.6 * len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        values = pd.to_numeric(predictions[column], errors="coerce").dropna().clip(0.0, 1.0)
        ax.hist(values, bins=50, color="#2c7fb8", alpha=0.85)
        ax.set_title(column)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Rows")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def importance_plot(plt, model_path: Path, output: Path, title: str, top_n: int) -> bool:
    if not model_path.exists():
        return False
    try:
        import xgboost as xgb
    except ImportError:
        return False
    model = xgb.Booster()
    model.load_model(str(model_path))
    scores = model.get_score(importance_type="gain")
    if not scores:
        return False
    data = (
        pd.Series(scores, name="gain")
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(data))))
    data.plot.barh(ax=ax, color="#4c956c")
    ax.set_title(title)
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    predictions_path = resolve_predictions_path(args)
    if not predictions_path.exists():
        raise SystemExit(f"Missing predictions file: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    metadata = read_metadata(args)
    shot_target = pick_shot_target(args, predictions, metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = import_matplotlib()
    written: list[Path] = []

    dist_path = args.output_dir / f"prediction_distributions_{args.competition}_{args.season}.png"
    distribution_plot(plt, predictions, dist_path)
    if dist_path.exists():
        written.append(dist_path)

    if shot_target and shot_target in predictions.columns and "shot_proba" in predictions.columns:
        path = args.output_dir / f"calibration_shot_{args.competition}_{args.season}.png"
        calibration_plot(plt, predictions, shot_target, "shot_proba", f"Shot calibration ({shot_target})", path, args.bins)
        if path.exists():
            written.append(path)
    if {"is_goal", "goal_proba"}.issubset(predictions.columns):
        path = args.output_dir / f"calibration_goal_{args.competition}_{args.season}.png"
        calibration_plot(plt, predictions, "is_goal", "goal_proba", "Goal calibration (xG+)", path, args.bins)
        if path.exists():
            written.append(path)

    model_label = args.model_id or args.season
    model_paths = {
        "shot": Path(str(metadata.get("shot_model") or args.models_dir / f"shot_model_{args.competition}_{model_label}.json")),
        "goal": Path(str(metadata.get("goal_model") or args.models_dir / f"xg_model_{args.competition}_{model_label}.json")),
    }
    for label, model_path in model_paths.items():
        path = args.output_dir / f"importance_{label}_{args.competition}_{args.season}.png"
        if importance_plot(plt, model_path, path, f"{label.title()} model feature importance", args.top_features):
            written.append(path)

    print("Wrote plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
