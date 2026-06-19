#!/usr/bin/env python3
"""Evaluate xG+ prediction tables at frame and attack levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


SHOT_TARGET_CANDIDATES = ["hasShotsIn1s", "hasShotsIn0.5s", "hasShotsIn3s", "hasShotsIn5s", "hasShotsIn10s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate xG+ model predictions.")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025")
    parser.add_argument("--shot-target", default=None)
    return parser.parse_args()


def bool_target(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    truthy = {"true", "1", "yes", "y", "t"}
    return series.fillna("").astype(str).str.lower().isin(truthy)


def safe_metric(fn, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(fn(y_true, y_prob))


def classification_metrics(y: pd.Series, prob: pd.Series) -> dict[str, Any]:
    data = pd.DataFrame({"y": bool_target(y).astype(int), "prob": pd.to_numeric(prob, errors="coerce")})
    data = data.dropna()
    if data.empty:
        return {"rows": 0}
    y_true = data["y"].to_numpy()
    y_prob = data["prob"].clip(0.0, 1.0).to_numpy()
    return {
        "rows": int(len(data)),
        "positive_rows": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "mean_prediction": float(y_prob.mean()),
        "auc": safe_metric(roc_auc_score, y_true, y_prob),
        "average_precision": safe_metric(average_precision_score, y_true, y_prob),
        "log_loss": safe_metric(lambda a, b: log_loss(a, b, labels=[0, 1]), y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions is not None:
        return args.predictions
    return Path("data/predictions") / f"predictions_{args.competition}_{args.season}.csv"


def read_model_shot_target(args: argparse.Namespace) -> str | None:
    metadata_path = args.models_dir / f"model_metadata_{args.competition}_{args.season}.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = metadata.get("shot_target")
    return str(value) if value else None


def pick_shot_target(args: argparse.Namespace, predictions: pd.DataFrame) -> str | None:
    if args.shot_target:
        return args.shot_target
    metadata_target = read_model_shot_target(args)
    if metadata_target and metadata_target in predictions.columns:
        return metadata_target
    for candidate in SHOT_TARGET_CANDIDATES:
        if candidate in predictions.columns:
            return candidate
    return None


def attack_goal_probability(values: pd.Series) -> float:
    probs = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return float(1.0 - np.prod(1.0 - probs))


def build_attack_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if not {"game", "attack_merged"}.issubset(predictions.columns):
        return pd.DataFrame()
    rows = predictions.copy()
    rows["is_shot_bool"] = bool_target(rows["is_shot"]) if "is_shot" in rows.columns else False
    rows["is_goal_bool"] = bool_target(rows["is_goal"]) if "is_goal" in rows.columns else False
    grouped = rows.groupby(["game", "attack_merged"], dropna=False)
    out = grouped.agg(
        rows=("shot_proba", "size"),
        period=("period", "first"),
        start_time=("periodGameClockTime", "min"),
        end_time=("periodGameClockTime", "max"),
        max_shot_proba=("shot_proba", "max"),
        max_xg=("xG", "max"),
        attack_goal_proba=("goal_proba", attack_goal_probability),
        has_shot=("is_shot_bool", "max"),
        has_goal=("is_goal_bool", "max"),
    ).reset_index()
    return out


def main() -> None:
    args = parse_args()
    predictions_path = resolve_predictions_path(args)
    if not predictions_path.exists():
        raise SystemExit(f"Missing predictions file: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shot_target = pick_shot_target(args, predictions)
    metrics: dict[str, Any] = {
        "competition": args.competition,
        "season": args.season,
        "predictions": str(predictions_path),
        "rows": int(len(predictions)),
        "shot_target": shot_target,
    }

    if shot_target and shot_target in predictions.columns:
        metrics["frame_shot_target"] = classification_metrics(predictions[shot_target], predictions["shot_proba"])
    if {"is_shot", "shot_proba"}.issubset(predictions.columns):
        metrics["frame_exact_shot_sanity"] = classification_metrics(predictions["is_shot"], predictions["shot_proba"])
    if {"is_goal", "goal_proba"}.issubset(predictions.columns):
        metrics["frame_goal_xgplus"] = classification_metrics(predictions["is_goal"], predictions["goal_proba"])
    if {"is_shot", "is_goal", "xG"}.issubset(predictions.columns):
        shot_rows = predictions[bool_target(predictions["is_shot"])].copy()
        metrics["shot_goal_xg"] = classification_metrics(shot_rows["is_goal"], shot_rows["xG"]) if not shot_rows.empty else {"rows": 0}

    attack_summary = build_attack_summary(predictions)
    if not attack_summary.empty:
        attack_path = args.output_dir / f"attack_summary_{args.competition}_{args.season}.csv"
        attack_summary.to_csv(attack_path, index=False)
        metrics["attack_summary"] = str(attack_path)
        metrics["attack_shot"] = classification_metrics(attack_summary["has_shot"], attack_summary["max_shot_proba"])
        metrics["attack_goal"] = classification_metrics(attack_summary["has_goal"], attack_summary["attack_goal_proba"])

    metrics_path = args.output_dir / f"evaluation_metrics_{args.competition}_{args.season}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote metrics: {metrics_path}")
    if not attack_summary.empty:
        print(f"Wrote attack summary: {metrics['attack_summary']}")


if __name__ == "__main__":
    main()
