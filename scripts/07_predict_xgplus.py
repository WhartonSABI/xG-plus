#!/usr/bin/env python3
"""Apply trained XGBoost xG+ models to merged frame data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURES = ["r", "theta", "z", "speed", "GK_r", "GK_theta", "openGoal"]
for i in range(5):
    DEFAULT_FEATURES += [f"DefDist{i}", f"DefAngle{i}", f"OffDist{i}", f"OffAngle{i}"]

SHOT_WINDOWS = ["hasShotsIn10s", "hasShotsIn5s", "hasShotsIn3s", "hasShotsIn1s", "hasShotsIn0.5s"]
CORE_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate xS, xG, and xG+ predictions from XGBoost models.")
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged_data"))
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--shots-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--shot-model-path", type=Path, default=None)
    parser.add_argument("--goal-model-path", type=Path, default=None)
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025")
    parser.add_argument("--limit-rows", type=int, default=0, help="0 means all rows; useful for smoke tests.")
    return parser.parse_args()


def import_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("xgboost is required for prediction. Install it before running this step.") from exc
    return xgb


def read_merged_data(merged_dir: Path, competition: str, season: str, limit_rows: int) -> pd.DataFrame:
    files = sorted(merged_dir.glob(f"train_{competition}_{season}_*.csv"))
    if not files:
        raise SystemExit(f"No merged chunk files found under {merged_dir}")
    frames: list[pd.DataFrame] = []
    remaining = limit_rows
    for path in files:
        if limit_rows > 0 and remaining <= 0:
            break
        if limit_rows > 0:
            df = pd.read_csv(path, nrows=remaining)
            remaining -= len(df)
        else:
            df = pd.read_csv(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise SystemExit("Merged data is empty.")
    return out


def default_paths(args: argparse.Namespace) -> None:
    if args.predictions_path is None:
        args.predictions_path = Path("data/predictions") / f"predictions_{args.competition}_{args.season}.csv"
    if args.shots_path is None:
        args.shots_path = Path("data/predictions") / f"shots_{args.competition}_{args.season}.csv"
    if args.metadata_path is None:
        args.metadata_path = args.models_dir / f"model_metadata_{args.competition}_{args.season}.json"
    if args.shot_model_path is None:
        args.shot_model_path = args.models_dir / f"shot_model_{args.competition}_{args.season}.json"
    if args.goal_model_path is None:
        args.goal_model_path = args.models_dir / f"xg_model_{args.competition}_{args.season}.json"


def read_model_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_path(path: Path | None, metadata_value: Any) -> Path:
    if path is not None:
        return path
    if metadata_value:
        return Path(str(metadata_value))
    raise SystemExit("Model path could not be resolved.")


def bool_target(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    truthy = {"true", "1", "yes", "y", "t"}
    return series.fillna("").astype(str).str.lower().isin(truthy)


def empty_series(length: int) -> pd.Series:
    return pd.Series([pd.NA] * length)


def build_prediction_table(
    df: pd.DataFrame,
    shot_proba: np.ndarray,
    xg: np.ndarray,
    goal_proba: np.ndarray,
    shot_target: str | None,
) -> pd.DataFrame:
    predictions = pd.DataFrame({col: df[col] if col in df.columns else empty_series(len(df)) for col in CORE_COLUMNS})
    predictions["is_shot"] = bool_target(df["is_shot"]).astype(int) if "is_shot" in df.columns else 0
    predictions["shot_proba"] = shot_proba
    predictions["xG"] = xg
    predictions["goal_proba"] = goal_proba
    predictions["is_goal"] = bool_target(df["is_goal"]).astype(int) if "is_goal" in df.columns else 0

    extras = ["competition", "season", "attack", "frameNum", "videoTimeMs"]
    extras += [col for col in SHOT_WINDOWS if col in df.columns]
    if shot_target and shot_target in df.columns and shot_target not in extras:
        extras.append(shot_target)
    for col in extras:
        if col in df.columns and col not in predictions.columns:
            predictions[col] = df[col]
    return predictions


def main() -> None:
    args = parse_args()
    default_paths(args)
    metadata = read_model_metadata(args.metadata_path)
    features = list(metadata.get("features") or DEFAULT_FEATURES)
    shot_target = metadata.get("shot_target")

    shot_model_path = model_path(args.shot_model_path, metadata.get("shot_model"))
    goal_model_path = model_path(args.goal_model_path, metadata.get("goal_model"))
    if not shot_model_path.exists():
        raise SystemExit(f"Missing shot model: {shot_model_path}")
    if not goal_model_path.exists():
        raise SystemExit(f"Missing goal model: {goal_model_path}")

    df = read_merged_data(args.merged_dir, args.competition, args.season, args.limit_rows)
    for feature in features:
        if feature not in df.columns:
            df[feature] = np.nan
    x = df[features].apply(pd.to_numeric, errors="coerce")

    xgb = import_xgboost()
    dmatrix = xgb.DMatrix(x, feature_names=features)
    shot_model = xgb.Booster()
    shot_model.load_model(str(shot_model_path))
    goal_model = xgb.Booster()
    goal_model.load_model(str(goal_model_path))

    shot_proba = shot_model.predict(dmatrix)
    xg = goal_model.predict(dmatrix)
    goal_proba = np.clip(shot_proba * xg, 0.0, 1.0)

    predictions = build_prediction_table(df, shot_proba, xg, goal_proba, shot_target)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_path, index=False)

    shots = predictions[bool_target(predictions["is_shot"])].copy()
    args.shots_path.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(args.shots_path, index=False)

    prediction_meta = {
        "competition": args.competition,
        "season": args.season,
        "rows": int(len(predictions)),
        "shot_rows": int(len(shots)),
        "features": features,
        "shot_target": shot_target,
        "shot_model": str(shot_model_path),
        "goal_model": str(goal_model_path),
        "predictions": str(args.predictions_path),
        "shots": str(args.shots_path),
    }
    meta_path = args.predictions_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(prediction_meta, indent=2), encoding="utf-8")

    print(f"Wrote predictions: {args.predictions_path} ({len(predictions)} rows)")
    print(f"Wrote shot rows: {args.shots_path} ({len(shots)} rows)")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
