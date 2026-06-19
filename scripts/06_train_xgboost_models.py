#!/usr/bin/env python3
"""Train XGBoost shot and goal models from merged xG+ frame data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


FEATURES = ["r", "theta", "z", "speed", "GK_r", "GK_theta", "openGoal"]
for i in range(5):
    FEATURES += [f"DefDist{i}", f"DefAngle{i}", f"OffDist{i}", f"OffAngle{i}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost xG+ models from merged CSV chunks.")
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged_data"))
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025")
    parser.add_argument("--shot-target", default="hasShotsIn1s")
    parser.add_argument("--goal-target", default="is_goal")
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--early-stopping-rounds", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-shot-rows", type=int, default=100)
    return parser.parse_args()


def import_xgboost():
    try:
        import xgboost as xgb
        from xgboost import DMatrix
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("xgboost is required for this step. Install it before training.") from exc
    return xgb, DMatrix


def read_training_data(merged_dir: Path, competition: str, season: str) -> pd.DataFrame:
    files = sorted(merged_dir.glob(f"train_{competition}_{season}_*.csv"))
    if not files:
        raise SystemExit(f"No merged chunk files found under {merged_dir}")
    df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    if df.empty:
        raise SystemExit("Merged training data is empty.")
    return df


def bool_target(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).to_numpy()
    truthy = {"true", "1", "yes", "y", "t"}
    return series.fillna("").astype(str).str.lower().isin(truthy).astype(int).to_numpy()


def xgb_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": args.device,
        "eval_metric": "logloss",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "seed": args.seed,
    }


def safe_metric(fn, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(fn(y_true, y_prob))


def fit_model(
    df: pd.DataFrame,
    target: str,
    groups: pd.Series,
    args: argparse.Namespace,
    label: str,
):
    xgb, DMatrix = import_xgboost()
    y = bool_target(df[target])
    if len(np.unique(y)) < 2:
        raise SystemExit(f"{label} target {target!r} has fewer than two classes.")

    x = df[FEATURES]
    dtrain = DMatrix(x, label=y, feature_names=FEATURES)
    rounds = args.num_boost_round
    n_groups = groups.nunique(dropna=True)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    if min(args.cv_folds, n_groups, n_pos, n_neg) >= 2:
        n_splits = min(args.cv_folds, n_groups, n_pos, n_neg)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(x, y, groups=groups))
        try:
            cv_results = xgb.cv(
                params=xgb_params(args),
                dtrain=dtrain,
                num_boost_round=args.num_boost_round,
                folds=folds,
                metrics="logloss",
                seed=args.seed,
                early_stopping_rounds=args.early_stopping_rounds,
                as_pandas=True,
                verbose_eval=False,
            )
            rounds = int(cv_results["test-logloss-mean"].idxmin()) + 1
        except Exception as exc:
            print(f"warn: {label} CV failed; using {rounds} rounds: {exc}")
    else:
        print(f"warn: {label} CV skipped; insufficient groups/classes")

    model = xgb.train(params=xgb_params(args), dtrain=dtrain, num_boost_round=rounds)
    prob = model.predict(dtrain)
    metrics = {
        "rows": int(len(df)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(np.mean(y)),
        "rounds": rounds,
        "auc": safe_metric(roc_auc_score, y, prob),
        "average_precision": safe_metric(average_precision_score, y, prob),
        "log_loss": safe_metric(log_loss, y, prob),
    }
    return model, metrics


def main() -> None:
    args = parse_args()
    df = read_training_data(args.merged_dir, args.competition, args.season)
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan
    for target in [args.shot_target, args.goal_target, "is_shot"]:
        if target not in df.columns:
            raise SystemExit(f"Missing required target column: {target}")

    shot_groups = df["game"] if "game" in df.columns else pd.Series(np.arange(len(df)))
    shot_model, shot_metrics = fit_model(df, args.shot_target, shot_groups, args, "shot")

    shot_rows = df[bool_target(df["is_shot"]) == 1].copy()
    if len(shot_rows) < args.min_shot_rows:
        raise SystemExit(f"Only {len(shot_rows)} shot rows found; need at least {args.min_shot_rows}.")
    goal_groups = shot_rows["game"] if "game" in shot_rows.columns else pd.Series(np.arange(len(shot_rows)))
    goal_model, goal_metrics = fit_model(shot_rows, args.goal_target, goal_groups, args, "goal")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    shot_path = args.models_dir / f"shot_model_{args.competition}_{args.season}.json"
    goal_path = args.models_dir / f"xg_model_{args.competition}_{args.season}.json"
    meta_path = args.models_dir / f"model_metadata_{args.competition}_{args.season}.json"
    metrics_path = args.models_dir / f"metrics_{args.competition}_{args.season}.json"

    shot_model.save_model(shot_path)
    goal_model.save_model(goal_path)
    metadata = {
        "competition": args.competition,
        "season": args.season,
        "features": FEATURES,
        "shot_target": args.shot_target,
        "goal_target": args.goal_target,
        "shot_model": str(shot_path),
        "goal_model": str(goal_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps({"shot": shot_metrics, "goal": goal_metrics}, indent=2), encoding="utf-8")

    print(f"Wrote shot model: {shot_path}")
    print(f"Wrote goal model: {goal_path}")
    print(f"Wrote metadata: {meta_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
