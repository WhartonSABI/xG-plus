#!/usr/bin/env python3
"""Train local xG+ models and produce prediction tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["r", "theta", "z", "speed", "GK_r", "GK_theta", "openGoal"]
for i in range(5):
    FEATURES += [f"DefDist{i}", f"DefAngle{i}", f"OffDist{i}", f"OffAngle{i}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xG+ models from merged local CSVs.")
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged_data"))
    parser.add_argument("--predictions-path", type=Path, default=Path("data/predictions/predictions_pl_2024-2025.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025")
    parser.add_argument("--shot-target", default="hasShotsIn5s")
    parser.add_argument("--min-shot-rows", type=int, default=100)
    return parser.parse_args()


def make_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def main() -> None:
    args = parse_args()
    files = sorted(args.merged_dir.glob(f"train_{args.competition}_{args.season}_*.csv"))
    if not files:
        raise SystemExit(f"No merged chunk files found under {args.merged_dir}")

    df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    if df.empty:
        raise SystemExit("Merged dataset is empty.")

    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan
    if args.shot_target not in df.columns:
        raise SystemExit(f"Missing target column: {args.shot_target}")
    if "is_goal" not in df.columns:
        raise SystemExit("Missing target column: is_goal")

    X = df[FEATURES]
    y_shot = df[args.shot_target].astype(int).to_numpy()
    shot_model = make_logistic()
    shot_model.fit(X, y_shot)
    shot_proba = shot_model.predict_proba(X)[:, 1]

    shot_mask = df["is_shot"].fillna(False).astype(bool).to_numpy()
    goal_model = make_logistic()
    if int(shot_mask.sum()) >= args.min_shot_rows:
        X_goal = df.loc[shot_mask, FEATURES]
        y_goal = df.loc[shot_mask, "is_goal"].astype(int).to_numpy()
        if len(np.unique(y_goal)) >= 2:
            goal_model.fit(X_goal, y_goal)
            xg = goal_model.predict_proba(X)[:, 1]
        else:
            xg = np.full(len(df), float(y_goal.mean()) if len(y_goal) else 0.0)
    else:
        xg = np.full(len(df), float(df["is_goal"].mean()) if len(df) else 0.0)

    goal_proba = np.clip(shot_proba * xg, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "game": df["game"],
            "date": df.get("date"),
            "home_id": df.get("home_id"),
            "home_name": df.get("home_name"),
            "away_id": df.get("away_id"),
            "away_name": df.get("away_name"),
            "attack_team_id": df.get("attack_team_id"),
            "attack_merged": df.get("attack_merged"),
            "period": df.get("period"),
            "periodGameClockTime": df.get("periodGameClockTime"),
            "player_id": df.get("player_id"),
            "player_name": df.get("player_name"),
            "is_shot": df["is_shot"].astype(int),
            "shot_proba": shot_proba,
            "xG": xg,
            "goal_proba": goal_proba,
            "is_goal": df["is_goal"].astype(int),
        }
    )

    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_path, index=False)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(shot_model, args.models_dir / f"shot_model_{args.competition}_{args.season}.joblib")
    joblib.dump(goal_model, args.models_dir / f"xg_model_{args.competition}_{args.season}.joblib")

    metrics: dict[str, Any] = {
        "rows": int(len(df)),
        "shot_positive_rate": float(np.mean(y_shot)),
        "goal_positive_rate": float(df["is_goal"].astype(int).mean()),
        "shot_auc": safe_auc(y_shot, shot_proba),
        "shot_ap": safe_ap(y_shot, shot_proba),
    }
    metrics_path = args.models_dir / f"metrics_{args.competition}_{args.season}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote predictions: {args.predictions_path}")
    print(f"Wrote models: {args.models_dir}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
