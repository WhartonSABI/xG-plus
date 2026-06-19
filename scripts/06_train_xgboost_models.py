#!/usr/bin/env python3
"""Train XGBoost shot and goal models from merged xG+ frame data."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


FEATURES = ["r", "theta", "z", "speed", "GK_r", "GK_theta", "openGoal"]
for i in range(5):
    FEATURES += [f"DefDist{i}", f"DefAngle{i}", f"OffDist{i}", f"OffAngle{i}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost xG+ models from merged CSV chunks.")
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged_data"))
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025", help="Single training season when --train-seasons is not set.")
    parser.add_argument("--train-seasons", nargs="+", default=None, help="One or more seasons to train on.")
    parser.add_argument("--model-id", default=None, help="Output model label. Defaults to the joined training seasons.")
    parser.add_argument("--shot-target", default="hasShotsIn1s")
    parser.add_argument("--goal-target", default="is_goal")
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-group", choices=["matchday", "game"], default="matchday")
    parser.add_argument("--games-per-matchday", type=int, default=10)
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--hyperparameter-trials", type=int, default=0, help="0 means tune rounds only.")
    parser.add_argument("--learning-rate-grid", default="0.03,0.05,0.08")
    parser.add_argument("--max-depth-grid", default="3,5,7")
    parser.add_argument("--min-child-weight-grid", default="1,5,20")
    parser.add_argument("--subsample-grid", default="0.75,0.9,1.0")
    parser.add_argument("--colsample-bytree-grid", default="0.75,0.9,1.0")
    parser.add_argument("--reg-lambda-grid", default="1,5,10")
    parser.add_argument("--reg-alpha-grid", default="0,0.1")
    parser.add_argument("--gamma-grid", default="0,1")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--nthread", type=int, default=0, help="0 lets XGBoost choose.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-shot-rows", type=int, default=100)
    parser.add_argument("--limit-rows-per-season", type=int, default=0, help="Smoke-test limit; 0 means all rows.")
    return parser.parse_args()


def import_xgboost():
    try:
        import xgboost as xgb
        from xgboost import DMatrix
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("xgboost is required for this step. Install it before training.") from exc
    return xgb, DMatrix


def train_seasons(args: argparse.Namespace) -> list[str]:
    return args.train_seasons or [args.season]


def model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    return "_".join(train_seasons(args))


def read_training_data(args: argparse.Namespace) -> pd.DataFrame:
    seasons = train_seasons(args)
    usecols = sorted(set(FEATURES + [args.shot_target, args.goal_target, "is_shot", "game", "date", "season"]))
    frames: list[pd.DataFrame] = []
    for season in seasons:
        files = sorted(args.merged_dir.glob(f"train_{args.competition}_{season}_*.csv"))
        if not files:
            raise SystemExit(f"No merged chunk files found for {args.competition} {season} under {args.merged_dir}")
        remaining = args.limit_rows_per_season
        for path in tqdm(files, desc=f"Read {season} chunks", unit="chunk"):
            if args.limit_rows_per_season > 0 and remaining <= 0:
                break
            nrows = remaining if args.limit_rows_per_season > 0 else None
            df = pd.read_csv(path, usecols=lambda col: col in usecols, nrows=nrows)
            missing = [col for col in usecols if col not in df.columns]
            if missing:
                raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")
            frames.append(df)
            if args.limit_rows_per_season > 0:
                remaining -= len(df)
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise SystemExit("Merged training data is empty.")
    return out


def bool_target(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).to_numpy()
    truthy = {"true", "1", "yes", "y", "t"}
    return series.fillna("").astype(str).str.lower().isin(truthy).astype(int).to_numpy()


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan
    return df[FEATURES].apply(pd.to_numeric, errors="coerce")


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def base_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": args.device,
        "eval_metric": "logloss",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
        "gamma": args.gamma,
        "seed": args.seed,
    }
    if args.nthread > 0:
        params["nthread"] = args.nthread
    return params


def param_signature(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    ignored = {"objective", "tree_method", "device", "eval_metric", "seed", "nthread"}
    return tuple(sorted((key, value) for key, value in params.items() if key not in ignored))


def parameter_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = base_params(args)
    if args.hyperparameter_trials <= 0:
        return [base]

    grid = {
        "eta": parse_float_grid(args.learning_rate_grid),
        "max_depth": parse_int_grid(args.max_depth_grid),
        "min_child_weight": parse_float_grid(args.min_child_weight_grid),
        "subsample": parse_float_grid(args.subsample_grid),
        "colsample_bytree": parse_float_grid(args.colsample_bytree_grid),
        "lambda": parse_float_grid(args.reg_lambda_grid),
        "alpha": parse_float_grid(args.reg_alpha_grid),
        "gamma": parse_float_grid(args.gamma_grid),
    }
    all_overrides = [dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_overrides)

    candidates = [base]
    seen = {param_signature(base)}
    for overrides in all_overrides:
        params = {**base, **overrides}
        signature = param_signature(params)
        if signature in seen:
            continue
        candidates.append(params)
        seen.add(signature)
        if len(candidates) >= args.hyperparameter_trials:
            break
    return candidates


def infer_matchday_groups(df: pd.DataFrame, games_per_matchday: int) -> pd.Series:
    if games_per_matchday <= 0:
        raise SystemExit("--games-per-matchday must be positive.")
    schedule = df[["season", "game", "date"]].drop_duplicates().copy()
    schedule["date_sort"] = pd.to_datetime(schedule["date"], errors="coerce")
    schedule["game_sort"] = pd.to_numeric(schedule["game"], errors="coerce")
    schedule = schedule.sort_values(["season", "date_sort", "game_sort", "game"], kind="mergesort")
    schedule["matchday"] = schedule.groupby("season").cumcount() // games_per_matchday + 1
    lookup = schedule.set_index(["season", "game"])["matchday"]
    keys = pd.MultiIndex.from_frame(df[["season", "game"]])
    matchdays = lookup.reindex(keys)
    if matchdays.isna().any():
        raise SystemExit("Could not infer matchday for every row.")
    return df["season"].astype(str) + "_MW" + matchdays.astype(int).astype(str).str.zfill(2).to_numpy()


def cv_groups(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.cv_group == "game":
        return df["season"].astype(str) + "_G" + df["game"].astype(str)
    return infer_matchday_groups(df, args.games_per_matchday)


def safe_metric(fn, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(fn(y_true, y_prob))


def build_folds(x: pd.DataFrame, y: np.ndarray, groups: pd.Series, args: argparse.Namespace):
    n_groups = groups.nunique(dropna=True)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    n_splits = min(args.cv_folds, n_groups, n_pos, n_neg)
    if n_splits < 2:
        return []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    return list(splitter.split(x, y, groups=groups))


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

    x = numeric_features(df)
    dtrain = DMatrix(x, label=y, feature_names=FEATURES)
    folds = build_folds(x, y, groups, args)
    candidates = parameter_candidates(args)
    cv_records: list[dict[str, Any]] = []

    if folds:
        print(
            f"{label}: evaluating {len(candidates)} parameter set(s) with "
            f"{len(folds)} {args.cv_group}-grouped folds",
            flush=True,
        )
        iterator = tqdm(candidates, desc=f"{label} CV trials", unit="trial")
        for idx, params in enumerate(iterator, start=1):
            try:
                cv_results = xgb.cv(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=args.num_boost_round,
                    folds=folds,
                    metrics="logloss",
                    seed=args.seed,
                    early_stopping_rounds=args.early_stopping_rounds,
                    as_pandas=True,
                    verbose_eval=False,
                )
                best_idx = int(cv_results["test-logloss-mean"].idxmin())
                record = {
                    "trial": idx,
                    "rounds": best_idx + 1,
                    "test_logloss_mean": float(cv_results.loc[best_idx, "test-logloss-mean"]),
                    "test_logloss_std": float(cv_results.loc[best_idx, "test-logloss-std"]),
                    "train_logloss_mean": float(cv_results.loc[best_idx, "train-logloss-mean"]),
                    "params": params,
                }
                iterator.set_postfix(
                    logloss=f"{record['test_logloss_mean']:.5f}",
                    rounds=record["rounds"],
                )
            except Exception as exc:
                record = {"trial": idx, "rounds": args.num_boost_round, "error": str(exc), "params": params}
                print(f"warn: {label} CV trial {idx} failed: {exc}")
                iterator.set_postfix(error="1")
            cv_records.append(record)
    else:
        print(f"warn: {label} CV skipped; insufficient {args.cv_group} groups/classes")

    valid_records = [item for item in cv_records if "test_logloss_mean" in item]
    if valid_records:
        selected = min(valid_records, key=lambda item: item["test_logloss_mean"])
        selected_params = selected["params"]
        rounds = int(selected["rounds"])
    else:
        selected_params = candidates[0]
        rounds = args.num_boost_round

    print(f"{label}: fitting final model with {rounds} boosting rounds", flush=True)
    model = xgb.train(params=selected_params, dtrain=dtrain, num_boost_round=rounds)
    prob = model.predict(dtrain)
    metrics = {
        "rows": int(len(df)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(np.mean(y)),
        "cv_group": args.cv_group,
        "cv_groups": int(groups.nunique(dropna=True)),
        "cv_folds": int(len(folds)),
        "rounds": rounds,
        "selected_params": selected_params,
        "cv_trials": cv_records,
        "train_auc": safe_metric(roc_auc_score, y, prob),
        "train_average_precision": safe_metric(average_precision_score, y, prob),
        "train_log_loss": safe_metric(lambda a, b: log_loss(a, b, labels=[0, 1]), y, prob),
    }
    return model, metrics


def main() -> None:
    args = parse_args()
    seasons = train_seasons(args)
    label = model_id(args)
    df = read_training_data(args)
    for target in [args.shot_target, args.goal_target, "is_shot"]:
        if target not in df.columns:
            raise SystemExit(f"Missing required target column: {target}")

    groups = cv_groups(df, args)
    shot_model, shot_metrics = fit_model(df, args.shot_target, groups, args, "shot")

    shot_rows = df[bool_target(df["is_shot"]) == 1].copy()
    if len(shot_rows) < args.min_shot_rows:
        raise SystemExit(f"Only {len(shot_rows)} shot rows found; need at least {args.min_shot_rows}.")
    goal_groups = cv_groups(shot_rows, args)
    goal_model, goal_metrics = fit_model(shot_rows, args.goal_target, goal_groups, args, "goal")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    shot_path = args.models_dir / f"shot_model_{args.competition}_{label}.json"
    goal_path = args.models_dir / f"xg_model_{args.competition}_{label}.json"
    meta_path = args.models_dir / f"model_metadata_{args.competition}_{label}.json"
    metrics_path = args.models_dir / f"metrics_{args.competition}_{label}.json"

    shot_model.save_model(shot_path)
    goal_model.save_model(goal_path)
    metadata = {
        "competition": args.competition,
        "model_id": label,
        "train_seasons": seasons,
        "features": FEATURES,
        "shot_target": args.shot_target,
        "goal_target": args.goal_target,
        "cv_group": args.cv_group,
        "games_per_matchday": args.games_per_matchday,
        "hyperparameter_trials": args.hyperparameter_trials,
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
