#!/usr/bin/env python3
"""Run local xG+ extraction, labeling, training, prediction, and evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local xG+ pipeline from raw mirrors through diagnostics.")
    parser.add_argument("--tracking-root", type=Path, default=Path("pff-tracking"))
    parser.add_argument("--event-root", type=Path, default=Path("pff-events"))
    parser.add_argument("--competition", default="pl")
    parser.add_argument("--season", default="2024-2025", help="Single season to process; ignored when --seasons is set.")
    parser.add_argument("--seasons", nargs="+", default=None, help="Process multiple seasons, e.g. 2022-2023 2023-2024.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--repair-corrupt-tracking",
        action="store_true",
        help="During extraction, re-download a corrupt local tracking game once and retry it.",
    )
    parser.add_argument(
        "--continue-on-unrepairable-corrupt-tracking",
        action="store_true",
        help="If a corrupt tracking game cannot be repaired, record it and continue preprocessing.",
    )
    parser.add_argument(
        "--tracking-credentials-from",
        type=Path,
        default=None,
        help="Optional archived Python file containing AWS os.environ assignments for tracking repair/downloads.",
    )
    parser.add_argument("--repair-workers", type=int, default=4)
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run through extraction and event labeling, then stop before model fitting.",
    )
    parser.add_argument("--allow-missing-events", action="store_true")
    parser.add_argument("--run-scrapers", action="store_true", help="Mirror raw event/tracking data before validating.")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--skip-raw-validation", action="store_true")
    parser.add_argument("--allow-temp-files", action="store_true", help="Warn on raw mirror *.tmp files instead of failing validation.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--shot-target", default="hasShotsIn1s")
    parser.add_argument("--min-shot-rows", type=int, default=100)
    parser.add_argument("--process-dir", type=Path, default=Path("data/process_games"))
    parser.add_argument("--merged-games-dir", type=Path, default=Path("data/merged_games"))
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged_data"))
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--shots-path", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--plots-dir", type=Path, default=Path("data/plots"))
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    scripts_dir = Path(__file__).resolve().parent
    seasons = args.seasons or [args.season]

    if args.preprocess_only:
        args.skip_train = True
        args.skip_predict = True
        args.skip_evaluate = True
        args.skip_plots = True

    if len(seasons) > 1 and not args.preprocess_only and (args.predictions_path is not None or args.shots_path is not None):
        raise SystemExit("--predictions-path and --shots-path are only supported with one season unless --preprocess-only is used.")

    if not args.skip_env_check:
        env_cmd = [
            sys.executable,
            str(scripts_dir / "00_validate_environment.py"),
            "--tracking-root",
            str(args.tracking_root),
            "--event-root",
            str(args.event_root),
        ]
        if not args.skip_train or not args.skip_predict:
            env_cmd.append("--require-xgboost")
        run(env_cmd)

    if args.run_scrapers:
        run(
            [
                sys.executable,
                str(scripts_dir / "01_scrape_events.py"),
                "--output-root",
                str(args.event_root),
                "--competition",
                args.competition,
                "--seasons",
                *seasons,
                "--workers",
                str(args.workers),
            ]
        )
        run(
            [
                sys.executable,
                str(scripts_dir / "02_scrape_tracking.py"),
                "--output-root",
                str(args.tracking_root),
                "--competition",
                args.competition,
                "--seasons",
                *seasons,
                "--workers",
                str(args.workers),
            ]
        )

    if not args.skip_raw_validation:
        run(
            [
                sys.executable,
                str(scripts_dir / "03_validate_raw_data.py"),
                "--tracking-root",
                str(args.tracking_root),
                "--event-root",
                str(args.event_root),
                "--competition",
                args.competition,
                "--seasons",
                *seasons,
            ]
            + (["--allow-temp-files"] if args.allow_temp_files else [])
        )

    for season in seasons:
        predictions_path = args.predictions_path or Path("data/predictions") / f"predictions_{args.competition}_{season}.csv"
        shots_path = args.shots_path or Path("data/predictions") / f"shots_{args.competition}_{season}.csv"

        extract_cmd = [
            sys.executable,
            str(scripts_dir / "04_extract_attacks.py"),
            "--tracking-root",
            str(args.tracking_root),
            "--competition",
            args.competition,
            "--season",
            season,
            "--workers",
            str(args.workers),
            "--output-dir",
            str(args.process_dir),
        ]
        if args.limit_games > 0:
            extract_cmd += ["--limit-games", str(args.limit_games)]
        if args.skip_existing:
            extract_cmd.append("--skip-existing")
        if args.repair_corrupt_tracking:
            extract_cmd.append("--repair-corrupt-tracking")
        if args.continue_on_unrepairable_corrupt_tracking:
            extract_cmd.append("--continue-on-unrepairable-corrupt-tracking")
        if args.tracking_credentials_from is not None:
            extract_cmd += ["--tracking-credentials-from", str(args.tracking_credentials_from)]
        extract_cmd += ["--repair-workers", str(args.repair_workers)]

        label_cmd = [
            sys.executable,
            str(scripts_dir / "05_merge_events_and_label_frames.py"),
            "--process-dir",
            str(args.process_dir),
            "--tracking-root",
            str(args.tracking_root),
            "--event-root",
            str(args.event_root),
            "--output-dir",
            str(args.merged_games_dir),
            "--chunk-dir",
            str(args.merged_dir),
            "--competition",
            args.competition,
            "--season",
            season,
            "--chunk-size",
            str(args.chunk_size),
            "--workers",
            str(args.workers),
        ]
        if args.allow_missing_events:
            label_cmd.append("--allow-missing-events")
        if args.skip_existing:
            label_cmd.append("--skip-existing")

        train_cmd = [
            sys.executable,
            str(scripts_dir / "06_train_xgboost_models.py"),
            "--merged-dir",
            str(args.merged_dir),
            "--models-dir",
            str(args.models_dir),
            "--competition",
            args.competition,
            "--season",
            season,
            "--shot-target",
            args.shot_target,
            "--min-shot-rows",
            str(args.min_shot_rows),
        ]

        predict_cmd = [
            sys.executable,
            str(scripts_dir / "07_predict_xgplus.py"),
            "--merged-dir",
            str(args.merged_dir),
            "--models-dir",
            str(args.models_dir),
            "--predictions-path",
            str(predictions_path),
            "--shots-path",
            str(shots_path),
            "--competition",
            args.competition,
            "--season",
            season,
        ]

        evaluate_cmd = [
            sys.executable,
            str(scripts_dir / "08_evaluate_models.py"),
            "--predictions",
            str(predictions_path),
            "--models-dir",
            str(args.models_dir),
            "--output-dir",
            str(args.evaluation_dir),
            "--competition",
            args.competition,
            "--season",
            season,
            "--shot-target",
            args.shot_target,
        ]

        plots_cmd = [
            sys.executable,
            str(scripts_dir / "09_make_model_plots.py"),
            "--predictions",
            str(predictions_path),
            "--models-dir",
            str(args.models_dir),
            "--output-dir",
            str(args.plots_dir),
            "--competition",
            args.competition,
            "--season",
            season,
            "--shot-target",
            args.shot_target,
        ]

        run(extract_cmd)
        run(label_cmd)
        if not args.skip_train:
            run(train_cmd)
        if not args.skip_predict:
            run(predict_cmd)
        if not args.skip_evaluate:
            run(evaluate_cmd)
        if not args.skip_plots:
            run(plots_cmd)

    print("xG+ preprocessing complete." if args.preprocess_only else "xG+ pipeline complete.")


if __name__ == "__main__":
    main()
