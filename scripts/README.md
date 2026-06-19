# xG+ Local Pipeline

This folder contains the cleaned, numbered local pipeline. Raw PFF mirrors live in `pff-events/` and `pff-tracking/`; derived outputs are written under `data/`.

## Order

| Step | Script | Purpose |
| --- | --- | --- |
| 00 | `00_validate_environment.py` | Check Python dependencies, raw-data roots, AWS env vars, and accidental key literals. |
| 01 | `01_scrape_events.py` | Mirror event CSVs from S3 into `pff-events/{competition}/{season}`. |
| 02 | `02_scrape_tracking.py` | Mirror tracking JSONL/metadata/rosters from S3 into `pff-tracking/{competition}/{season}/{game}`. |
| 03 | `03_validate_raw_data.py` | Confirm expected game counts, event/tracking parity, required files, zero-byte files, and temp downloads. |
| 04 | `04_extract_attacks.py` | Parse tracking JSONL and build per-frame attacking-sequence features. |
| 05 | `05_merge_events_and_label_frames.py` | Join event CSV shot outcomes to tracking frames by `game_event_id` and label shot windows within the same merged attack. |
| 06 | `06_train_xgboost_models.py` | Train XGBoost shot and goal models from merged frame chunks. |
| 07 | `07_predict_xgplus.py` | Apply the trained models and write `shot_proba`, `xG`, and `goal_proba` predictions. |
| 08 | `08_evaluate_models.py` | Write frame-level and attack-level evaluation metrics. |
| 09 | `09_make_model_plots.py` | Write calibration, distribution, and feature-importance plots. |
| 10 | `10_export_attack_samples.py` | Export selected attacks to metric/tracking CSV pairs for sharing or animation. |
| 11 | `11_render_sample_attack_gif.py` | Render a GIF from exported metric/tracking CSV pairs. |
| 12 | `12_render_pff_attack_gif.py` | Render a GIF directly from PFF tracking and prediction rows. |

## Typical Run

```bash
python scripts/run_pipeline.py --competition pl --season 2024-2025 --workers 8 --skip-existing
```

Use `--run-scrapers` only when AWS credentials are configured and you want to refresh the raw mirrors. The default run assumes raw files already exist locally.

The shot model target defaults to `hasShotsIn1s`: a frame is positive when the same attacking side takes a shot within the next second of the same merged attack. Exact shot rows are also retained as `is_shot`; goal rows are retained as `is_goal`.

To repair one local tracking game without re-downloading a full season:

```bash
python scripts/02_scrape_tracking.py --competition pl --seasons 2023-2024 --games 13472 --force --workers 4 --credentials-from archived/local/sagemaker/features.py
```

Downloaded `.bz2` files are decompressed once before replacing the local copy, so a truncated download fails fast.

To run only raw validation, attack extraction, feature engineering, event labeling, and chunk export, stop before fitting probability models:

```bash
python scripts/run_pipeline.py --competition pl --seasons 2022-2023 2023-2024 2024-2025 --workers 8 --preprocess-only --skip-existing
```

`--skip-existing` reuses existing per-game attack and merged CSV files only when their sidecar metadata matches the current extraction/labeling logic. Outputs from older logic, interrupted runs, or missing sidecars are regenerated.

For a 12-core terminal run across all local PL seasons:

```bash
scripts/run_preprocessing_12core.sh
```

This enables corrupt-tracking repair by default: if one local tracking `.bz2` is truncated, the extractor re-downloads that game with `--force`, validates the replacement, retries that game once, and keeps processing. If the source download is also corrupt, the game is recorded in `data/process_games/extraction_failures_{competition}_{season}.json` and preprocessing continues. Temporary raw download files are warnings in this preprocessing mode. The script uses `archived/local/sagemaker/features.py` for AWS credentials when that ignored local file exists. Override with `TRACKING_CREDENTIALS_FROM=/path/to/file`.

To resume and opt out of already-created per-game files:

```bash
scripts/run_preprocessing_12core.sh --skip-existing
```

## Pitch Geometry

Tracking coordinates are treated as native PFF metric coordinates for each match, not rescaled to a standard field. Step 04 resolves the active `metadata.json` stadium pitch record by match date and uses that pitch length for final-third entry, distance to goal, and shot angle features. Step 05 carries `pitch_id`, `pitch_length`, and `pitch_width` into merged training rows, and prediction/export/render steps preserve those columns for diagnostics and visuals.

After changing pitch geometry logic, rerun step 04 and downstream steps without `--skip-existing` so old processed feature files are refreshed.
