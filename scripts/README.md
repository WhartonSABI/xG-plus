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
