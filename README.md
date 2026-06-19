# xG+: A Probabilistic Framework for Goalscoring in Soccer

#### Authors: Jonathan Pipping-Gamón, Tianshu Feng, and Paul Sabin

This repository contains materials for the xG+ project, including slides,
paper, raw PFF mirrors, and a local modeling pipeline.

## Directory Structure

```
.
├── asis2026/                      # ASIS 2026 presentation files
│   ├── figures/
│   └── presentation.pptx
├── pff-events/                    # Raw event CSV mirror
├── pff-tracking/                  # Raw tracking JSONL/metadata/roster mirror
├── scripts/                       # Numbered scrape -> features -> labels -> XGBoost pipeline
├── jsm2025/                       # JSM 2025 presentation files
│   ├── figures/
│   └── presentation.pptx
└── paper/                         # Full paper files
    ├── figures/
    └── paper.pdf
```

Run the local pipeline from the repository root:

```bash
python scripts/run_pipeline.py --competition pl --season 2024-2025 --workers 8 --skip-existing
```

See `scripts/README.md` for the step-by-step order and outputs.
