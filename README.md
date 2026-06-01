# SPROUT — AI-based Seedling emergence PRedictiOn and trait extraction Using RGB Time-series

## Overview

SPROUT is a scalable, RGB-based phenotyping pipeline for automated crop seedling emergence monitoring. It integrates spatial segmentation, temporal deep learning, and cross-experiment validation into a modular image-analysis system that works directly on RGB time-series data.

The pipeline automatically:
- Decomposes tray-level images into individual seed positions via instance segmentation (Detectron2 Mask R-CNN)
- Reduces data volume by isolating emergence regions via object detection (Ultralytics YOLO)
- Builds a NumPy time-series dataset from cropped seed images
- Predicts the precise emergence time of individual seedlings using temporal deep learning (TCN/LSTM)
- Adds prediction-only tolerance windows around the first predicted emergence event
- Reconstructs emergence curves and extracts biologically meaningful traits:
  - **Final emergence rate** — proportion of seeds that successfully emerge
  - **EC50 (time lag)** — time at which 50 % of seeds have emerged
  - **Emergence synchronicity** — variability in individual emergence times

---

## Paper

| Field | Value |
|---|---|
| **Journal** | *Computers and Electronics in Agriculture* |
| **Manuscript** | COMPAG-D-26-02593 |
| **Title** | SPROUT: AI-based Seedling emergence PRedictiOn and trait extraction Using RGB Time-series |
| **Corresponding Author** | Dr. Nuria De Diego |
| **Co-Authors** | Pavel Klimeš, PhD · Vladimir Voral · Nabila M Gomez Mansur, PhD · Jakub Vašák · Jana Kholova, PhD · Sanja Cavar Zeljkovic, PhD · Monika Rozehnalová, PhD · Lukas Spichal, PhD · Jan Masner, PhD |
| **Code Author** | Jakub Vašák |

---

## Repository Structure

```
sprout-emergence-prediction/
├── config/
│   └── paths.toml              # Central path config — edit before running
├── pipeline/                   # Standalone stage scripts (thin wrappers over src/)
│   ├── 1_segmentation/
│   │   ├── 1-0_test_detectron_model.py   # Test the Mask R-CNN model
│   │   └── 1-1_segment_boxes.py          # Stage 1 — tray instance segmentation
│   ├── 2_cropping/
│   │   └── 2-1_crop_segments.py          # Stage 2 — YOLO seed-region cropping
│   ├── 3_timeseries_dataset/
│   │   └── 3-1_timeseries_dataset.py     # Stage 3 — NumPy time-series builder
│   ├── 4_emergence_predictions/
│   │   └── 4-1_emergence_predictions.py  # Stage 4 — TCN/LSTM emergence prediction
│   └── 5_prediction_postprocessing/
│       └── 5-1_postprocess_emergence_predictions.py  # Stage 5 — prediction tolerance columns
├── src/
│   └── emergence/              # Production Python package
│       ├── path_config.py      # TOML config loader shared across stages
│       └── stages/             # One module per pipeline stage
│           ├── segmentation.py
│           ├── cropping.py
│           ├── timeseries_dataset.py
│           ├── emergence_predictions.py
│           └── emergence_prediction_postprocessing.py
├── data/
│   ├── raw/                    # Original time-series images (not in git)
│   ├── interim/                # Intermediate outputs (segmentations, crops)
│   ├── processed/              # Final NumPy datasets and prediction outputs
│   └── results/                # Derived analysis results
├── model/                      # Trained model weights — not in git (see below)
├── create_dataset.py           # Entrypoint: stages 1–3 (segmentation → dataset)
├── emergence_prediction.py     # Entrypoint: stage 4 (emergence prediction)
├── postprocess_emergence_predictions.py  # Entrypoint: stage 5 (prediction tolerance columns)
└── pyproject.toml
```

---

## About the Notebooks

1. **Reproducibility record** — a step-by-step log of how every model was trained and every experiment was run, intended to accompany the paper for peer review.
2. **Fine-tuning guide** — a hands-on starting point for anyone who wants to adapt or retrain our models on their own data (see `src_finetuning/`).

For production use, please use the scripts in `src/`.

---

## Pre-trained Models

We provide our trained model weights to the community for inference and fine-tuning:

| Model | Framework | Purpose |
|---|---|---|
| `model_final.pth` | PyTorch / Detectron2 | Mask R-CNN — tray instance segmentation |
| `germination_detector.pt` | PyTorch / Ultralytics | YOLO — seedling region detection |
| TCN / LSTM `.keras` | TensorFlow 2 / Keras | Temporal emergence prediction |

> Model weights are **not** stored in this repository. Download links will be provided upon paper acceptance.

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management. Python 3.11 is required.

> **GPU strongly recommended.** The pipeline uses CUDA for Detectron2 and YOLO inference. TensorFlow TCN/LSTM models also benefit from GPU acceleration.

---

## Configuration

All paths are centralised in `config/paths.toml`. Edit this file before running any pipeline stage:

```toml
[paths]
project_root = "/your/absolute/path/to/sprout-emergence-prediction"
project_name = "my_experiment"
```

Each stage has its own `[paths.<stage>]` section with input/output directories and model weight paths. No hardcoded paths appear in the pipeline scripts.

Prediction post-processing reuses `[paths.emergence_predictions].output_dir` and the shared `project_name`, `time_steps`, and `data_range` values. For example, if `output_dir = "data/processed/predictions"` and `project_name = "my_experiment"`, post-processed files are read from and written to `data/processed/predictions/my_experiment/`.

---

## Running the Pipeline

Always use `uv run` — never bare `python`.

### Option A — Root-level entrypoints (recommended)

```bash
# Stages 1–3: segmentation, cropping, and dataset creation
uv run create_dataset.py

# Stage 4: temporal emergence prediction
uv run emergence_prediction.py

# Stage 5: add prediction-only tolerance columns
uv run postprocess_emergence_predictions.py
```

### Option B — Individual stage scripts

```bash
# Stage 1 — Segment trays
uv run pipeline/1_segmentation/1-1_segment_boxes.py

# Stage 2 — Crop seed regions
uv run pipeline/2_cropping/2-1_crop_segments.py

# Stage 3 — Build NumPy time-series dataset
uv run pipeline/3_timeseries_dataset/3-1_timeseries_dataset.py

# Stage 4 — Run emergence predictions
uv run pipeline/4_emergence_predictions/4-1_emergence_predictions.py

# Stage 5 — Add prediction-only tolerance columns
uv run pipeline/5_prediction_postprocessing/5-1_postprocess_emergence_predictions.py
```

The public Python API for stage 1 is
`emergence.stages.segmentation.segment_boxes`. Stage 5 is available as
`emergence.stages.emergence_prediction_postprocessing.run_prediction_postprocessing`.

---

## Time-series Dataset Windows

Stage 3 builds prediction-ready NumPy arrays from cropped per-well image
sequences. Windows are generated independently within each
`Tray_<id>/<well_id>` directory, so a model input window never crosses from one
`TRAY`/`BOX_POSITION` into another.

The `slide_step` value controls the stride between consecutive window starts.
With the default `slide_step=1` and `time_steps=3`, consecutive windows overlap
by two frames. For example, one well sequence can produce `[16,18,20]`,
`[18,20,22]`, and `[20,22,00]`.

---

## Emergence Prediction Outputs

Stage 4 writes the model predictions for the configured `time_steps` and `data_range`:

| File | Description |
|---|---|
| `predictions_model_<time_steps>-<data_range>.csv/.xlsx` | Full prediction table with `PREDICTIONS`, `BOX1..BOXn`, `TRAY`, and `BOX_POSITION`. |
| `first_germination_model_<time_steps>-<data_range>.csv/.xlsx` | First positive prediction per `TRAY` and `BOX_POSITION`. |

Stage 5 keeps those files unchanged and writes enriched copies:

| File | Description |
|---|---|
| `predictions_model_<time_steps>-<data_range>_with_tolerance.csv/.xlsx` | Full prediction table plus `prediction_+1`, `prediction_-1`, `prediction_+2`, and `prediction_-2` box-name columns. |
| `first_germination_model_<time_steps>-<data_range>_with_tolerance.csv/.xlsx` | First positive prediction per `TRAY` and `BOX_POSITION`, retaining the tolerance columns. |

The directional tolerance columns contain PNG names from proxy windows derived from model predictions. For each `TRAY` and `BOX_POSITION`, the first row where `PREDICTIONS == 1` is treated as the proxy emergence target. `prediction_-1` and `prediction_-2` list unique `BOX` image names from the previous one or two rows, while `prediction_+1` and `prediction_+2` list unique `BOX` image names from the next one or two rows. These columns are not true-label accuracy metrics because they do not use manual annotations.

---

## License

To be determined upon paper acceptance.
All the scripts and models were created by Vašák, J.

---

## Citation

If you use SPROUT in your research, please cite:

> De Diego, N., Klimeš, P., Voral, V., Gomez Mansur, N. M., Vašák, J., Kholova, J., Cavar Zeljkovic, S., Rozehnalová, M., Spichal, L., & Masner, J. (*in review*). SPROUT: AI-based Seedling emergence PRedictiOn and trait extraction Using RGB Time-series. *Computers and Electronics in Agriculture*. Manuscript COMPAG-D-26-02593.
