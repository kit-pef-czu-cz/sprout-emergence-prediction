# SPROUT — AI-based Seedling emergence PRedictiOn and trait extraction Using RGB Time-series

> **⚠️ Refactoring in progress.** All models are trained, all experiments are complete, and conclusions have been drawn. This repository is currently being cleaned up and restructured for publication — converting Jupyter Notebooks into a proper Python package and polishing the pipeline scripts. The science is done; the engineering is being made publishable.

---

## Overview

SPROUT is a scalable, RGB-based phenotyping pipeline for automated crop seedling emergence monitoring. It integrates spatial segmentation, temporal deep learning, and cross-experiment validation into a modular image-analysis system that works directly on RGB time-series data.

The pipeline automatically:
- Decomposes tray-level images into individual seed positions via instance segmentation (Detectron2 Mask R-CNN)
- Reduces data volume by isolating emergence regions via object detection (Ultralytics YOLO)
- Builds a NumPy time-series dataset from cropped seed images
- Predicts the precise emergence time of individual seedlings using temporal deep learning (TCN/LSTM)
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
│   └── 4_emergence_predictions/
│       └── 4-1_emergence_predictions.py  # Stage 4 — TCN/LSTM emergence prediction
├── src/
│   └── fenotypizace/           # Production Python package
│       ├── path_config.py      # TOML config loader shared across stages
│       └── stages/             # One module per pipeline stage
│           ├── segmentation.py
│           ├── cropping.py
│           ├── timeseries_dataset.py
│           └── emergence_predictions.py
├── data/
│   ├── raw/                    # Original time-series images (not in git)
│   ├── interim/                # Intermediate outputs (segmentations, crops)
│   ├── processed/              # Final NumPy datasets ready for model training
│   └── results/                # Saved predictions
├── model/                      # Trained model weights — not in git (see below)
├── create_dataset.py           # Entrypoint: stages 1–3 (segmentation → dataset)
├── emergence_prediction.py     # Entrypoint: stage 4 (emergence prediction)
└── pyproject.toml
```

---

## About the Notebooks

The Jupyter Notebooks in `notebooks/` are **not** part of the production pipeline. They serve two purposes:

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

---

## Running the Pipeline

Always use `uv run` — never bare `python`.

### Option A — Root-level entrypoints (recommended)

```bash
# Stages 1–3: segmentation, cropping, and dataset creation
uv run create_dataset.py

# Stage 4: temporal emergence prediction
uv run emergence_prediction.py
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
```

---

## License

To be determined upon paper acceptance.
All the scripts and models were created by Vašák, J.

---

## Citation

If you use SPROUT in your research, please cite:

> De Diego, N., Klimeš, P., Voral, V., Gomez Mansur, N. M., Vašák, J., Kholova, J., Cavar Zeljkovic, S., Rozehnalová, M., Spichal, L., & Masner, J. (*in review*). SPROUT: AI-based Seedling emergence PRedictiOn and trait extraction Using RGB Time-series. *Computers and Electronics in Agriculture*. Manuscript COMPAG-D-26-02593.
