"""Public package exports for the emergence library."""

from __future__ import annotations

import logging

from . import path_config
from .path_config import load_stage_config
from .stages import (
    BB_THRESHOLD,
    BOX_SIZE,
    CROP_SIZE,
    DEFAULT_DATA_RANGE,
    DEFAULT_TIME_STEPS,
    PREDICTION_THRESHOLD,
    REPORT_FILENAME,
    SCORE_THRESHOLD,
    CropReference,
    CropSegmentsPaths,
    DatasetSpec,
    DirectoryResult,
    EmergencePredictionPaths,
    PostprocessedPredictionOutputs,
    PredictionOutputs,
    PredictionPostprocessingPaths,
    SegmentBoxesPaths,
    TimeseriesDatasetPaths,
    add_proxy_tolerance_columns,
    build_first_germination_with_tolerance,
    build_timeseries_dataset,
    crop,
    load_crop_segments_paths,
    load_emergence_prediction_paths,
    load_prediction_postprocessing_paths,
    load_segment_boxes_paths,
    load_timeseries_dataset_paths,
    run_emergence_predictions,
    run_prediction_postprocessing,
    segment_boxes,
)


def configure_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logger for CLI / script usage.

    Call this once at the application entry point (e.g. main.py or a CLI
    script). Library code should never call this — only
    ``logger = logging.getLogger(__name__)`` belongs in library modules.
    """
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


__all__ = [  # noqa: RUF022
    # cropping
    "BB_THRESHOLD",
    # segmentation
    "BOX_SIZE",
    "CROP_SIZE",
    # timeseries_dataset
    "DEFAULT_DATA_RANGE",
    "DEFAULT_TIME_STEPS",
    "PREDICTION_THRESHOLD",
    "REPORT_FILENAME",
    "SCORE_THRESHOLD",
    "CropReference",
    "CropSegmentsPaths",
    # emergence_predictions
    "DatasetSpec",
    "DirectoryResult",
    "EmergencePredictionPaths",
    "PredictionOutputs",
    "SegmentBoxesPaths",
    "TimeseriesDatasetPaths",
    "build_timeseries_dataset",
    "configure_logging",
    "crop",
    "load_crop_segments_paths",
    "load_emergence_prediction_paths",
    "load_segment_boxes_paths",
    "load_stage_config",
    "load_timeseries_dataset_paths",
    "path_config",
    "run_emergence_predictions",
    "segment_boxes",
    # emergence_prediction_postprocessing
    "PostprocessedPredictionOutputs",
    "PredictionPostprocessingPaths",
    "add_proxy_tolerance_columns",
    "build_first_germination_with_tolerance",
    "load_prediction_postprocessing_paths",
    "run_prediction_postprocessing",
]
