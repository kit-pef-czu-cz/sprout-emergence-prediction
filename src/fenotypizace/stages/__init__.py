"""Public stage exports for the fenotypizace package."""

from __future__ import annotations

from fenotypizace.stages.cropping import (
    BB_THRESHOLD,
    CROP_SIZE,
    REPORT_FILENAME,
    CropReference,
    CropSegmentsPaths,
    DirectoryResult,
    crop,
    load_crop_segments_paths,
)
from fenotypizace.stages.emergence_predictions import (
    PREDICTION_THRESHOLD,
    DatasetSpec,
    EmergencePredictionPaths,
    PredictionOutputs,
    load_emergence_prediction_paths,
    run_emergence_predictions,
)
from fenotypizace.stages.segmentation import (
    BOX_SIZE,
    SCORE_THRESHOLD,
    SegmentBoxesPaths,
    fenotypizace,
    load_segment_boxes_paths,
    segment_boxes,
)
from fenotypizace.stages.timeseries_dataset import (
    DEFAULT_DATA_RANGE,
    DEFAULT_TIME_STEPS,
    TimeseriesDatasetPaths,
    build_timeseries_dataset,
    load_timeseries_dataset_paths,
)

__all__ = [
    # segmentation
    "BOX_SIZE",
    "SCORE_THRESHOLD",
    "SegmentBoxesPaths",
    "fenotypizace",
    "load_segment_boxes_paths",
    "segment_boxes",
    # cropping
    "BB_THRESHOLD",
    "CROP_SIZE",
    "CropReference",
    "CropSegmentsPaths",
    "DirectoryResult",
    "REPORT_FILENAME",
    "crop",
    "load_crop_segments_paths",
    # timeseries_dataset
    "DEFAULT_DATA_RANGE",
    "DEFAULT_TIME_STEPS",
    "TimeseriesDatasetPaths",
    "build_timeseries_dataset",
    "load_timeseries_dataset_paths",
    # emergence_predictions
    "DatasetSpec",
    "EmergencePredictionPaths",
    "PREDICTION_THRESHOLD",
    "PredictionOutputs",
    "load_emergence_prediction_paths",
    "run_emergence_predictions",
]
