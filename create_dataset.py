"""Root entrypoint for the dataset-creation pipeline."""

from __future__ import annotations

import logging

from fenotypizace import (
    build_timeseries_dataset,
    crop,
    load_crop_segments_paths,
    load_segment_boxes_paths,
    load_timeseries_dataset_paths,
    segment_boxes,
)
from fenotypizace.path_config import resolve_default_config_path
from fenotypizace.stages import segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run segmentation, cropping, and dataset creation with shared config."""
    config_path = resolve_default_config_path()

    segment_boxes(
        paths=load_segment_boxes_paths(config_path=config_path),
        box_size=segmentation.BOX_SIZE,
        save_vis=False,
    )
    crop(paths=load_crop_segments_paths(config_path=config_path))
    build_timeseries_dataset(
        paths=load_timeseries_dataset_paths(config_path=config_path),
        config_path=config_path,
    )
    logger.info("Dataset creation complete")


if __name__ == "__main__":
    main()
