"""Pipeline runner: segment plant boxes from tray images."""

from __future__ import annotations

import logging

from emergence.stages.segmentation import (
    load_segment_boxes_paths,
    segment_boxes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load config and run the segmentation stage."""
    paths = load_segment_boxes_paths()
    segment_boxes(paths=paths, save_vis=False)
    logger.info("All done")


if __name__ == "__main__":
    main()
