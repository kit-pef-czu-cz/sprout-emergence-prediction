"""Pipeline runner: segment plant boxes from tray images.

Thin entry point — all logic lives in ``fenotypizace.stages.segmentation``.
"""

from __future__ import annotations

import logging

from fenotypizace.stages.segmentation import (
    BOX_SIZE,
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
    segment_boxes(paths=paths, box_size=BOX_SIZE, save_vis=False)
    logger.info("All done")


if __name__ == "__main__":
    main()
