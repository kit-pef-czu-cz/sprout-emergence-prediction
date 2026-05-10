"""Pipeline runner: YOLO crop pipeline for segmented tray sequences.

Thin entry point — all logic lives in ``fenotypizace.stages.cropping``.
"""

from __future__ import annotations

import logging

from fenotypizace.stages.cropping import crop, load_crop_segments_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load config and run the cropping stage."""
    paths = load_crop_segments_paths()
    crop(paths)


if __name__ == "__main__":
    main()
