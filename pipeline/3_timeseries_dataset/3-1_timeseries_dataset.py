"""Pipeline runner: build time-series NumPy datasets from cropped image sequences.

Thin entry point — all logic lives in ``fenotypizace.stages.timeseries_dataset``.
"""

from __future__ import annotations

import logging

from fenotypizace.stages.timeseries_dataset import (
    build_timeseries_dataset,
    load_timeseries_dataset_paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load config and run the time-series dataset stage."""
    paths = load_timeseries_dataset_paths()
    build_timeseries_dataset(paths=paths)
    logger.info("All done")


if __name__ == "__main__":
    main()
