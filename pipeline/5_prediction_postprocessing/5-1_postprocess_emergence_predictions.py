"""Pipeline runner: post-process temporal emergence predictions.

Thin entry point — all logic lives in
``emergence.stages.emergence_prediction_postprocessing``.
"""

from __future__ import annotations

import logging

from emergence.stages.emergence_prediction_postprocessing import (
    load_prediction_postprocessing_paths,
    run_prediction_postprocessing,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load config and run prediction post-processing."""
    outputs = run_prediction_postprocessing(load_prediction_postprocessing_paths())
    logger.info("Post-processed predictions saved to %s", outputs.predictions_csv)
    logger.info(
        "Post-processed first-germination table saved to %s",
        outputs.first_germination_csv,
    )


if __name__ == "__main__":
    main()
