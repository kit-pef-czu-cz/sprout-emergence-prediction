"""Pipeline runner: run temporal emergence predictions from stage-3 NumPy datasets.

Thin entry point — all logic lives in ``emergence.stages.emergence_predictions``.
"""

from __future__ import annotations

import logging

from emergence.stages.emergence_predictions import (
    load_emergence_prediction_paths,
    run_emergence_predictions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load config and run the emergence-prediction stage."""
    paths = load_emergence_prediction_paths()
    dataset_spec, outputs = run_emergence_predictions(paths)
    logger.info(
        "Emergence predictions complete for image size %s, time steps %s, range %s",
        dataset_spec.image_size,
        dataset_spec.time_steps,
        dataset_spec.data_range,
    )
    logger.info("Predictions saved to %s", outputs.predictions_csv)
    logger.info("First-germination table saved to %s", outputs.first_germination_csv)


if __name__ == "__main__":
    main()
