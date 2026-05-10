"""Root entrypoint for the emergence-prediction pipeline."""

from __future__ import annotations

import logging

from fenotypizace import load_emergence_prediction_paths, run_emergence_predictions
from fenotypizace.path_config import resolve_default_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run emergence predictions with the shared config."""
    config_path = resolve_default_config_path()
    dataset_spec, outputs = run_emergence_predictions(
        load_emergence_prediction_paths(config_path=config_path)
    )
    logger.info(
        "Emergence predictions complete for image size %s, time steps %s, range %s",
        dataset_spec.image_size,
        dataset_spec.time_steps,
        dataset_spec.data_range,
    )
    logger.info("Predictions saved to %s", outputs.predictions_csv)


if __name__ == "__main__":
    main()
