"""Root entrypoint for emergence prediction post-processing."""

from __future__ import annotations

import logging

from emergence import (
    load_prediction_postprocessing_paths,
    run_prediction_postprocessing,
)
from emergence.path_config import resolve_default_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Post-process emergence predictions with the shared config."""
    config_path = resolve_default_config_path()
    outputs = run_prediction_postprocessing(
        load_prediction_postprocessing_paths(config_path=config_path)
    )
    logger.info("Post-processed predictions saved to %s", outputs.predictions_csv)
    logger.info(
        "Post-processed first-germination table saved to %s",
        outputs.first_germination_csv,
    )


if __name__ == "__main__":
    main()
