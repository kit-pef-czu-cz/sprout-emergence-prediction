"""Detectron2 model testing and inference on plant segmentation images.

Basically users can test the trained model on a directory of images and
visualize the predictions with bounding boxes and masks. Thanks to that, they
can quickly check how well the model performs on new data and identify any
issues with the predictions.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from fenotypizace.path_config import (
    CONFIG_PATH,
    load_stage_config,
    resolve_config_path,
)
from fenotypizace.path_config import (
    get_required_string as get_config_string,
)
from fenotypizace.path_config import (
    resolve_project_path as resolve_shared_project_path,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

torch.backends.cudnn.benchmark = True

# Constants
TRAIN_DATASET_NAME: str = "my_trainset"
MODEL_CONFIG_PATH: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

NUM_WORKERS: int = 16
NUM_CLASSES: int = 1
SCORE_THRESHOLD: float = 0.7
VISUALIZER_SCALE: float = 0.8

SUPPORTED_IMAGE_FORMATS: set[str] = {".png", ".jpg", ".jpeg"}
PRINT_INTERVAL: int = 100


@dataclass(frozen=True)
class DetectronPaths:
    """Filesystem paths required by the Detectron2 inference script."""

    train_images: Path
    train_json: Path
    model_weights: Path
    input_dir: Path
    output_parent: Path


def configure_logging() -> None:
    """Configure logging for command-line execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


configure_logging()


def get_required_string(config: dict[str, Any], key: str, config_path: Path) -> str:
    """Return a required non-empty string value from a TOML mapping."""
    return get_config_string(config, key, config_path)


def resolve_project_path(
    project_root: Path, relative_path: str, key: str, config_path: Path
) -> Path:
    """Resolve a relative config value against the configured project root."""
    return resolve_shared_project_path(project_root, relative_path, key, config_path)


def load_detectron_paths(config_path: Path = CONFIG_PATH) -> DetectronPaths:
    """Load Detectron2 script paths from the shared TOML config."""
    context, detectron_config = load_stage_config("detectron_test", config_path)

    return DetectronPaths(
        train_images=resolve_config_path(
            context,
            get_required_string(detectron_config, "train_images", config_path),
            "paths.detectron_test.train_images",
        ),
        train_json=resolve_config_path(
            context,
            get_required_string(detectron_config, "train_json", config_path),
            "paths.detectron_test.train_json",
        ),
        model_weights=resolve_config_path(
            context,
            get_required_string(detectron_config, "model_weights", config_path),
            "paths.detectron_test.model_weights",
        ),
        input_dir=resolve_config_path(
            context,
            get_required_string(detectron_config, "input_dir", config_path),
            "paths.detectron_test.input_dir",
            include_project_name=True,
        ),
        output_parent=resolve_config_path(
            context,
            get_required_string(detectron_config, "output_parent", config_path),
            "paths.detectron_test.output_parent",
            include_project_name=True,
        ),
    )


def setup_dataset_metadata(paths: DetectronPaths) -> object:
    """Register and retrieve Detectron2 dataset metadata for visualization.

    Returns:
        MetadataCatalog: Metadata object containing dataset information.
    """
    register_coco_instances(
        TRAIN_DATASET_NAME,
        {},
        str(paths.train_json),
        str(paths.train_images),
    )
    metadata = MetadataCatalog.get(TRAIN_DATASET_NAME)
    logger.info(f"Dataset registered. Metadata: {metadata}")
    return metadata


def setup_model_configuration(paths: DetectronPaths) -> CfgNode:
    """Configure and initialize Detectron2 model settings.

    Returns:
        CfgNode: Configured Detectron2 configuration object.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_PATH))
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    cfg.MODEL.WEIGHTS = str(paths.model_weights)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    logger.info("Model configuration initialized")
    return cfg


def load_model(cfg: CfgNode) -> DefaultPredictor:
    """Load trained Detectron2 model weights and create predictor.

    Args:
        cfg: Detectron2 configuration object.

    Returns:
        DefaultPredictor: Initialized predictor for inference.
    """
    predictor = DefaultPredictor(cfg)
    logger.info("Model loaded successfully")
    return predictor


def load_image(image_path: Path) -> object:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        Image array in BGR format or None if loading fails.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Could not load image: {image_path.name}")
    return image


def visualize_predictions(image: object, outputs: dict, metadata: object) -> object:
    """Visualize model predictions on image with masks and boxes.

    Args:
        image: Input image in BGR format.
        outputs: Model predictions containing instances.
        metadata: Dataset metadata for visualization.

    Returns:
        Annotated image with predictions drawn.
    """
    visualizer = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=VISUALIZER_SCALE,
        instance_mode=ColorMode.IMAGE,
    )
    visualizer = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    return visualizer.get_image()[:, :, ::-1]


def process_images(
    input_dir: Path,
    output_dir: Path,
    predictor: DefaultPredictor,
    metadata: object,
) -> int:
    """Process all images in input directory and save predictions.

    Args:
        input_dir: Directory containing input images.
        output_dir: Directory where output images will be saved.
        predictor: Detectron2 predictor for inference.
        metadata: Dataset metadata for visualization.

    Returns:
        Number of successfully processed images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    for image_path in input_dir.iterdir():
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
            continue

        image = load_image(image_path)
        if image is None:
            logger.info(f"Can't load image {image_path}...")
            continue

        # Run inference
        outputs = predictor(image)

        # Visualize predictions
        output_image = visualize_predictions(image, outputs, metadata)

        # Save output
        output_path = output_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(output_path), output_image)

        processed_count += 1
        if processed_count % PRINT_INTERVAL == 0:
            logger.info(f"Processed {processed_count} images...")
        # if processed_count == 5:
        #     break

    return processed_count


def main() -> None:
    """Main execution function for model testing and inference."""
    paths = load_detectron_paths()

    # Setup
    metadata = setup_dataset_metadata(paths)
    cfg = setup_model_configuration(paths)
    predictor = load_model(cfg)

    # Define directories
    input_dir = paths.input_dir
    output_dir = paths.output_parent / f"detectron_{input_dir.name}"

    # Process images
    processed_count = process_images(input_dir, output_dir, predictor, metadata)

    # Report results
    logger.info(f"Completed! Processed {processed_count} images.")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
