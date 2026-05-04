"""Build sliding-window NumPy datasets from cropped plant image sequences."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from path_config import (  # noqa: E402
    CONFIG_PATH,
    load_stage_config,
    resolve_config_path,
)
from path_config import (  # noqa: E402
    get_required_string as get_config_string,
)
from path_config import (  # noqa: E402
    resolve_project_path as resolve_shared_project_path,
)

logger = logging.getLogger(__name__)

SLIDE_STEP: int = 1
COLORS: int = 3
IMG_SIZE: int = 75
TIME_STEPS: tuple[int, ...] = (2,)
DATA_RANGE: tuple[int, ...] = (8,)
SKIP_DIRECTORIES: frozenset[str] = frozenset({"bounding_boxes", "runs"})


@dataclass(frozen=True)
class TimeseriesDatasetPaths:
    """Filesystem paths required by the time-series dataset stage."""

    input_dir: Path
    output_dir: Path


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
    """Resolve a relative config path against the configured project root."""
    return resolve_shared_project_path(project_root, relative_path, key, config_path)


def load_timeseries_dataset_paths(
    config_path: Path = CONFIG_PATH,
) -> TimeseriesDatasetPaths:
    """Load time-series dataset paths from the shared TOML configuration."""
    context, timeseries_config = load_stage_config("timeseries_dataset", config_path)

    return TimeseriesDatasetPaths(
        input_dir=resolve_config_path(
            context,
            get_required_string(timeseries_config, "input_dir", config_path),
            "paths.timeseries_dataset.input_dir",
            include_project_name=True,
        ),
        output_dir=resolve_config_path(
            context,
            get_required_string(timeseries_config, "output_dir", config_path),
            "paths.timeseries_dataset.output_dir",
            include_project_name=True,
        ),
    )


def load_dataset(paths: TimeseriesDatasetPaths) -> tuple[Path, Path, int, int, int]:
    """Return stage-3 paths and fixed preprocessing settings."""
    return paths.input_dir, paths.output_dir, SLIDE_STEP, COLORS, IMG_SIZE


def load_image_array(image_path: Path) -> np.ndarray:
    """Load one cropped PNG image as a NumPy array."""
    with Image.open(image_path) as image:
        return np.array(image)


def list_box_directories(data_directory: Path) -> list[Path]:
    """Return sorted per-box directories under the cropped image root."""
    folders = sorted(
        folder
        for folder in data_directory.iterdir()
        if folder.is_dir() and folder.name not in SKIP_DIRECTORIES
    )
    return sorted(
        entry for folder in folders for entry in folder.iterdir() if entry.is_dir()
    )


def load_from_dir(data_directory: Path, crop_size: int) -> tuple[np.ndarray, list[str]]:
    """Load cropped images from disk and return arrays with their file names."""
    imgs_array: list[np.ndarray] = []
    previous_img: np.ndarray | None = None
    image_names: list[str] = []

    boxes = list_box_directories(data_directory)

    for box in tqdm(boxes, desc="Processing boxes", unit="box"):
        files_ranged = sorted(box.glob("*.png"))

        for filename in files_ranged:
            filename_occurrence = filename.name
            np_image = load_image_array(filename)

            if np_image.shape[0] < crop_size or np_image.shape[1] < crop_size:
                np_image = previous_img
                logger.warning("Image error, image is too small: %s", filename_occurrence)
            else:
                previous_img = np_image

            if np_image is not None:
                imgs_array.append(np_image)
                image_names.append(filename_occurrence)

    images = np.array(imgs_array)
    logger.info("Images array shape: %s", images.shape)
    return images, image_names


def generate_samples(
    images: np.ndarray, step: int, time_steps: int, image_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sliding-window samples and aligned image-name windows."""
    img_trays: list[np.ndarray] = [images[0:time_steps]]
    image_names_trays: list[list[str]] = [image_names[0:time_steps]]
    total_iters = (images.shape[0] - time_steps) // step

    for index in tqdm(
        range(1, images.shape[0] - time_steps, step),
        desc="Generating sequence samples",
        total=total_iters,
        unit="sequence",
    ):
        img_trays.append(images[index : index + time_steps])
        image_names_trays.append(image_names[index : index + time_steps])

    return np.array(img_trays), np.array(image_names_trays)


def load_data_images(
    images_path: Path,
    save_path: Path,
    data_range: tuple[int, ...] = DATA_RANGE,
    time_steps: tuple[int, ...] = TIME_STEPS,
    img_crop_size: int = IMG_SIZE,
    slide_step: int = SLIDE_STEP,
) -> None:
    """Load cropped image sequences and save NumPy datasets for prediction."""
    save_path.mkdir(parents=True, exist_ok=True)

    for time_step in time_steps:
        logger.info("Step size: %s", time_step)

        for size in data_range:
            if size < time_step:
                logger.warning(
                    "Sequence length (%s) is shorter than required time step (%s)",
                    size,
                    time_step,
                )
                continue

            logger.info("Range size: %s", size)

            raw_images, image_names = load_from_dir(images_path, img_crop_size)
            samples, image_names_trays = generate_samples(
                raw_images, slide_step, time_step, image_names
            )

            np.save(
                save_path / f"image_names_{img_crop_size}_{time_step}_{size}.npy",
                image_names_trays,
            )
            logger.info("Samples shape: %s", samples.shape)
            np.save(save_path / f"data_{img_crop_size}_{time_step}_{size}.npy", samples)


def main() -> None:
    """Run the time-series dataset generation stage."""
    paths = load_timeseries_dataset_paths()
    data_images_path, save_path, slide_step, colors, img_size = load_dataset(paths)

    logger.info("Starting time-series dataset generation from %s", data_images_path)
    logger.info("Size of the cropped images: %s with colors: %s", img_size, colors)

    load_data_images(
        images_path=data_images_path,
        data_range=DATA_RANGE,
        time_steps=TIME_STEPS,
        img_crop_size=img_size,
        slide_step=slide_step,
        save_path=save_path,
    )
    logger.info("Time-series dataset generation complete")


if __name__ == "__main__":
    main()
