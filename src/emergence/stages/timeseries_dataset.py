"""Build sliding-window NumPy datasets from cropped plant image sequences."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from emergence.path_config import (
    CONFIG_PATH,
    load_stage_config,
    resolve_config_path,
)
from emergence.path_config import (
    get_required_string as get_config_string,
)
from emergence.path_config import (
    resolve_project_path as resolve_shared_project_path,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SLIDE_STEP: int = 1
COLORS: int = 3
IMG_SIZE: int = 75
DEFAULT_TIME_STEPS: tuple[int, ...] = (2,)
DEFAULT_DATA_RANGE: tuple[int, ...] = (8,)
SKIP_DIRECTORIES: frozenset[str] = frozenset({"bounding_boxes", "runs"})


@dataclass(frozen=True)
class TimeseriesDatasetPaths:
    """Filesystem paths required by the time-series dataset stage."""

    input_dir: Path
    output_dir: Path


@dataclass
class WellImageSequence:
    """Cropped image sequence loaded from one tray/well directory."""

    images: list[np.ndarray]
    image_names: list[str]
    directory: Path


def get_required_string(config: dict[str, Any], key: str, config_path: Path) -> str:
    """Return a required non-empty string value from a TOML mapping.

    Args:
        config: TOML table as a plain dict.
        key: Key to look up in the table.
        config_path: Config file path, used in error messages.

    Returns:
        The non-empty string value associated with ``key``.

    Author:
        Jakub Vašák

    """
    return get_config_string(config, key, config_path)


def resolve_project_path(
    project_root: Path, relative_path: str, key: str, config_path: Path
) -> Path:
    """Resolve a relative config path against the configured project root.

    Args:
        project_root: Absolute project root directory.
        relative_path: Path string from the TOML config.
        key: Config key name, used in error messages.
        config_path: Config file path, used in error messages.

    Returns:
        Absolute path built by joining ``project_root`` and ``relative_path``.

    Author:
        Jakub Vašák

    """
    return resolve_shared_project_path(project_root, relative_path, key, config_path)


def load_timeseries_dataset_paths(
    config_path: Path = CONFIG_PATH,
) -> TimeseriesDatasetPaths:
    """Load time-series dataset paths from the shared TOML configuration.

    Args:
        config_path: Path to the TOML config file; defaults to ``CONFIG_PATH``.

    Returns:
        Populated ``TimeseriesDatasetPaths`` dataclass.

    Author:
        Jakub Vašák

    """
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
    """Return stage-3 paths and fixed preprocessing settings.

    Args:
        paths: Resolved filesystem paths for this stage.

    Returns:
        Tuple of ``(input_dir, output_dir, slide_step, colors, img_size)``.

    Author:
        Jakub Vašák

    """
    return paths.input_dir, paths.output_dir, SLIDE_STEP, COLORS, IMG_SIZE


def load_image_array(image_path: Path) -> np.ndarray:
    """Load one cropped PNG image as a NumPy array.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a NumPy array.

    Author:
        Jakub Vašák

    """
    with Image.open(image_path) as image:
        return np.array(image)


def list_box_directories(data_directory: Path) -> list[Path]:
    """Return sorted per-box directories under the cropped image root.

    Args:
        data_directory: Root cropped image directory.

    Returns:
        Sorted list of per-box directories that contain image sequences.

    Author:
        Jakub Vašák

    """
    folders = sorted(
        folder
        for folder in data_directory.iterdir()
        if folder.is_dir() and folder.name not in SKIP_DIRECTORIES
    )
    return sorted(
        entry for folder in folders for entry in folder.iterdir() if entry.is_dir()
    )


def load_from_dir(data_directory: Path, crop_size: int) -> list[WellImageSequence]:
    """Load cropped images as independent per-well sequences.

    Args:
        data_directory: Root directory containing per-box image subdirectories.
        crop_size: Minimum required pixel size; smaller images fall back to the
            previous frame.

    Returns:
        List of ``WellImageSequence`` objects, one per box directory.

    Author:
        Jakub Vašák

    """
    sequences: list[WellImageSequence] = []

    boxes = list_box_directories(data_directory)

    for box in tqdm(boxes, desc="Processing boxes", unit="box"):
        imgs_array: list[np.ndarray] = []
        image_names: list[str] = []
        previous_img: np.ndarray | None = None
        files_ranged = sorted(box.glob("*.png"))

        for filename in files_ranged:
            filename_occurrence = filename.name
            np_image = load_image_array(filename)

            if np_image.shape[0] < crop_size or np_image.shape[1] < crop_size:
                np_image = previous_img
                logger.warning(
                    "Image error, image is too small: %s", filename_occurrence
                )
            else:
                previous_img = np_image

            if np_image is not None:
                imgs_array.append(np_image)
                image_names.append(filename_occurrence)

        if imgs_array:
            sequences.append(
                WellImageSequence(
                    images=imgs_array,
                    image_names=image_names,
                    directory=box,
                )
            )

    logger.info("Loaded %s well sequence(s)", len(sequences))
    return sequences


def generate_samples(
    sequences: list[WellImageSequence], step: int, time_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sliding-window samples within each tray/well sequence.

    Args:
        sequences: List of per-well image sequences.
        step: Stride between consecutive windows.
        time_steps: Number of frames per window.

    Returns:
        Tuple of ``(samples, image_names)`` arrays with shape
        ``(n_windows, time_steps, H, W, C)`` and ``(n_windows, time_steps)``.

    Raises:
        ValueError: If ``step`` or ``time_steps`` is less than 1.

    Author:
        Jakub Vašák

    """
    if step < 1:
        raise ValueError("step must be a positive integer")
    if time_steps < 1:
        raise ValueError("time_steps must be a positive integer")

    img_trays: list[list[np.ndarray]] = []
    image_names_trays: list[list[str]] = []

    for sequence in tqdm(
        sequences,
        desc="Generating sequence samples",
        unit="sequence",
    ):
        sequence_length = len(sequence.images)
        if sequence_length < time_steps:
            logger.warning(
                "Skipping %s: %s image(s) is fewer than %s time steps.",
                sequence.directory,
                sequence_length,
                time_steps,
            )
            continue

        for index in range(0, sequence_length - time_steps + 1, step):
            img_trays.append(sequence.images[index : index + time_steps])
            image_names_trays.append(sequence.image_names[index : index + time_steps])

    return np.array(img_trays), np.array(image_names_trays)


def load_data_images(
    images_path: Path,
    save_path: Path,
    data_range: tuple[int, ...] = DEFAULT_DATA_RANGE,
    time_steps: tuple[int, ...] = DEFAULT_TIME_STEPS,
    img_crop_size: int = IMG_SIZE,
    slide_step: int = SLIDE_STEP,
) -> None:
    """Load cropped image sequences and save sliding-window prediction datasets.

    Args:
        images_path: Root directory of cropped image sequences.
        save_path: Directory in which to write the ``.npy`` dataset files.
        data_range: Sequence length(s) to process.
        time_steps: Window size(s) to generate.
        img_crop_size: Minimum image dimension required before fallback.
        slide_step: Stride between consecutive windows.

    Author:
        Jakub Vašák

    """
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

            raw_sequences = load_from_dir(images_path, img_crop_size)
            samples, image_names_trays = generate_samples(
                raw_sequences, slide_step, time_step
            )

            np.save(
                save_path / f"image_names_{img_crop_size}_{time_step}_{size}.npy",
                image_names_trays,
            )
            logger.info("Samples shape: %s", samples.shape)
            np.save(save_path / f"data_{img_crop_size}_{time_step}_{size}.npy", samples)


def _normalize_window_values(
    window_values: int | tuple[int, ...] | None, *, fallback: int
) -> tuple[int, ...]:
    """Coerce a config-driven window value into the tuple form used by helpers."""
    if window_values is None:
        return (fallback,)
    if isinstance(window_values, int):
        return (window_values,)
    return tuple(window_values)


def build_timeseries_dataset(
    paths: TimeseriesDatasetPaths | None = None,
    *,
    config_path: Path = CONFIG_PATH,
    data_range: int | tuple[int, ...] | None = None,
    time_steps: int | tuple[int, ...] | None = None,
    img_crop_size: int = IMG_SIZE,
    slide_step: int = SLIDE_STEP,
) -> TimeseriesDatasetPaths:
    """Build the configured time-series datasets from cropped image sequences.

    Args:
        paths: Resolved filesystem paths; loaded from config when ``None``.
        config_path: Path to the TOML config file; defaults to ``CONFIG_PATH``.
        data_range: Override for the sequence length(s) to process.
        time_steps: Override for the window size(s) to generate.
        img_crop_size: Minimum image dimension required before fallback.
        slide_step: Stride between consecutive windows.

    Returns:
        Resolved ``TimeseriesDatasetPaths`` used for dataset generation.

    Author:
        Jakub Vašák

    """
    context, _ = load_stage_config("timeseries_dataset", config_path)
    resolved_paths = paths or load_timeseries_dataset_paths(config_path)
    resolved_time_steps = _normalize_window_values(
        time_steps,
        fallback=context.time_steps,
    )
    resolved_data_range = _normalize_window_values(
        data_range,
        fallback=context.data_range,
    )

    logger.info(
        "Starting time-series dataset generation from %s", resolved_paths.input_dir
    )
    logger.info("Size of the cropped images: %s with colors: %s", img_crop_size, COLORS)

    load_data_images(
        images_path=resolved_paths.input_dir,
        save_path=resolved_paths.output_dir,
        data_range=resolved_data_range,
        time_steps=resolved_time_steps,
        img_crop_size=img_crop_size,
        slide_step=slide_step,
    )
    logger.info("Time-series dataset generation complete")
    return resolved_paths
