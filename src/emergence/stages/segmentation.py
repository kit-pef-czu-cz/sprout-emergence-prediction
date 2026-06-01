"""Segment plant boxes from tray images using a trained Detectron2 model."""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from emergence.path_config import (
    CONFIG_PATH,
    get_required_string,
    load_stage_config,
    resolve_config_path,
)

warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)


from detectron2 import model_zoo  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.data.datasets import register_coco_instances  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.utils.visualizer import ColorMode, Visualizer  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


TRAIN_DATASET_NAME: str = "my_trainset"
MODEL_CONFIG_PATH: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_WORKERS: int = 16
NUM_CLASSES: int = 1
SCORE_THRESHOLD: float = 0.7
BOX_SIZE: int = 180


@dataclass(frozen=True)
class SegmentBoxesPaths:
    """Filesystem paths required by the box segmentation script."""

    train_images: Path
    train_json: Path
    model_weights: Path
    input_dir: Path
    output_dir: Path
    vis_output_dir: Path


def load_segment_boxes_paths(config_path: Path = CONFIG_PATH) -> SegmentBoxesPaths:
    """Load segmentation script paths from the shared TOML config."""
    context, segment_boxes_config = load_stage_config("segment_boxes", config_path)

    return SegmentBoxesPaths(
        train_images=resolve_config_path(
            context,
            get_required_string(segment_boxes_config, "train_images", config_path),
            "paths.segment_boxes.train_images",
        ),
        train_json=resolve_config_path(
            context,
            get_required_string(segment_boxes_config, "train_json", config_path),
            "paths.segment_boxes.train_json",
        ),
        model_weights=resolve_config_path(
            context,
            get_required_string(segment_boxes_config, "model_weights", config_path),
            "paths.segment_boxes.model_weights",
        ),
        input_dir=resolve_config_path(
            context,
            get_required_string(segment_boxes_config, "input_dir", config_path),
            "paths.segment_boxes.input_dir",
            include_project_name=True,
        ),
        output_dir=resolve_config_path(
            context,
            get_required_string(segment_boxes_config, "output_dir", config_path),
            "paths.segment_boxes.output_dir",
            include_project_name=True,
        ),
        vis_output_dir=resolve_config_path(
            context,
            get_required_string(segment_boxes_config, "vis_output_dir", config_path),
            "paths.segment_boxes.vis_output_dir",
            include_project_name=True,
        ),
    )


def load_model(paths: SegmentBoxesPaths) -> DefaultPredictor:
    """Load the Detectron2 model with the configured dataset metadata."""
    register_coco_instances(
        TRAIN_DATASET_NAME,
        {},
        str(paths.train_json),
        str(paths.train_images),
    )

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_PATH))
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    cfg.MODEL.WEIGHTS = str(paths.model_weights)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    return DefaultPredictor(cfg)


def extract_first_numbers(filename: str) -> str:
    """Extract the first one, two, or three numbers from the given filename.

    Args:
        filename: The name of the file.

    Returns:
        The extracted numbers.

    Raises:
        ValueError: If the filename does not start with a digit sequence.
    """
    match = re.match(r"(\d{1,3})", filename)
    if not match:
        raise ValueError(f"Filename {filename} has been wrongly formatted.")
    return match.group(1)


def add_rows_columns(
    box_size: int, df: pd.DataFrame, column: str = "Column_x_min"
) -> pd.DataFrame:
    """Generate bins to categorise boxes into columns/rows and label them.

    Args:
        box_size: Maximum pixel distance used to separate bin boundaries.
        df: DataFrame with box names and their coordinates.
        column: Column used for bin creation (x or y axis).

    Returns:
        DataFrame with Category_x or Category_y column added.
    """
    df = df.assign(
        Column_x_min=df.Column_x_min.astype(int),
        Column_x_max=df.Column_x_max.astype(int),
        Row_y_min=df.Row_y_min.astype(int),
        Row_y_max=df.Row_y_max.astype(int),
    )

    sorted_values = sorted(df[column])

    # Drop the outlier if the gap between the first two values exceeds 50px.
    if len(sorted_values) > 1 and sorted_values[1] - sorted_values[0] > 50:
        sorted_values = sorted_values[1:]

    bins = [sorted_values[0]]
    for value in sorted_values:
        if value - bins[-1] > box_size:
            bins.append(value)
    bins.append(sorted_values[-1] + 1)

    labels = range(1, len(bins))

    if column == "Column_x_min":
        df["Category_x"] = pd.cut(df[column], bins=bins, labels=labels, right=False)
        return df

    df["Category_y"] = pd.cut(df[column], bins=bins, labels=labels, right=False)
    return df


def build_box_output_dir(output_dir: Path, file_name: str, row_column: str) -> Path:
    """Build the output directory for one segmented box image."""
    return output_dir / f"Tray_{extract_first_numbers(file_name)}" / row_column


def save_images(df: pd.DataFrame, array: list[Image.Image], path: Path) -> None:
    """Save box images into per-location subdirectories.

    Args:
        df: DataFrame with box names and their locations.
        array: List of box images.
        path: Root path under which to create tray/location folders.
    """
    df = df.assign(
        Box_row_column=lambda x: (
            x["Category_y"].astype(str) + "-" + x["Category_x"].astype(str)
        ),
        Final_name=lambda x: (
            x["name"].str.rsplit("_", n=1).str[0] + "_" + x["Box_row_column"]
        ),
    )

    for i, img in enumerate(array):
        if pd.isna(df.loc[i, "Category_x"]) or pd.isna(df.loc[i, "Category_y"]):
            continue

        file_name = str(df.loc[i, "Final_name"])
        row_column = str(df.loc[i, "Box_row_column"])
        new_path = build_box_output_dir(path, file_name, row_column)
        new_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(new_path / f"{file_name}.png"), np.array(img))


def create_df() -> pd.DataFrame:
    """Return an empty DataFrame for accumulating box coordinate rows."""
    return pd.DataFrame(
        columns=["Column_x_min", "Column_x_max", "Row_y_min", "Row_y_max", "name"]
    )


def create_masks(predictions: dict) -> tuple[np.ndarray, int, list[Image.Image]]:
    """Extract per-instance masks from model predictions.

    Args:
        predictions: Output dict from a Detectron2 predictor.

    Returns:
        Tuple of (masks array, instance count, empty image list).
    """
    masks = np.asarray(predictions["instances"].pred_masks.to("cpu"))
    num_of_iter = len(masks)
    list_of_imgs: list[Image.Image] = []
    return (masks, num_of_iter, list_of_imgs)


def boundary_points(segmentation: tuple) -> tuple[int, int, int, int]:
    """Return the bounding box pixel coordinates for a single mask.

    Args:
        segmentation: (row_indices, col_indices) from np.where on a mask.

    Returns:
        (column_min, column_max, row_min, row_max)
    """
    column_min = int(np.min(segmentation[1]))
    column_max = int(np.max(segmentation[1]))
    row_min = int(np.min(segmentation[0]))
    row_max = int(np.max(segmentation[0]))
    return column_min, column_max, row_min, row_max


def load_image(image_path: Path) -> np.ndarray | None:
    """Load an image from disk as a NumPy array.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as an ndarray, or None on read failure.
    """
    try:
        return cv2.imread(str(image_path))
    except Exception:
        logger.exception("Error file: %s", image_path)
        return None


def append_row(
    x_min: int, x_max: int, y_min: int, y_max: int, img_name: str, i: int
) -> pd.DataFrame:
    """Build a single-row DataFrame for one detected box.

    Args:
        x_min: Min X coordinate of the box.
        x_max: Max X coordinate of the box.
        y_min: Min Y coordinate of the box.
        y_max: Max Y coordinate of the box.
        img_name: Base image name.
        i: Index of the box within the image.

    Returns:
        Single-row DataFrame.
    """
    return pd.DataFrame(
        {
            "Column_x_min": [x_min],
            "Column_x_max": [x_max],
            "Row_y_min": [y_min],
            "Row_y_max": [y_max],
            "name": [f"{img_name}_{i}"],
        }
    )


def loop_over_masks(
    box: np.ndarray,
    masks: np.ndarray,
    num_of_iter: int,
    imgs: list[Image.Image],
    img_name: str,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Image.Image]]:
    """Iterate over instance masks, crop each box, and accumulate results.

    Args:
        box: Full source image as a NumPy array.
        masks: Per-instance boolean mask array from the predictor.
        num_of_iter: Number of detected instances.
        imgs: Accumulator list for cropped images.
        img_name: Base name used to label each box.
        df: Accumulator DataFrame for box coordinates.

    Returns:
        Updated (df, imgs) tuple.
    """
    for i in range(num_of_iter):
        item_mask = masks[i]
        segmentation = np.where(item_mask)

        x_min, x_max, y_min, y_max = boundary_points(segmentation)

        cropped = Image.fromarray(box[y_min:y_max, x_min:x_max, :], mode="RGB")
        mask = Image.fromarray((item_mask * 255).astype("uint8"))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

        new_fg_image = Image.new("RGB", cropped_mask.size)
        new_fg_image.paste(cropped, (0, 0), cropped_mask)

        df = pd.concat(
            [df, append_row(x_min, x_max, y_min, y_max, img_name, i)],
            ignore_index=True,
        )
        imgs.append(new_fg_image)

    return df, imgs


def build_visualization_output_path(vis_save_path: Path, name: str) -> Path:
    """Build the output path for one visualization image."""
    return vis_save_path / f"{name}.png"


def save_vis_predictions(
    im: np.ndarray, outputs: dict, name: str, vis_save_path: Path
) -> None:
    """Save a visualisation overlay of model predictions.

    Args:
        im: Input image.
        outputs: Model prediction dict.
        name: Image stem used as the output filename.
        vis_save_path: Directory in which to write the PNG.
    """
    metadata = MetadataCatalog.get(TRAIN_DATASET_NAME)
    v = Visualizer(
        im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image = v.get_image()[:, :, ::-1]

    vis_save_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(build_visualization_output_path(vis_save_path, name)), output_image)


def segment_boxes(
    paths: SegmentBoxesPaths, box_size: int = BOX_SIZE, save_vis: bool = False
) -> None:
    """Segment growing boxes from all tray images in the input directory.

    Args:
        paths: Configured filesystem paths for this stage.
        box_size: Maximum pixel distance for grouping rows and columns.
        save_vis: When True, save prediction overlay images alongside crops.
    """
    predictor = load_model(paths)

    files = list(paths.input_dir.iterdir())
    logger.info(
        "Starting segmentation for %s images from %s", len(files), paths.input_dir
    )

    for img in tqdm(files, desc="Processing images", unit="image"):
        name = img.stem
        inp = load_image(img)
        if inp is None:
            continue

        try:
            outputs = predictor(inp)

            if save_vis:
                save_vis_predictions(inp, outputs, name, paths.vis_output_dir)

            masks, num_of_iter, imgs = create_masks(outputs)

            df_for_names, imgs = loop_over_masks(
                inp, masks, num_of_iter, imgs, name, create_df()
            )

            df_for_names = add_rows_columns(box_size, df_for_names, "Column_x_min")
            df_for_names = add_rows_columns(box_size, df_for_names, "Row_y_min")

            save_images(df_for_names, imgs, paths.output_dir)

        except Exception:
            logger.exception("Error with processing file: %s", img)
