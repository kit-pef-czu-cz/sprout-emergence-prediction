"""Segment plant boxes from tray images using a trained Detectron2 model."""

from __future__ import annotations

import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from path_config import (  # noqa: E402, I001
    CONFIG_PATH,
    get_required_string as get_config_string,
    load_stage_config,
    resolve_config_path,
    resolve_project_path as resolve_shared_project_path,
)

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
    """
    Extract the first one, two, or three numbers from the given filename.

    Args:
        filename (str): The name of the file.

    Returns:
        str: The extracted numbers.
    """
    # Regex to find the first sequence of one to three digits in the filename
    match = re.match(r"(\d{1,3})", filename)
    if not match:
        raise ValueError(f"Filename {filename} has been wrongly formatted.")
    return match.group(1)


def add_rows_columns(
    box_size: int, df: pd.DataFrame, column: str = "Column_x_min"
) -> pd.DataFrame:
    """Function that automatically generates bins for categorizing data into columns and rows.
    It is given that one box has a maximum of 180.
    The starting point is chosen as the X/Y pixel value of the first box.

    Args:
        df (pd.DataFrame): DF with names of boxes and their coordinates.
        column (str, optional): Creation of bins for columns or rows. Defaults to "Column_x_min".

    Returns:
        pd.DataFrame: DF with correct names of individual boxes.
    """
    df = df.assign(
        Column_x_min=df.Column_x_min.astype(int),
        Column_x_max=df.Column_x_max.astype(int),
        Row_y_min=df.Row_y_min.astype(int),
        Row_y_max=df.Row_y_max.astype(int),
    )

    # Sort the values in the column. It needs to be converted to a list for bin creation purposes. We also do not want to change the original DF, which contains indices.
    sorted_values = sorted(df[column])

    # Check if the gap between the first and second values is greater than 50
    if len(sorted_values) > 1 and sorted_values[1] - sorted_values[0] > 50:
        sorted_values = sorted_values[1:]

    # Dynamically create bins + Starting point of the first category = smallest value in the selected column
    bins = [sorted_values[0]]

    # Add values to bins if the next value is more than 180 pixels away from the previous point in the list
    for value in sorted_values:
        if value - bins[-1] > box_size:
            bins.append(value)

    # Add the last value to bins to include the last group. A value of 1 greater than the largest value in the column is added due to the definition of bins.
    bins.append(sorted_values[-1] + 1)

    # Create labels for categories -> 1 - 8 for rows, 1 - 9 for columns
    labels = range(1, len(bins))

    # Use pd.cut to assign values to categories based on bins
    if column == "Column_x_min":
        df["Category_x"] = pd.cut(df[column], bins=bins, labels=labels, right=False)
        return df

    df["Category_y"] = pd.cut(df[column], bins=bins, labels=labels, right=False)

    return df


def build_box_output_dir(output_dir: Path, file_name: str, row_column: str) -> Path:
    """Build the output directory for one segmented box image."""
    return output_dir / f"Tray_{extract_first_numbers(file_name)}" / row_column


def save_images(df: pd.DataFrame, array: list[Image.Image], path: Path) -> None:
    """Function to save images into folders based on the location of boxes.

    Args:
        df (pd.DataFrame): DF with box names and their locations.
        array (list): List of box images.
        path (Path): Path to save images.
    """
    # # Create a new column with the Box_row_column location
    # # Then create a new column Final_name with the name of the box based on its location
    df = df.assign(
        Box_row_column=lambda x: (
            x["Category_y"].astype(str) + "-" + x["Category_x"].astype(str)
        ),
        Final_name=lambda x: (
            x["name"].str.rsplit("_", n=1).str[0] + "_" + x["Box_row_column"]
        ),
    )

    # Create folders to save images and save boxes into folders
    for i, img in enumerate(array):
        # If the box is not categorized, skip it
        if pd.isna(df.loc[i, "Category_x"]) or pd.isna(df.loc[i, "Category_y"]):
            continue

        # Save the file name, box location, and create the path to save
        file_name = str(df.loc[i, "Final_name"])
        row_column = str(df.loc[i, "Box_row_column"])
        new_path = build_box_output_dir(path, file_name, row_column)

        # Create the folder if it doesn't exist
        new_path.mkdir(parents=True, exist_ok=True)

        # Save the image into the folder
        cv2.imwrite(str(new_path / f"{file_name}.png"), np.array(img))


def create_df() -> pd.DataFrame:
    """Function to create an empty DataFrame for storing boxes.

    Returns:
        pd.DataFrame: DataFrame with empty columns.
    """
    return pd.DataFrame(
        columns=["Column_x_min", "Column_x_max", "Row_y_min", "Row_y_max", "name"]
    )


def create_masks(predictions: dict) -> tuple[np.ndarray, int, list[Image.Image]]:
    """Function to create a mask from model predictions.

    Args:
        predictions (dict): Predictions from the model.

    Returns:
        tuple[np.ndarray, int, list[Image.Image]]: Created mask, number of objects in the image, list for storing images.
    """
    # Taking output/img from prediction of our modern and creates a mask from it, which is processed by CPU
    masks = np.asarray(predictions["instances"].pred_masks.to("cpu"))

    # Number of objects in the image
    num_of_iter = len(masks)

    # List for storing image vectors
    list_of_imgs: list[Image.Image] = []

    return (masks, num_of_iter, list_of_imgs)


def boundary_points(segmentation: tuple) -> tuple[int, int, int, int]:
    """Function to find the boundaries of a box.

    Args:
        segmentation (tuple): Coordinates of the box.

    Returns:
        tuple[int, int, int, int]: Minimum and maximum values for columns and rows.
    """
    # Find the boundary points of each box
    column_min = int(np.min(segmentation[1]))
    column_max = int(np.max(segmentation[1]))
    row_min = int(np.min(segmentation[0]))
    row_max = int(np.max(segmentation[0]))

    return column_min, column_max, row_min, row_max


def load_image(image_path: Path) -> np.ndarray | None:
    """Function to load an image from a path.

    Args:
        image_path (Path): Path to the image.

    Returns:
        np.ndarray | None: Image as np.array or None if the image cannot be loaded.
    """
    # Load the image and predict the model
    try:
        return cv2.imread(str(image_path))

    # If the image cannot be loaded, an error is displayed and the next image is processed
    except Exception:
        logger.exception("Error file: %s", image_path)
        return None


def append_row(
    x_min: int, x_max: int, y_min: int, y_max: int, img_name: str, i: int
) -> pd.DataFrame:
    """Function to add a row to a DataFrame.

    Args:
        x_min (int): Min X coordinate of the box.
        x_max (int): Max X coordinate of the box.
        y_min (int): Min Y coordinate of the box.
        y_max (int): Max Y coordinate of the box.
        img_name (str): Image name.
        i (int): Order of the box in the mask.

    Returns:
        pd.DataFrame: DataFrame with the new row.
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
    """Function to iterate over masks and create box segmentation.

    Args:
        box (np.ndarray): Object in the shape mask.
        masks (np.ndarray): Mask from the model prediction.
        num_of_iter (int): Number of objects in the image.
        imgs (list): List for storing images.
        img_name (str): Image name.
        df (pd.DataFrame): DF for storing boxes.

    Returns:
        tuple[pd.DataFrame, list[Image.Image]]: DF with stored boxes and list with images.
    """

    for i in range(num_of_iter):
        # Take only one object mask from the image
        item_mask = masks[i]
        segmentation = np.where(item_mask)

        # Find the boundary points of each box
        x_min, x_max, y_min, y_max = boundary_points(segmentation)

        # Cropping the box from the image, creating a mask, and then cropping the mask of the box from the image
        cropped = Image.fromarray(box[y_min:y_max, x_min:x_max, :], mode="RGB")
        mask = Image.fromarray((item_mask * 255).astype("uint8"))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

        # Combine the box and the mask into one image, with the box in the foreground bounded by the mask = black
        new_fg_image = Image.new("RGB", cropped_mask.size)
        new_fg_image.paste(cropped, (0, 0), cropped_mask)

        # Save the cropped box
        # If no column or row is recognized, the box is saved to the Unknown folder and a new row is added to the dataframe
        df = pd.concat(
            [df, append_row(x_min, x_max, y_min, y_max, img_name, i)], ignore_index=True
        )
        imgs.append(new_fg_image)

    return df, imgs


def build_visualization_output_path(vis_save_path: Path, name: str) -> Path:
    """Build the output path for one visualization image."""
    return vis_save_path / f"{name}.png"


def save_vis_predictions(
    im: np.ndarray, outputs: dict, name: str, vis_save_path: Path
) -> None:
    """Save visualization of model predictions.

    Args:
        im: Input image
        outputs: Model predictions
        name: Image name
        vis_save_path: Path to save the visualizations
    """
    metadata = MetadataCatalog.get(TRAIN_DATASET_NAME)
    v = Visualizer(
        im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image = v.get_image()[:, :, ::-1]

    # Create the folder if it doesn't exist
    vis_save_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(build_visualization_output_path(vis_save_path, name)), output_image)


def fenotypizace(
    paths: SegmentBoxesPaths, box_size: int, save_vis: bool = False
) -> None:
    """Function to create segmented growing boxes from a seeder.

    Args:
        paths (SegmentBoxesPaths): Configured filesystem paths for the script.
        box_size (int): Maximum pixel distance for grouping rows and columns.
    """
    # Load the model
    predictor = load_model(paths)

    # Get total number of files for progress bar
    files = list(paths.input_dir.iterdir())
    logger.info(
        "Starting segmentation for %s images from %s", len(files), paths.input_dir
    )

    for img in tqdm(files, desc="Processing images", unit="image"):
        # Name of the image without the .png extension
        name = img.stem

        # Load the image and predict the model, if the image cannot be loaded, the next image is processed
        inp = load_image(img)
        if inp is None:
            continue

        try:
            # Prediction of the model
            outputs = predictor(inp)

            # Save predictions if users true
            if save_vis:
                save_vis_predictions(inp, outputs, name, paths.vis_output_dir)

            # Create a mask and get the number of objects in the image and create a list of images
            masks, num_of_iter, imgs = create_masks(outputs)

            # Loop for processing individual objects in the image
            df_for_names, imgs = loop_over_masks(
                inp, masks, num_of_iter, imgs, name, create_df()
            )

            # Add columns for rows and columns
            df_for_names = add_rows_columns(box_size, df_for_names, "Column_x_min")
            df_for_names = add_rows_columns(box_size, df_for_names, "Row_y_min")

            # Save the images into folders
            save_images(df_for_names, imgs, paths.output_dir)

        except Exception:
            logger.exception("Error with processing file: %s", img)


def main() -> None:
    """Main function to run the segmentation process."""
    paths = load_segment_boxes_paths()

    # Run the segmentation process
    fenotypizace(paths=paths, box_size=BOX_SIZE, save_vis=False)
    logger.info("All done")


if __name__ == "__main__":
    main()
