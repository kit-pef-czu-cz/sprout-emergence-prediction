"""Segment plant boxes from tray images using a trained Detectron2 model."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
TRAIN_DATASET_NAME = "my_trainset"
MODEL_CONFIG_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_WORKERS = 8
NUM_CLASSES = 1
SCORE_THRESHOLD = 0.7
DEFAULT_SEGMENTATION_BOX_SIZE = 180


@dataclass(frozen=True)
class SegmentationPaths:
    """Filesystem paths required by the finetuning segmentation script."""

    train_images: Path
    train_json: Path
    model_weights: Path
    input_dir: Path
    output_dir: Path
    vis_output_dir: Path
    box_size: int = DEFAULT_SEGMENTATION_BOX_SIZE


# Advanced users: edit these paths directly in this script when needed.
ROOT_PATH = Path("/home/vasakjakub/fenotypizace")
SEGMENTATION_MODEL_DIR = ROOT_PATH / "models" / "segmentation_model"
DEFAULT_SEGMENTATION_PATHS = SegmentationPaths(
    train_images=SEGMENTATION_MODEL_DIR / "annotations" / "train" / "images",
    train_json=SEGMENTATION_MODEL_DIR / "annotations" / "train" / "result.json",
    model_weights=SEGMENTATION_MODEL_DIR / "model_final.pth",
    input_dir=ROOT_PATH / "data" / "test_files",
    output_dir=ROOT_PATH / "data" / "segmentations",
    vis_output_dir=ROOT_PATH / "data" / "seg_vis_results",
)

setup_logger()


def register_training_dataset(paths: SegmentationPaths) -> None:
    """Register the training dataset metadata once per interpreter session."""
    if TRAIN_DATASET_NAME in DatasetCatalog.list():
        return
    register_coco_instances(
        TRAIN_DATASET_NAME,
        {},
        str(paths.train_json),
        str(paths.train_images),
    )


def load_model(paths: SegmentationPaths) -> DefaultPredictor:
    """Load the Detectron2 predictor for the finetuning segmentation stage."""
    register_training_dataset(paths)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_PATH))
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    cfg.MODEL.WEIGHTS = str(paths.model_weights)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    return DefaultPredictor(cfg)


def extract_first_numbers(filename: str) -> str:
    """Extract the first one-, two-, or three-digit prefix from a filename."""
    match = re.match(r"(\d{1,3})", filename)
    if not match:
        raise ValueError(f"Filename {filename} has been wrongly formatted.")
    return match.group(1)


def add_rows_columns(
    box_size: int,
    df: pd.DataFrame,
    column: str = "Column_x_min",
) -> pd.DataFrame:
    """Generate row/column bins used to label the segmented boxes."""
    df = df.assign(
        Column_x_min=df.Column_x_min.astype(int),
        Column_x_max=df.Column_x_max.astype(int),
        Row_y_min=df.Row_y_min.astype(int),
        Row_y_max=df.Row_y_max.astype(int),
    )

    sorted_values = sorted(df[column])
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


def build_output_dir(root: Path, file_name: str, row_column: str) -> Path:
    """Build the output directory for one segmented box image."""
    return root / f"Tray_{extract_first_numbers(file_name)}" / row_column


def save_images(df: pd.DataFrame, images: list[Image.Image], path: Path) -> None:
    """Save segmented plant boxes into tray/location directories."""
    df = df.assign(
        Box_row_column=lambda frame: (
            frame["Category_y"].astype(str) + "-" + frame["Category_x"].astype(str)
        ),
        Final_name=lambda frame: (
            frame["name"].str.rsplit("_", n=1).str[0] + "_" + frame["Box_row_column"]
        ),
    )

    for index, image in enumerate(images):
        if pd.isna(df.loc[index, "Category_x"]) or pd.isna(df.loc[index, "Category_y"]):
            continue

        file_name = str(df.loc[index, "Final_name"])
        row_column = str(df.loc[index, "Box_row_column"])
        output_dir = build_output_dir(path, file_name, row_column)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"{file_name}.png"), np.array(image))


def create_df() -> pd.DataFrame:
    """Create an empty DataFrame for storing segmented box coordinates."""
    return pd.DataFrame(
        columns=["Column_x_min", "Column_x_max", "Row_y_min", "Row_y_max", "name"]
    )


def create_masks(predictions: dict) -> tuple[np.ndarray, int, list[Image.Image]]:
    """Extract instance masks and prepare the image accumulator list."""
    masks = np.asarray(predictions["instances"].pred_masks.to("cpu"))
    return masks, len(masks), []


def boundary_points(
    segmentation: tuple[np.ndarray, np.ndarray],
) -> tuple[int, int, int, int]:
    """Return the bounding-box coordinates of one segmentation mask."""
    column_min = int(np.min(segmentation[1]))
    column_max = int(np.max(segmentation[1]))
    row_min = int(np.min(segmentation[0]))
    row_max = int(np.max(segmentation[0]))
    return column_min, column_max, row_min, row_max


def load_image(image_path: Path) -> np.ndarray | None:
    """Load an image from disk or return ``None`` when the read fails."""
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.warning("Could not load image: %s", image_path)
    return image


def append_row(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    img_name: str,
    index: int,
) -> pd.DataFrame:
    """Create a one-row DataFrame for one segmented box."""
    return pd.DataFrame(
        {
            "Column_x_min": [x_min],
            "Column_x_max": [x_max],
            "Row_y_min": [y_min],
            "Row_y_max": [y_max],
            "name": [f"{img_name}_{index}"],
        }
    )


def loop_over_masks(
    box: np.ndarray,
    masks: np.ndarray,
    num_of_iter: int,
    images: list[Image.Image],
    img_name: str,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Image.Image]]:
    """Convert predicted masks into cropped foreground box images."""
    for index in range(num_of_iter):
        item_mask = masks[index]
        segmentation = np.where(item_mask)
        x_min, x_max, y_min, y_max = boundary_points(segmentation)

        cropped = Image.fromarray(box[y_min:y_max, x_min:x_max, :], mode="RGB")
        mask = Image.fromarray((item_mask * 255).astype("uint8"))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

        new_fg_image = Image.new("RGB", cropped_mask.size)
        new_fg_image.paste(cropped, (0, 0), cropped_mask)

        df = pd.concat(
            [df, append_row(x_min, x_max, y_min, y_max, img_name, index)],
            ignore_index=True,
        )
        images.append(new_fg_image)

    return df, images


def save_vis_predictions(
    image: np.ndarray,
    outputs: dict,
    name: str,
    vis_save_path: Path,
) -> None:
    """Save a visualization overlay of Detectron2 predictions."""
    metadata = MetadataCatalog.get(TRAIN_DATASET_NAME)
    visualizer = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=0.8,
        instance_mode=ColorMode.IMAGE,
    )
    visualizer = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image = visualizer.get_image()[:, :, ::-1]
    vis_save_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_save_path / f"{name}.png"), output_image)


def segment_boxes(paths: SegmentationPaths, box_size: int | None = None) -> None:
    """Create segmented box crops from tray images."""
    predictor = load_model(paths)
    resolved_box_size = box_size or paths.box_size
    files = sorted(path for path in paths.input_dir.iterdir() if path.is_file())

    for image_path in tqdm(files, desc="Processing images", unit="image"):
        name = image_path.stem
        image = load_image(image_path)
        if image is None:
            continue

        try:
            outputs = predictor(image)
            save_vis_predictions(image, outputs, name, paths.vis_output_dir)
            masks, num_of_iter, cropped_images = create_masks(outputs)
            df_for_names, cropped_images = loop_over_masks(
                image,
                masks,
                num_of_iter,
                cropped_images,
                name,
                create_df(),
            )
            df_for_names = add_rows_columns(
                resolved_box_size, df_for_names, "Column_x_min"
            )
            df_for_names = add_rows_columns(
                resolved_box_size, df_for_names, "Row_y_min"
            )
            save_images(df_for_names, cropped_images, paths.output_dir)
        except Exception:
            LOGGER.exception("Error with processing file: %s", image_path)


def fenotypizace(paths: SegmentationPaths, box_size: int | None = None) -> None:
    """Backward-compatible alias for the legacy segmentation entrypoint."""
    segment_boxes(paths, box_size)


def main() -> None:
    """Run the segmentation process using the local path definitions above."""
    segment_boxes(DEFAULT_SEGMENTATION_PATHS)
    print("All done")


if __name__ == "__main__":
    main()
