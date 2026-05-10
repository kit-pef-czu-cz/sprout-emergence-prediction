"""Crop plant image sequences using YOLO detections as the reference box."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)
YOLO_PROJECT_NAME = "detect"
YOLO_CONFIDENCE = 0.6
YOLO_MAX_DET = 5
CROP_SIZE = (75, 75)
SMALL_BOX_LIMIT = 55
YOLO_COLUMNS = ["name", "center_x", "center_y", "width", "height", "conf"]


@dataclass(frozen=True)
class CroppingPaths:
    """Filesystem paths required by the finetuning cropping script."""

    model_weights: Path
    input_dir: Path
    output_dir: Path


# Advanced users: edit these paths directly in this script when needed.
ROOT_PATH = Path("/home/vasakjakub/fenotypizace")
DEFAULT_CROPPING_PATHS = CroppingPaths(
    model_weights=(
        ROOT_PATH
        / "models"
        / "object_detection_model"
        / "fine_tuning"
        / "germination_detector.pt"
    ),
    input_dir=ROOT_PATH / "data" / "segmentations",
    output_dir=ROOT_PATH / "data" / "cropped_files",
)


def load_yolo_model(paths: CroppingPaths) -> YOLO:
    """Load the configured YOLO model."""
    return YOLO(str(paths.model_weights))


def predict_imgs_single_directory(
    path_to_imgs: Path,
    conf: float,
    save_path: Path,
    name: str,
    model: YOLO,
) -> None:
    """Run YOLO prediction for all images in one directory."""
    model.predict(
        source=str(path_to_imgs),
        save=True,
        project=str(save_path),
        name=name,
        conf=conf,
        save_txt=True,
        max_det=YOLO_MAX_DET,
        save_conf=True,
        line_width=1,
        show_labels=False,
        show_conf=False,
        verbose=False,
    )


def delete_folder(save_path: Path) -> None:
    """Delete a folder and its contents if it exists."""
    shutil.rmtree(save_path, ignore_errors=True)


def parse_label_name(file_name: str) -> tuple[str, str]:
    """Extract tray id and row-column folder name from a YOLO label filename."""
    shards = file_name.split("_")
    if len(shards) < 3:
        raise ValueError(
            "Expected a YOLO label name with at least 3 underscore-separated parts, "
            f"got: {file_name}"
        )
    return shards[0], shards[2].replace(".txt", "")


def img_size(file_name: str, orig_data_path: Path) -> tuple[int, int]:
    """Return the size of the original PNG referenced by a YOLO label file."""
    tray_id, row_column = parse_label_name(file_name)
    img_path = (
        orig_data_path
        / f"Tray_{tray_id}"
        / row_column
        / file_name.replace(
            ".txt",
            ".png",
        )
    )
    with Image.open(img_path) as image:
        return image.size


def create_dataframe(label_path: Path, delimiter: str = " ") -> pd.DataFrame:
    """Load YOLO text labels into a normalized DataFrame."""
    if not label_path.exists():
        return pd.DataFrame(columns=YOLO_COLUMNS)

    rows: list[list[object]] = []
    for file_path in sorted(path for path in label_path.iterdir() if path.is_file()):
        labels_df = pd.read_csv(file_path, delimiter=delimiter, header=None)
        for _, row in labels_df.iterrows():
            rows.append([file_path.name, *row.tolist()[1:]])

    return pd.DataFrame(rows, columns=YOLO_COLUMNS)


def add_size_to_dataframe(df: pd.DataFrame, orig_data_path: Path) -> pd.DataFrame:
    """Add original image sizes and pixel-space bounding-box dimensions."""
    updated_df = df.copy()
    for row in updated_df.itertuples(index=True):
        width, height = img_size(str(row.name), orig_data_path)
        updated_df.at[row.Index, "real_width"] = width
        updated_df.at[row.Index, "real_height"] = height
        updated_df.at[row.Index, "real_center_x"] = float(row.center_x) * float(width)
        updated_df.at[row.Index, "real_center_y"] = float(row.center_y) * float(height)
        updated_df.at[row.Index, "bb_width"] = float(row.width) * float(width)
        updated_df.at[row.Index, "bb_height"] = float(row.height) * float(height)
    return updated_df


def bb_adding_smallbox(df: pd.DataFrame) -> pd.DataFrame:
    """Mark boxes whose width and height are within the small-box threshold."""
    updated_df = df.copy()
    updated_df["smallBox"] = updated_df.apply(
        lambda row: (
            "yes"
            if row["bb_width"] <= SMALL_BOX_LIMIT
            and row["bb_height"] <= SMALL_BOX_LIMIT
            else "no"
        ),
        axis=1,
    )
    return updated_df


def bb_small_box_edit(df: pd.DataFrame) -> pd.DataFrame:
    """Invalidate all rows after the first large box, matching legacy behavior."""
    updated_df = df.copy()
    box_already_seen = False
    for row in updated_df.itertuples(index=True):
        if box_already_seen:
            updated_df.at[row.Index, "smallBox"] = "no"
        if row.smallBox == "no":
            box_already_seen = True
    return updated_df


def finding_bounding_box_dataframe(
    df: pd.DataFrame,
) -> tuple[
    str | None,
    float | None,
    float | None,
    float | None,
    float | None,
    pd.DataFrame,
]:
    """Select the best small-box reference, preserving the original scoring logic."""
    updated_df = df.copy()
    updated_df["best"] = 0.0

    max_conf = float("-inf")
    best_name = None
    best_center_x = None
    best_center_y = None
    best_width = None
    best_height = None
    count_best = 0

    for index, row in updated_df.iterrows():
        conf_value = float(row["conf"])
        bb_width = float(row["bb_width"])
        bb_height = float(row["bb_height"])
        small_box = row["smallBox"]

        if (
            bb_width < SMALL_BOX_LIMIT
            and bb_height < SMALL_BOX_LIMIT
            and small_box == "yes"
        ):
            if conf_value > max_conf:
                max_conf = conf_value
                best_name = str(row["name"])
                best_center_x = float(row["center_x"])
                best_center_y = float(row["center_y"])
                best_width = float(row["real_width"])
                best_height = float(row["real_height"])
                count_best += 1

        updated_df.at[index, "best"] = count_best if conf_value == max_conf else 0.0

    return (
        best_name,
        best_center_x,
        best_center_y,
        best_width,
        best_height,
        updated_df,
    )


def df_write(best_name: str, df: pd.DataFrame, path_of_csv: Path) -> Path:
    """Write bounding-box diagnostics for the selected crop reference."""
    tray_id, row_column = parse_label_name(best_name)
    output_dir = path_of_csv / tray_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row_column}.csv"

    column_values: list[float] = []
    for row in df.itertuples(index=False):
        try:
            column_values.append(float(row.best))
        except ValueError:
            continue

    if column_values:
        max_value = max(column_values)
        for index, row in enumerate(df.itertuples(index=False)):
            try:
                value = float(row.best)
            except ValueError:
                continue
            df.at[index, "best"] = 1.0 if value == max_value else 0.0

    df.to_csv(output_path, index=False)
    return output_path


def prepare_cropping_img(best_name: str, orig_data_path: Path) -> list[Path]:
    """Return all PNG images in the source sequence referenced by ``best_name``."""
    tray_id, row_column = parse_label_name(best_name)
    folder_path = orig_data_path / f"Tray_{tray_id}" / row_column
    return sorted(path for path in folder_path.iterdir() if path.suffix == ".png")


def create_save_path(img_path: Path, save_address: Path) -> Path:
    """Return the output path where one cropped image should be written."""
    shards = img_path.name.split("_")
    if len(shards) < 3:
        raise ValueError(
            "Expected an image file name with at least 3 underscore-separated parts, "
            f"got: {img_path.name}"
        )

    row_column = shards[-1].replace(".png", "")
    tray_id = shards[-3]
    save_dir = save_address / f"Tray_{tray_id}" / row_column
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / img_path.name


def cropping_img(
    best_center_x: float,
    best_center_y: float,
    best_width: float,
    best_height: float,
    absolute_paths: list[Path],
    save_address: Path,
) -> None:
    """Crop each sequence image around the selected YOLO reference box."""
    for img_path in absolute_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            LOGGER.warning("Could not load image for cropping: %s", img_path)
            continue

        center_x = int(float(best_center_x) * float(best_width))
        center_y = int(float(best_center_y) * float(best_height))
        top_left_x = max(center_x - CROP_SIZE[0] // 2, 0)
        top_left_y = max(center_y - CROP_SIZE[1] // 2, 0)

        cropped_image = image[
            top_left_y : top_left_y + CROP_SIZE[1],
            top_left_x : top_left_x + CROP_SIZE[0],
        ]
        output_path = create_save_path(img_path, save_address)
        cv2.imwrite(str(output_path), cropped_image)


def print_folders_with_images(folder: Path) -> list[Path]:
    """Return all directories under ``folder`` that directly contain PNG images."""
    directory_paths = [
        path for path in folder.rglob("*") if path.is_dir() and any(path.glob("*.png"))
    ]
    directory_paths.sort()
    return directory_paths


def crop(paths: CroppingPaths) -> None:
    """Run the full cropping pipeline using repo-relative defaults."""
    save_path = paths.output_dir / "runs"
    label_path = save_path / YOLO_PROJECT_NAME / "labels"
    path_of_csv = paths.output_dir / "bounding_boxes"
    model = load_yolo_model(paths)
    directory_paths = print_folders_with_images(paths.input_dir)

    for directory_path in directory_paths:
        delete_folder(save_path)
        predict_imgs_single_directory(
            directory_path,
            YOLO_CONFIDENCE,
            save_path,
            YOLO_PROJECT_NAME,
            model,
        )
        try:
            df = create_dataframe(label_path)
            if df.empty:
                continue
            df = add_size_to_dataframe(df, paths.input_dir)
            df = bb_adding_smallbox(df)
            df = bb_small_box_edit(df)
            best_name, best_center_x, best_center_y, best_width, best_height, df = (
                finding_bounding_box_dataframe(df)
            )
            if best_name is None:
                continue
            df_write(best_name, df, path_of_csv)
            absolute_paths = prepare_cropping_img(best_name, paths.input_dir)
            cropping_img(
                best_center_x if best_center_x is not None else 0.0,
                best_center_y if best_center_y is not None else 0.0,
                best_width if best_width is not None else 0.0,
                best_height if best_height is not None else 0.0,
                absolute_paths,
                paths.output_dir,
            )
        except Exception as exc:
            LOGGER.warning(
                "An error occurred while cropping %s: %s", directory_path, exc
            )
            continue


predictImgsSINGLEDirectory = predict_imgs_single_directory
bb_smallBox_edit = bb_small_box_edit
finding_BB_DF = finding_bounding_box_dataframe


def main() -> None:
    """Run cropping with the local path definitions above."""
    crop(DEFAULT_CROPPING_PATHS)


if __name__ == "__main__":
    main()
