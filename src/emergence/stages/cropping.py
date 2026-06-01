"""YOLO crop pipeline with a retry that disables the small-box invalidation rule.

If a directory fails with ``boxes_invalidated_after_first_large_box``, it retries
selection for that directory only, keeping the same threshold but skipping the
rule that invalidates all boxes after the first large one.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from emergence.path_config import (
    CONFIG_PATH,
    get_required_string,
    load_stage_config,
    resolve_config_path,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

CROP_SIZE: tuple[int, int] = (75, 75)
BB_THRESHOLD: int = 55
YOLO_CONF: float = 0.6
YOLO_NAME: str = "detect"
YOLO_MAX_DET: int = 5
REPORT_FILENAME: str = "crop_run_report_retry_no_small_box_edit.csv"

YOLO_COLUMNS: list[str] = ["name", "center_x", "center_y", "width", "height", "conf"]


@dataclass(frozen=True)
class CropSegmentsPaths:
    """Filesystem paths required by the crop-segments pipeline."""

    model_weights: Path
    input_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class CropReference:
    """Reference bounding box used for cropping a full image sequence."""

    name: str
    center_x: float
    center_y: float
    image_width: float
    image_height: float


@dataclass(frozen=True)
class DirectoryResult:
    """Outcome of processing a single tray/row-column directory."""

    sequence_dir: str
    status: str
    reason: str
    label_files: int
    detections: int
    source_images: int
    best_name: str | None = None
    output_csv: str | None = None


def load_crop_segments_paths(config_path: Path = CONFIG_PATH) -> CropSegmentsPaths:
    """Load crop-segment paths from the shared TOML configuration."""
    context, crop_config = load_stage_config("crop_segments", config_path)

    return CropSegmentsPaths(
        model_weights=resolve_config_path(
            context,
            get_required_string(crop_config, "model_weights", config_path),
            "paths.crop_segments.model_weights",
        ),
        input_dir=resolve_config_path(
            context,
            get_required_string(crop_config, "input_dir", config_path),
            "paths.crop_segments.input_dir",
            include_project_name=True,
        ),
        output_dir=resolve_config_path(
            context,
            get_required_string(crop_config, "output_dir", config_path),
            "paths.crop_segments.output_dir",
            include_project_name=True,
        ),
    )


def load_yolo_model(model_weights: Path) -> YOLO:
    """Load the YOLO model from the given weights file."""
    return YOLO(str(model_weights))


def predict_imgs_single_directory(
    path_to_imgs: Path,
    conf: float,
    save_path: Path,
    name: str,
    model: YOLO,
) -> None:
    """Run YOLO prediction on a single directory of images."""
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


def delete_folder(folder_path: Path) -> None:
    """Delete a folder and its contents if it exists."""
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(folder_path, ignore_errors=True)


def format_sequence_label(sequence_dir: Path) -> str:
    """Return a human-readable tray/box label for logging."""
    return f"{sequence_dir.parent.name} / {sequence_dir.name}"


def parse_label_name(label_name: str) -> tuple[str, str]:
    """Extract tray identifier and row-column directory from a label filename.

    Args:
        label_name: Label filename (e.g. ``001_date_1-2.txt``).

    Returns:
        ``(tray_id, row_col_dir)`` tuple.

    Raises:
        ValueError: If the filename has fewer than 3 underscore-separated parts.
    """
    shards = label_name.split("_")
    if len(shards) < 3:
        raise ValueError(
            "Expected label filename with at least 3 underscore-separated parts, "
            f"got: {label_name}"
        )
    tray_id = shards[0]
    row_col_dir = shards[2].replace(".txt", "")
    return tray_id, row_col_dir


def img_size(file_name: str, orig_data_path: Path) -> tuple[int, int]:
    """Return pixel dimensions of the source image referenced by a label file."""
    tray_id, row_col_dir = parse_label_name(file_name)
    img_file = file_name.replace(".txt", ".png")
    img_path = orig_data_path / f"Tray_{tray_id}" / row_col_dir / img_file
    with Image.open(img_path) as img:
        width, height = img.size
    return width, height


def create_empty_dataframe() -> pd.DataFrame:
    """Return an empty YOLO label DataFrame with the expected columns."""
    return pd.DataFrame(columns=YOLO_COLUMNS)


def create_dataframe(label_path: Path, delimiter: str = " ") -> pd.DataFrame:
    """Build a deterministic DataFrame from YOLO label files.

    Args:
        label_path: Directory containing ``.txt`` label files.
        delimiter: Column delimiter used in label files.

    Returns:
        DataFrame with one row per detected bounding box.
    """
    if not label_path.exists():
        return create_empty_dataframe()

    rows: list[list[Any]] = []
    label_files = sorted(
        file_path for file_path in label_path.iterdir() if file_path.is_file()
    )
    for file_path in label_files:
        df_row = pd.read_csv(str(file_path), delimiter=delimiter, header=None)
        if len(df_row.columns) < 6:
            logger.warning("Skipping malformed label file: %s", file_path)
            continue
        for _, row in df_row.iterrows():
            rows.append([file_path.name, *row.tolist()[1:]])

    if not rows:
        return create_empty_dataframe()

    return pd.DataFrame(rows, columns=YOLO_COLUMNS)


def add_size_to_dataframe(df: pd.DataFrame, orig_data_path: Path) -> pd.DataFrame:
    """Append real image sizes and pixel-space bounding box dimensions."""
    if df.empty:
        return df.copy()

    updated_df = df.copy()
    for row in updated_df.itertuples(index=True):
        index = row.Index
        width, height = img_size(str(row.name), orig_data_path)
        real_center_x = float(row.center_x) * float(width)
        real_center_y = float(row.center_y) * float(height)
        bb_width = float(row.width) * float(width)
        bb_height = float(row.height) * float(height)

        updated_df.at[index, "real_width"] = width  # noqa: PD008
        updated_df.at[index, "real_height"] = height  # noqa: PD008
        updated_df.at[index, "real_center_x"] = real_center_x  # noqa: PD008
        updated_df.at[index, "real_center_y"] = real_center_y  # noqa: PD008
        updated_df.at[index, "bb_width"] = bb_width  # noqa: PD008
        updated_df.at[index, "bb_height"] = bb_height  # noqa: PD008

    return updated_df


def bb_adding_smallbox(df: pd.DataFrame, threshold: int = BB_THRESHOLD) -> pd.DataFrame:
    """Mark boxes as small when both dimensions are <= threshold."""
    updated_df = df.copy()
    updated_df["smallBox"] = updated_df.apply(
        lambda row: (
            "yes"
            if row["bb_width"] <= threshold and row["bb_height"] <= threshold
            else "no"
        ),
        axis=1,
    )
    return updated_df


def bb_small_box_edit(df: pd.DataFrame) -> pd.DataFrame:
    """Invalidate all rows after the first large box."""
    updated_df = df.copy()
    large_box_seen = False
    for row in updated_df.itertuples(index=True):
        index = row.Index
        if large_box_seen:
            updated_df.at[index, "smallBox"] = "no"  # noqa: PD008
        if row.smallBox == "no":
            large_box_seen = True
    return updated_df


def get_failure_reason(df: pd.DataFrame, threshold: int = BB_THRESHOLD) -> str:
    """Explain why no valid reference box could be selected."""
    if df.empty:
        return "no_yolo_labels"

    threshold_mask = (df["bb_width"] <= threshold) & (df["bb_height"] <= threshold)
    strict_mask = (df["bb_width"] < threshold) & (df["bb_height"] < threshold)
    eligible_mask = strict_mask & (df["smallBox"] == "yes")

    if not threshold_mask.any():
        return "all_boxes_too_large"
    if threshold_mask.any() and not strict_mask.any():
        return "only_boundary_boxes_at_threshold"
    if strict_mask.any() and not eligible_mask.any():
        return "boxes_invalidated_after_first_large_box"
    return "no_reference_box"


def find_best_bounding_box(
    df: pd.DataFrame, threshold: int = BB_THRESHOLD
) -> tuple[CropReference | None, pd.DataFrame, str]:
    """Find the highest-confidence valid reference bounding box.

    Args:
        df: DataFrame with ``smallBox`` column already populated.
        threshold: Maximum side length (px) for a box to qualify.

    Returns:
        ``(reference, annotated_df, reason)`` where ``reason`` is ``"cropped"``
        on success or a failure code otherwise.
    """
    updated_df = df.copy()
    updated_df["best"] = 0.0

    if updated_df.empty:
        return None, updated_df, get_failure_reason(updated_df, threshold)

    max_conf = float("-inf")
    best_index: int | None = None
    best_reference: CropReference | None = None

    for index, row in updated_df.iterrows():
        conf_value = float(row["conf"])
        bb_width = float(row["bb_width"])
        bb_height = float(row["bb_height"])
        small_box = row["smallBox"]

        if bb_width < threshold and bb_height < threshold and small_box == "yes":  # noqa: SIM102
            if conf_value > max_conf:
                max_conf = conf_value
                best_index = int(index)
                best_reference = CropReference(
                    name=str(row["name"]),
                    center_x=float(row["center_x"]),
                    center_y=float(row["center_y"]),
                    image_width=float(row["real_width"]),
                    image_height=float(row["real_height"]),
                )

    if best_index is None or best_reference is None:
        return None, updated_df, get_failure_reason(updated_df, threshold)

    updated_df.at[best_index, "best"] = 1.0  # noqa: PD008
    return best_reference, updated_df, "cropped"


def select_reference(
    df: pd.DataFrame,
    threshold: int = BB_THRESHOLD,
    apply_small_box_edit_rule: bool = True,
) -> tuple[CropReference | None, pd.DataFrame, str]:
    """Apply selection rules and return the best crop reference, if any."""
    selected_df = bb_adding_smallbox(df, threshold)
    if apply_small_box_edit_rule:
        selected_df = bb_small_box_edit(selected_df)
    return find_best_bounding_box(selected_df, threshold)


def df_write(reference_name: str, df: pd.DataFrame, path_of_csv: Path) -> Path:
    """Write bounding-box diagnostics to a per-sequence CSV file."""
    tray_id, row_col_dir = parse_label_name(reference_name)
    output_dir = path_of_csv / tray_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row_col_dir}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def prepare_cropping_img(reference_name: str, orig_data_path: Path) -> list[Path]:
    """Return all PNG files from the sequence referenced by the winning label."""
    tray_id, row_col_dir = parse_label_name(reference_name)
    folder_path = orig_data_path / f"Tray_{tray_id}" / row_col_dir
    return sorted(path for path in folder_path.iterdir() if path.suffix == ".png")


def create_save_path(img_path: Path, save_address: Path) -> Path:
    """Construct the output path for a cropped image.

    Args:
        img_path: Source image path.
        save_address: Root output directory.

    Returns:
        Full output path (parent directories are created).

    Raises:
        ValueError: If the filename has fewer than 3 underscore-separated parts.
    """
    shards = img_path.name.split("_")
    if len(shards) < 3:
        raise ValueError(
            "Expected image filename with at least 3 underscore-separated parts, "
            f"got: {img_path.name}"
        )

    row_col_dir = shards[-1].replace(".png", "")
    tray_id = shards[-3]
    save_dir = save_address / f"Tray_{tray_id}" / row_col_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / img_path.name


def calculate_crop_bounds(
    center_x: int,
    center_y: int,
    image_width: int,
    image_height: int,
    crop_size: tuple[int, int] = CROP_SIZE,
) -> tuple[int, int, int, int]:
    """Return an in-bounds crop window, shifting it inward if the center is near an edge."""
    crop_width, crop_height = crop_size
    max_top_left_x = max(image_width - crop_width, 0)
    max_top_left_y = max(image_height - crop_height, 0)

    top_left_x = min(max(center_x - crop_width // 2, 0), max_top_left_x)
    top_left_y = min(max(center_y - crop_height // 2, 0), max_top_left_y)
    bottom_right_x = min(top_left_x + crop_width, image_width)
    bottom_right_y = min(top_left_y + crop_height, image_height)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def cropping_img(
    reference: CropReference, absolute_paths: list[Path], save_address: Path
) -> None:
    """Crop all images in a sequence around the selected reference box."""
    for img_path in absolute_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read image for cropping: %s", img_path)
            continue

        center_x = int(reference.center_x * reference.image_width)
        center_y = int(reference.center_y * reference.image_height)
        image_height, image_width = image.shape[:2]

        if image_width < CROP_SIZE[0] or image_height < CROP_SIZE[1]:
            logger.warning(
                "Source image is smaller than crop size for %s: %sx%s",
                img_path,
                image_width,
                image_height,
            )

        top_left_x, top_left_y, bottom_right_x, bottom_right_y = calculate_crop_bounds(
            center_x=center_x,
            center_y=center_y,
            image_width=image_width,
            image_height=image_height,
        )

        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        output_path = create_save_path(img_path, save_address)
        cv2.imwrite(str(output_path), cropped_image)


def print_folders_with_images(folder: Path) -> list[Path]:
    """Return sorted directories beneath folder that directly contain PNG files."""
    directory_paths = [
        child
        for child in folder.rglob("*")
        if child.is_dir() and any(child.glob("*.png"))
    ]
    directory_paths.sort()
    return directory_paths


def process_directory(
    sequence_dir: Path,
    paths: CropSegmentsPaths,
    model: YOLO,
    save_path: Path,
    label_path: Path,
    csv_output_dir: Path,
) -> DirectoryResult:
    """Process a single sequence directory and return a structured result."""
    sequence_label = format_sequence_label(sequence_dir)
    delete_folder(save_path)
    predict_imgs_single_directory(sequence_dir, YOLO_CONF, save_path, YOLO_NAME, model)

    df = create_dataframe(label_path)
    label_files = df["name"].nunique() if not df.empty else 0
    detections = len(df)
    source_images = len(list(sequence_dir.glob("*.png")))

    if df.empty:
        logger.warning("Skipping %s: no YOLO labels found", sequence_label)
        return DirectoryResult(
            sequence_dir=str(sequence_dir),
            status="skipped",
            reason="no_yolo_labels",
            label_files=label_files,
            detections=detections,
            source_images=source_images,
        )

    base_df = add_size_to_dataframe(df, paths.input_dir)
    reference, result_df, reason = select_reference(base_df)
    status = "cropped"

    if reference is None and reason == "boxes_invalidated_after_first_large_box":
        logger.info(
            "Retrying %s without small-box invalidation rule after baseline failure",
            sequence_label,
        )
        reference, result_df, reason = select_reference(
            base_df,
            threshold=BB_THRESHOLD,
            apply_small_box_edit_rule=False,
        )
        if reference is not None:
            status = "cropped_retry_no_small_box_edit"
            reason = "cropped_retry_no_small_box_edit"

    if reference is None:
        logger.warning("Skipping %s: %s", sequence_label, reason)
        return DirectoryResult(
            sequence_dir=str(sequence_dir),
            status="skipped",
            reason=reason,
            label_files=label_files,
            detections=detections,
            source_images=source_images,
        )

    output_csv = df_write(reference.name, result_df, csv_output_dir)
    absolute_paths = prepare_cropping_img(reference.name, paths.input_dir)
    cropping_img(reference, absolute_paths, paths.output_dir)

    return DirectoryResult(
        sequence_dir=str(sequence_dir),
        status=status,
        reason=reason,
        label_files=label_files,
        detections=detections,
        source_images=len(absolute_paths),
        best_name=reference.name,
        output_csv=str(output_csv),
    )


def write_report(results: list[DirectoryResult], report_path: Path) -> Path:
    """Write a CSV report summarising processed and skipped directories."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(asdict(result) for result in results)
    report_df.to_csv(report_path, index=False)
    return report_path


def crop(paths: CropSegmentsPaths, report_name: str = REPORT_FILENAME) -> Path:
    """Run the full crop pipeline and return the path to the CSV report.

    Args:
        paths: Configured filesystem paths for this stage.
        report_name: Filename for the summary CSV written to ``output_dir``.

    Returns:
        Path to the written report CSV.
    """
    save_path = paths.output_dir / "runs"
    label_path = save_path / YOLO_NAME / "labels"
    csv_output_dir = paths.output_dir / "bounding_boxes"

    model = load_yolo_model(paths.model_weights)
    directory_paths = print_folders_with_images(paths.input_dir)
    results: list[DirectoryResult] = []
    total_sequences = len(directory_paths)

    for index, sequence_dir in enumerate(directory_paths, start=1):
        logger.info(
            "Processing %s (%s/%s)",
            format_sequence_label(sequence_dir),
            index,
            total_sequences,
        )
        try:
            result = process_directory(
                sequence_dir=sequence_dir,
                paths=paths,
                model=model,
                save_path=save_path,
                label_path=label_path,
                csv_output_dir=csv_output_dir,
            )
            results.append(result)
        except Exception:
            logger.exception("Error processing directory %s", sequence_dir)
            results.append(
                DirectoryResult(
                    sequence_dir=str(sequence_dir),
                    status="skipped",
                    reason="unexpected_error",
                    label_files=0,
                    detections=0,
                    source_images=len(list(sequence_dir.glob("*.png"))),
                )
            )

    report_path = paths.output_dir / report_name
    write_report(results, report_path)
    logger.info("Cropping finished. Report written to %s", report_path)
    return report_path



