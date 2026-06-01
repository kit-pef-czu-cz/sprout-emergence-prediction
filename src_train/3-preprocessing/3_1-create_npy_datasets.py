"""Create time-series NumPy datasets from cropped seed images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


# Advanced users: edit these paths and parameters directly in this script.
ROOT_PATH = Path("/home/vasakjakub/fenotypizace")
PROJECT_NAME = "nabila"
INPUT_DIR = ROOT_PATH / "data" / "interim" / "cropped_files" / PROJECT_NAME
ANNOTATION_FILE = ROOT_PATH / "data" / "external" / f"{PROJECT_NAME}_annotations.xlsx"
OUTPUT_DIR = ROOT_PATH / "data" / "processed" / "numpy_dataset" / "test_val_train"

IMAGE_SIZE = 75
# Sliding-window stride used within each tray/well sequence.
SLIDE_STEP = 1
N_TIME_STEPS = [3]
N_DATA_RANGE = [9]

# Ohnoutkova trays
# TRAYS_TRAIN = [10, 12, 13, 14, 15, 17, 16, 18, 41, 42, 44, 45]
# TRAYS_VAL = [11, 40, 47]
# TRAYS_TEST = [43, 46, 48]

# Nabila trays
TRAYS_TRAIN = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 15]
TRAYS_VAL = [5, 14]
TRAYS_TEST = [4, 13, 16]


@dataclass(frozen=True)
class PreprocessingPaths:
    """Filesystem paths and image size required by preprocessing.

    Attributes:
        input_dir: Directory containing cropped images grouped by tray and well.
        annotation_file: Excel file with emergence annotations for each well.
        output_dir: Directory where generated NumPy arrays are saved.
        image_size: Expected crop width and height in pixels.
    """

    input_dir: Path
    annotation_file: Path
    output_dir: Path
    image_size: int = 75


DEFAULT_PREPROCESSING_PATHS = PreprocessingPaths(
    input_dir=INPUT_DIR,
    annotation_file=ANNOTATION_FILE,
    output_dir=OUTPUT_DIR,
    image_size=IMAGE_SIZE,
)


@dataclass
class WellImageSequence:
    """Images, labels, and names loaded from one tray/well directory."""

    images: list[np.ndarray]
    labels: list[int]
    image_names: list[str]
    tray_location: int
    well_id: str


def read_annotations(annotation_path: Path) -> pd.DataFrame:
    """Read emergence annotations and add expected occurrence filenames.

    Args:
        annotation_path: Excel file with emergence timestamps, tray IDs, and well IDs.

    Returns:
        Annotations with the expected first emergence PNG filename added in
        ``time_of_first_occurence_png``.
    """
    annotations = pd.read_excel(annotation_path)
    annotations["time_of_first_occurence_png"] = (
        annotations["time_of_first_occurence"].astype(str)
        + "_"
        + annotations["well_id"].astype(str)
        + ".png"
    )
    return annotations


def filter_annotations(annotations: pd.DataFrame, use_trays: list[int]) -> pd.DataFrame:
    """Filter annotations for selected trays while preserving sort order."""
    annotations = annotations.loc[annotations["tray_location"].isin(use_trays)].copy()
    annotations["time_of_first_occurence_png"] = (
        annotations["time_of_first_occurence"].astype(str)
        + "_"
        + annotations["well_id"].astype(str)
        + ".png"
    )
    return annotations.sort_values(by=["tray_location", "well_id"])


def load_annotations(annotation_path: Path, use_trays: list[int]) -> pd.DataFrame:
    """Load emergence annotations for the selected trays.

    Args:
        annotation_path: Excel file with emergence timestamps, tray IDs, and well IDs.
        use_trays: Tray IDs to keep for the current dataset split.

    Returns:
        Filtered annotations sorted by tray and well, with the expected first
        emergence PNG filename added in ``time_of_first_occurence_png``.
    """
    return filter_annotations(read_annotations(annotation_path), use_trays)


def index_containing_substring(
    img_list: list[str] | list[Path], first_occurence: str
) -> int:
    """Find the first image path that contains the target occurrence text.

    Args:
        img_list: Sorted image paths for one well.
        first_occurence: Timestamp or filename fragment marking emergence.

    Returns:
        Index of the first matching image, or ``-1`` when no image matches.
    """
    for index, image_name in enumerate(img_list):
        if first_occurence in str(image_name):
            return index
    return -1


def load_from_dir(
    df: pd.DataFrame,
    data_directory: Path,
    crop_size: int,
    length: int,
    names: bool = False,
) -> list[WellImageSequence]:
    """Load crops from disk as independent tray/well image sequences.

    The emergence class is set to ``1`` from the first occurrence image onward.
    Undersized crops reuse the previous valid image from the same well only.

    Args:
        df: Emergence annotations for the trays in the current split.
        data_directory: Root directory with ``Tray_<id>/<well_id>/*.png`` crops.
        crop_size: Minimum accepted crop width and height in pixels.
        length: Number of frames to keep before and after emergence.
        names: Whether to store image filenames alongside images and labels.

    Returns:
        Per-well sequences with images, binary labels, and optional filenames.
    """
    sequences: list[WellImageSequence] = []

    for item in tqdm(
        df.itertuples(index=False), total=len(df), desc="Processing images"
    ):
        directory = data_directory / f"Tray_{item.tray_location}" / str(item.well_id)
        files = sorted(directory.glob("*.png"))

        germ_class = 0
        sequence_images: list[np.ndarray] = []
        sequence_labels: list[int] = []
        sequence_image_names: list[str] = []
        previous_img: np.ndarray | None = None
        emerged_index = index_containing_substring(
            files, str(item.time_of_first_occurence)
        )
        if emerged_index == -1:
            LOGGER.warning(
                "Index not found for first occurrence image %s",
                item.time_of_first_occurence_png,
            )
            continue

        # Training/validation use a balanced window around emergence. Test data
        # keeps all images so predictions can be traced back to source filenames.
        files_ranged = (
            files if names else files[emerged_index - length : emerged_index + length]
        )
        first_occurrence_name = item.time_of_first_occurence_png

        for filename in files_ranged:
            filename_occurrence = filename.name
            if first_occurrence_name == filename_occurrence:
                germ_class = 1

            with Image.open(filename) as image:
                np_image = np.array(image)

            if np_image.shape[0] < crop_size or np_image.shape[1] < crop_size:
                if previous_img is None:
                    LOGGER.warning(
                        "Image is too small and no previous image exists for "
                        "Tray_%s/%s; skipping: %s",
                        item.tray_location,
                        item.well_id,
                        filename_occurrence,
                    )
                    continue
                LOGGER.warning(
                    "Image is too small and will be replaced with previous "
                    "image from the same well: %s",
                    filename_occurrence,
                )
                np_image = previous_img
            else:
                previous_img = np_image

            sequence_images.append(np_image)
            sequence_labels.append(germ_class)
            if names:
                sequence_image_names.append(filename_occurrence)

        if sequence_images:
            sequences.append(
                WellImageSequence(
                    images=sequence_images,
                    labels=sequence_labels,
                    image_names=sequence_image_names,
                    tray_location=item.tray_location,
                    well_id=str(item.well_id),
                )
            )

    LOGGER.info("Loaded %s well sequence(s)", len(sequences))
    return sequences


def generate_samples(
    sequences: list[WellImageSequence],
    time_steps: int,
    step: int = SLIDE_STEP,
    image_names: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding-window samples per tray/well sequence.

    Args:
        sequences: Per-well image sequences to sample.
        time_steps: Number of consecutive frames per output sample.
        step: Sliding-window stride between consecutive sample starts.
        image_names: Whether to return filename windows aligned with samples.

    Returns:
        Sliding-window image samples and last-frame labels. When ``image_names``
        is true, also returns filename arrays for test-set traceability.
    """
    if step < 1:
        raise ValueError("step must be a positive integer")
    if time_steps < 1:
        raise ValueError("time_steps must be a positive integer")

    img_trays: list[list[np.ndarray]] = []
    annotations: list[int] = []
    image_names_trays: list[list[str]] = []

    for sequence in tqdm(sequences, desc="Generating samples"):
        sequence_length = len(sequence.images)
        if sequence_length < time_steps:
            LOGGER.warning(
                "Skipping Tray_%s/%s: %s images is fewer than %s time steps.",
                sequence.tray_location,
                sequence.well_id,
                sequence_length,
                time_steps,
            )
            continue

        for start in range(0, sequence_length - time_steps + 1, step):
            end = start + time_steps
            img_trays.append(sequence.images[start:end])
            annotations.append(sequence.labels[end - 1])
            if image_names:
                image_names_trays.append(sequence.image_names[start:end])

    if image_names:
        return np.array(img_trays), np.array(annotations), np.array(image_names_trays)
    return np.array(img_trays), np.array(annotations)


def load_data_images(
    images_path: Path,
    annotations_path: Path,
    use_trays: list[int],
    save_path: Path,
    dataset: str,
    data_range: list[int],
    time_steps: list[int],
    img_crop_size: int = 75,
    slide_step: int = 1,
    annotations_df: pd.DataFrame | None = None,
) -> None:
    """Load images and annotations, then save the generated NumPy datasets.

    Args:
        images_path: Root directory with cropped images grouped by tray and well.
        annotations_path: Excel file with emergence annotations.
        use_trays: Tray IDs included in this dataset split.
        save_path: Directory where ``.npy`` files are written.
        dataset: Dataset split name. Supported values are ``train``, ``val``,
            and ``test``.
        data_range: Frame-window radii around first emergence to process.
        time_steps: Sequence lengths to generate for the temporal model.
        img_crop_size: Expected crop width and height in pixels. Defaults to 75.
        slide_step: Sliding-window stride within each tray/well sequence.
            Defaults to 1.
        annotations_df: Optional preloaded annotations. When provided, the Excel
            file is not reread for this split.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    all_annotations = (
        read_annotations(annotations_path) if annotations_df is None else annotations_df
    )
    split_annotations = filter_annotations(all_annotations, use_trays)

    for time_step in time_steps:
        LOGGER.info("Step size: %s", time_step)

        for size in data_range:
            if size < time_step:
                LOGGER.warning(
                    "Sequence length (%s) is shorter than required time step (%s)",
                    size,
                    time_step,
                )
                continue

            LOGGER.info("Range size: %s", size)

            if dataset in {"train", "val"}:
                raw_sequences = load_from_dir(
                    split_annotations,
                    images_path,
                    img_crop_size,
                    size,
                )
                samples, labels = generate_samples(
                    raw_sequences,
                    time_step,
                    step=slide_step,
                )
            elif dataset == "test":
                raw_sequences = load_from_dir(
                    split_annotations,
                    images_path,
                    img_crop_size,
                    size,
                    names=True,
                )
                samples, labels, image_names_trays = generate_samples(
                    raw_sequences,
                    time_step,
                    step=slide_step,
                    image_names=True,
                )
                np.save(
                    save_path
                    / f"data_{PROJECT_NAME}_{img_crop_size}_{time_step}_{size}_{dataset}_img_names.npy",
                    image_names_trays,
                )
            else:
                LOGGER.error(
                    "Unknown dataset %s. Please use train, val, or test.",
                    dataset,
                )
                return

            LOGGER.info("Generated samples shape: %s", samples.shape)
            np.save(
                save_path
                / f"data_{PROJECT_NAME}_{img_crop_size}_{time_step}_{size}_{dataset}.npy",
                samples,
            )
            np.save(
                save_path
                / f"data_{PROJECT_NAME}_{img_crop_size}_{time_step}_{size}_{dataset}_ann.npy",
                labels,
            )


def load_dataset() -> tuple[
    Path, Path, Path, list[int], list[int], list[int], int, int
]:
    """Return the configured finetuning dataset paths and tray splits.

    Returns:
        Input image directory, annotation file, output directory, train tray IDs,
        validation tray IDs, test tray IDs, sliding-window stride, and image size.
    """
    paths = DEFAULT_PREPROCESSING_PATHS

    return (
        paths.input_dir,
        paths.annotation_file,
        paths.output_dir,
        TRAYS_TRAIN,
        TRAYS_VAL,
        TRAYS_TEST,
        SLIDE_STEP,
        paths.image_size,
    )


def main() -> None:
    """Create train, validation, and test datasets for temporal model finetuning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    (
        data_images_path,
        annotation_file,
        save_path,
        trays_train,
        trays_val,
        trays_test,
        slide_step,
        img_size,
    ) = load_dataset()
    annotations_df = read_annotations(annotation_file)

    load_data_images(
        images_path=data_images_path,
        annotations_path=annotation_file,
        use_trays=trays_train,
        data_range=N_DATA_RANGE,
        time_steps=N_TIME_STEPS,
        img_crop_size=img_size,
        slide_step=slide_step,
        dataset="train",
        save_path=save_path,
        annotations_df=annotations_df,
    )
    load_data_images(
        images_path=data_images_path,
        annotations_path=annotation_file,
        use_trays=trays_val,
        data_range=N_DATA_RANGE,
        time_steps=N_TIME_STEPS,
        img_crop_size=img_size,
        slide_step=slide_step,
        dataset="val",
        save_path=save_path,
        annotations_df=annotations_df,
    )
    load_data_images(
        images_path=data_images_path,
        annotations_path=annotation_file,
        use_trays=trays_test,
        data_range=N_DATA_RANGE,
        time_steps=N_TIME_STEPS,
        img_crop_size=img_size,
        slide_step=slide_step,
        dataset="test",
        save_path=save_path,
        annotations_df=annotations_df,
    )


if __name__ == "__main__":
    main()
