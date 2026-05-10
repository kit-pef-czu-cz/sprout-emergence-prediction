"""Create time-series NumPy datasets from the finetuning crop directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class PreprocessingPaths:
    """Filesystem paths required by the finetuning preprocessing script."""

    input_dir: Path
    annotation_file: Path
    output_dir: Path
    image_size: int = 75


# Advanced users: edit these paths directly in this script when needed.
ROOT_PATH = Path("/home/vasakjakub/fenotypizace")
DEFAULT_PREPROCESSING_PATHS = PreprocessingPaths(
    input_dir=ROOT_PATH / "data" / "cropped_files",
    annotation_file=(
        ROOT_PATH
        / "models"
        / "prediction_model"
        / "annotations"
        / "emergence_annotations.xlsx"
    ),
    output_dir=ROOT_PATH / "data" / "numpy_dataset",
)


def load_annotations(annotation_path: Path, use_trays: list[int]) -> pd.DataFrame:
    """Load emergence annotations and filter them to the selected trays."""
    df = pd.read_excel(annotation_path)
    df = df.loc[df["tray_location"].isin(use_trays)].copy()
    df["time_of_first_occurence_png"] = (
        df["time_of_first_occurence"].astype(str)
        + "_"
        + df["well_id"].astype(str)
        + ".png"
    )
    return df.sort_values(by=["tray_location", "well_id"])


def index_containing_substring(img_list: list[str], first_occurence: str) -> int:
    """Return the index of the first image whose name contains the target text."""
    for index, image_name in enumerate(img_list):
        if first_occurence in image_name:
            return index
    return -1


def load_from_dir(
    df: pd.DataFrame,
    data_directory: Path,
    crop_size: int,
    length: int,
    names: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]:
    """Load crops from disk and build the flat image/label arrays.

    The undersized-image fallback is intentionally preserved: if a crop is smaller
    than ``crop_size``, it is replaced with ``previous_img`` rather than dropped.
    """
    labels: list[int] = []
    imgs_array: list[np.ndarray] = []
    previous_img: np.ndarray | None = None
    image_names: list[str] = []

    for item in tqdm(
        df.itertuples(index=False), total=len(df), desc="Processing images"
    ):
        directory = data_directory / f"Tray_{item.tray_location}" / str(item.well_id)
        files = sorted(str(path) for path in directory.glob("*.png"))

        germ_class = 0
        emerged_index = index_containing_substring(
            files, str(item.time_of_first_occurence)
        )
        if emerged_index == -1:
            print(f"Index not found for {item.time_of_first_occurence_png}")
            continue

        files_ranged = (
            files if names else files[emerged_index - length : emerged_index + length]
        )
        first_occurrence_name = item.time_of_first_occurence_png

        for filename in files_ranged:
            filename_occurrence = Path(filename).name
            if first_occurrence_name == filename_occurrence:
                germ_class = 1

            with Image.open(filename) as image:
                np_image = np.array(image)

            if np_image.shape[0] < crop_size or np_image.shape[1] < crop_size:
                np_image = previous_img
                print(f"Image error, image is too small: {filename_occurrence}")
            else:
                previous_img = np_image

            if np_image is not None:
                imgs_array.append(np_image)
                labels.append(germ_class)

            if names:
                image_names.append(filename_occurrence)

    images_array = np.array(imgs_array)
    labels_array = np.array(labels)
    print(images_array.shape)
    print(labels_array.shape)

    if names:
        return images_array, labels_array, image_names
    return images_array, labels_array


def generate_samples(
    images: np.ndarray,
    labels: np.ndarray,
    step: int,
    time_steps: int,
    image_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding-window samples for temporal model training/evaluation."""
    img_trays = np.array([images[0:time_steps]])
    annotations_array = np.array([0])

    if image_names is not None:
        image_names_trays = np.array([image_names[0:time_steps]])
        for index in tqdm(
            range(1, images.shape[0] - time_steps, step),
            desc="Generating test samples with image names",
        ):
            img_trays = np.append(img_trays, [images[index : index + time_steps]], 0)
            annotations_array = np.append(annotations_array, [labels[index]], 0)
            image_names_trays = np.append(
                image_names_trays,
                [image_names[index : index + time_steps]],
                0,
            )
        return img_trays, annotations_array, image_names_trays

    for index in tqdm(
        range(1, images.shape[0] - time_steps, step),
        desc="Generating train/val samples",
    ):
        img_trays = np.append(img_trays, [images[index : index + time_steps]], 0)
        annotations_array = np.append(annotations_array, [labels[index]], 0)
    return img_trays, annotations_array


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
) -> None:
    """Load images and annotations from disk and save them as NumPy datasets."""
    save_path.mkdir(parents=True, exist_ok=True)
    annotations_df = load_annotations(annotations_path, use_trays)

    for time_step in time_steps:
        print(f"Step size: {time_step}")

        for size in data_range:
            if size < time_step:
                print(
                    f"Warning: Sequence length ({size}) is shorter than required time step ({time_step})"
                )
                continue

            print(f"Range size: {size}\n")

            if dataset in {"train", "val"}:
                raw_images, raw_annotations = load_from_dir(
                    annotations_df,
                    images_path,
                    img_crop_size,
                    size,
                )
                samples, labels = generate_samples(
                    raw_images,
                    raw_annotations,
                    slide_step,
                    time_step,
                )
            elif dataset == "test":
                raw_images, raw_annotations, image_names = load_from_dir(
                    annotations_df,
                    images_path,
                    img_crop_size,
                    size,
                    names=True,
                )
                samples, labels, image_names_trays = generate_samples(
                    raw_images,
                    raw_annotations,
                    slide_step,
                    time_step,
                    image_names,
                )
                np.save(
                    save_path
                    / f"data_{img_crop_size}_{time_step}_{size}_{dataset}_img_names.npy",
                    image_names_trays,
                )
            else:
                print("Unknown dataset. Please use train, val or test.")
                return

            print(samples.shape)
            print()
            np.save(
                save_path / f"data_{img_crop_size}_{time_step}_{size}_{dataset}.npy",
                samples,
            )
            np.save(
                save_path
                / f"data_{img_crop_size}_{time_step}_{size}_{dataset}_ann.npy",
                labels,
            )


def load_dataset() -> tuple[
    Path, Path, Path, list[int], list[int], list[int], int, int
]:
    """Return the default finetuning dataset paths and tray splits."""
    paths = DEFAULT_PREPROCESSING_PATHS

    trays_train = [10, 11, 12, 13, 14, 17, 16, 18, 41, 42, 45]
    trays_val = [15, 40, 44, 47]
    trays_test = [4, 13, 16, 43, 46, 48]
    slide_step = 1

    return (
        paths.input_dir,
        paths.annotation_file,
        paths.output_dir,
        trays_train,
        trays_val,
        trays_test,
        slide_step,
        paths.image_size,
    )


def main() -> None:
    """Create train/val/test NumPy datasets for the finetuning temporal models."""
    n_time_steps = [3]
    n_data_range = [9]

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

    load_data_images(
        images_path=data_images_path,
        annotations_path=annotation_file,
        use_trays=trays_train,
        data_range=n_data_range,
        time_steps=n_time_steps,
        img_crop_size=img_size,
        slide_step=slide_step,
        dataset="train",
        save_path=save_path,
    )
    load_data_images(
        images_path=data_images_path,
        annotations_path=annotation_file,
        use_trays=trays_val,
        data_range=n_data_range,
        time_steps=n_time_steps,
        img_crop_size=img_size,
        slide_step=slide_step,
        dataset="val",
        save_path=save_path,
    )
    load_data_images(
        images_path=data_images_path,
        annotations_path=annotation_file,
        use_trays=trays_test,
        data_range=n_data_range,
        time_steps=n_time_steps,
        img_crop_size=img_size,
        slide_step=slide_step,
        dataset="test",
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
