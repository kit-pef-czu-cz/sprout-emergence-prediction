"""Evaluate finetuning temporal models against the saved NumPy test datasets."""

from __future__ import annotations

import logging
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tcn import TCN  # noqa: F401
from tensorflow import keras  # type: ignore

LOGGER = logging.getLogger(__name__)


# Advanced users: edit these paths and parameters directly in this script.
ROOT_PATH = Path("sprout-emergence-prediction")
PROJECT_NAME = "nabila"
IMAGE_SIZE = 75
TIME_STEP = 3
DATA_RANGE = 9
DATASET_SPLIT = "test"
MODEL_NAME_TEMPLATE = f"ef2_tcn_{TIME_STEP}-{DATA_RANGE}_org_nab_relearned"
PREDICTION_THRESHOLD = 0.5
CPROB_WINDOW = 3

MODEL_DIR = ROOT_PATH / "models" / "prediction_model"
NUMPY_DATASET_DIR = (
    ROOT_PATH / "data" / "processed" / "numpy_dataset" / "test_val_train"
)
OUTPUT_DIR = ROOT_PATH / "data" / "evaluation" / PROJECT_NAME
TEMP_DIR = OUTPUT_DIR / "temp"
INPUT_CSV = TEMP_DIR / "BASE.csv"
OUTPUT_CSV = TEMP_DIR / "BASE_conv.csv"


@dataclass(frozen=True)
class EvaluationPaths:
    """Filesystem paths and parameters required by finetuning evaluation."""

    model_dir: Path
    numpy_dataset_dir: Path
    output_dir: Path
    temp_dir: Path
    input_csv: Path
    output_csv: Path
    project_name: str
    image_size: int
    time_step: int
    data_range: int
    dataset_split: str
    model_name_template: str
    prediction_threshold: float


EVALUATION_PATHS = EvaluationPaths(
    model_dir=MODEL_DIR,
    numpy_dataset_dir=NUMPY_DATASET_DIR,
    output_dir=OUTPUT_DIR,
    temp_dir=TEMP_DIR,
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    project_name=PROJECT_NAME,
    image_size=IMAGE_SIZE,
    time_step=TIME_STEP,
    data_range=DATA_RANGE,
    dataset_split=DATASET_SPLIT,
    model_name_template=MODEL_NAME_TEMPLATE,
    prediction_threshold=PREDICTION_THRESHOLD,
)
TEMP_PATH = EVALUATION_PATHS.temp_dir
INPUT_FILE_PATH = EVALUATION_PATHS.input_csv
OUTPUT_FILE_PATH = EVALUATION_PATHS.output_csv


def delete_and_create(temp_path: Path) -> None:
    """Delete and recreate the temporary evaluation workspace.

    Args:
        temp_path: Directory to delete and recreate.

    Author:
        Jakub Vašák

    """
    shutil.rmtree(temp_path, ignore_errors=True)
    temp_path.mkdir(parents=True, exist_ok=True)


def import_npy(
    data_path: Path,
    names_path: Path,
    model_path: Path,
    annotations_path: Path,
    save_path: Path,
    temp_csv_path: Path,
) -> pd.DataFrame:
    """Load NumPy inputs, run predictions, and write evaluation CSVs.

    Args:
        data_path: Path to the data NPY file with image sequences.
        names_path: Path to the NPY file with image names.
        model_path: Path to the saved model file.
        annotations_path: Path to the NPY file with ground-truth annotations.
        save_path: Directory where evaluation CSV outputs are written.
        temp_csv_path: Path for the temporary CSV snapshot.

    Returns:
        DataFrame with combined names, annotations, and prediction columns.

    Author:
        Jakub Vašák

    """
    # Load data
    new_data = np.load(data_path)
    names = np.load(names_path)
    annot = np.load(annotations_path)

    # Load model and predict
    loaded_model = keras.models.load_model(str(model_path))
    predictions = loaded_model.predict(new_data)

    # Create DataFrames
    predictions_flat = predictions.flatten()
    labels = (predictions_flat >= EVALUATION_PATHS.prediction_threshold).astype(int)
    df_pred = pd.DataFrame(labels, columns=["prob"])

    df_names = pd.DataFrame(names).rename(columns={0: "names"}).iloc[:, 0]
    df_annot = pd.DataFrame(annot).rename(columns={0: "annot"}).iloc[:, 0]

    # Combine DataFrames
    df_combined = pd.concat([df_names, df_annot, df_pred], axis=1)

    # Apply transformations
    df_combined = add_loc(df_combined)
    df_combined = add_around_two(df_combined)
    df_combined = reverse(df_combined)
    df_combined = cleaning_part1(df_combined)
    df_combined = reverse(df_combined)
    df_combined = add_around_one(df_combined)
    df_combined = reverse(df_combined)
    df_combined = add_around_one_rev(df_combined)
    df_combined = reverse(df_combined)
    df_combined = apply_prediction_only_cprob(df_combined, window=CPROB_WINDOW)
    df_combined = box_count(df_combined)
    df_combined = switch_columns(df_combined)

    generate_csv(df_combined, model_path, save_path)
    df_combined.to_csv(temp_csv_path, index=False)

    return df_combined


def add_loc(df: pd.DataFrame) -> pd.DataFrame:
    """Add location identifiers derived from image names.

    Args:
        df: DataFrame containing the ``names`` column with image filenames.

    Returns:
        DataFrame with an added ``loc`` column.

    Author:
        Jakub Vašák

    """
    df["loc"] = (
        df["names"]
        .str.split("_")
        .apply(lambda shards: f"{shards[0]}-{shards[-1].replace('.png', '')}")
    )
    return df


def add_around_two(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``prob+-2`` column based on adjacent onset rules.

    Args:
        df: DataFrame containing ``prob``, ``annot``, and ``loc`` columns.

    Returns:
        DataFrame with the ``prob+-2`` column populated.

    Author:
        Jakub Vašák

    """
    df["prob+-2"] = float("nan")
    count = 0
    half = None

    for i in range(len(df)):
        if count <= 0:
            half = df.loc[i, "prob"]

            if i > 0:
                prev_annot = df.loc[i - 1, "annot"]
                prev_name = df.loc[i - 1, "loc"]
                curr_annot = df.loc[i, "annot"]
                curr_name = df.loc[i, "loc"]

                if curr_annot != prev_annot and curr_name == prev_name:
                    half = df.loc[i, "annot"]
                    count = 2

            df.loc[i, "prob+-2"] = half

        else:
            half = df.loc[i, "annot"]
            df.loc[i, "prob+-2"] = half
            count -= 1

    df["prob+-2"] = df["prob+-2"].astype(int)
    return df


def reverse(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse the order of DataFrame rows.

    Args:
        df: DataFrame to reverse.

    Returns:
        DataFrame with reversed row order and reset index.

    Author:
        Jakub Vašák

    """
    return df.iloc[::-1].reset_index(drop=True)


def compute_cprob(
    predictions: np.ndarray | pd.Series, window: int = CPROB_WINDOW
) -> np.ndarray:
    """Smooth a binary prediction sequence without using annotations.

    Args:
        predictions: Binary prediction sequence.
        window: Window length for enforcing stable zeros and ones.

    Returns:
        Smoothed prediction sequence.

    Author:
        Jakub Vašák

    """
    cprob = np.array(predictions, dtype=int)
    prediction_count = len(cprob)
    if prediction_count < window:
        return cprob

    for i in range(prediction_count - window, -1, -1):
        if all(cprob[i + k] == 0 for k in range(window)):
            cprob[: i + window] = 0
            break

    for i in range(prediction_count - window + 1):
        if all(cprob[i + k] == 1 for k in range(window)):
            cprob[i:] = 1
            break

    return cprob


def compute_cprob_tolerance(
    cprob: np.ndarray | pd.Series,
    annotations: np.ndarray | pd.Series,
    tolerance_steps: int,
) -> np.ndarray:
    """Apply onset-tolerance rule to predictions using annotations.

    Args:
        cprob: Cleaned prediction sequence.
        annotations: Ground-truth annotation sequence.
        tolerance_steps: Allowed offset between prediction and annotation onset.

    Returns:
        Prediction sequence adjusted for the onset tolerance window.

    Author:
        Jakub Vašák

    """
    cprob_values = np.array(cprob, dtype=int)
    annotation_values = np.array(annotations, dtype=int)

    annotation_ones = np.where(annotation_values == 1)[0]
    if len(annotation_ones) == 0:
        return cprob_values.copy()

    prediction_ones = np.where(cprob_values == 1)[0]
    if len(prediction_ones) == 0:
        return cprob_values.copy()

    onset_distance = abs(int(prediction_ones[0]) - int(annotation_ones[0]))
    if 0 < onset_distance <= tolerance_steps:
        return annotation_values.copy()
    return cprob_values.copy()


def compute_cprob_tol(
    cprob: np.ndarray | pd.Series, annotations: np.ndarray | pd.Series
) -> np.ndarray:
    """Compute ``cprob+-1`` using the canonical tolerance rule.

    Args:
        cprob: Cleaned prediction sequence.
        annotations: Ground-truth annotation sequence.

    Returns:
        Prediction sequence adjusted with a one-step tolerance.

    Author:
        Jakub Vašák

    """
    return compute_cprob_tolerance(cprob, annotations, tolerance_steps=1)


def classify_box(
    predictions: np.ndarray | pd.Series,
    annotations: np.ndarray | pd.Series,
) -> str:
    """Classify one location by first-emergence prediction semantics.

    Args:
        predictions: Prediction sequence for a single location.
        annotations: Ground-truth annotation sequence for the location.

    Returns:
        Confusion category for the location (``TP``, ``TN``, ``FP``, or ``FN``).

    Author:
        Jakub Vašák

    """
    prediction_values = np.array(predictions, dtype=int)
    annotation_values = np.array(annotations, dtype=int)
    has_annotation = np.any(annotation_values == 1)
    has_prediction = np.any(prediction_values == 1)

    if not has_annotation:
        return "TN" if not has_prediction else "FP"
    if not has_prediction:
        return "FN"
    if int(np.where(prediction_values == 1)[0][0]) == int(
        np.where(annotation_values == 1)[0][0]
    ):
        return "TP"
    return "FP"


def apply_prediction_only_cprob(
    df: pd.DataFrame, window: int = CPROB_WINDOW
) -> pd.DataFrame:
    """Add cleaned prediction columns per location while preserving row order.

    Args:
        df: DataFrame with ``prob``, ``annot``, and ``loc`` columns.
        window: Window length for ``cprob`` smoothing.

    Returns:
        DataFrame with ``cprob``, ``cprob+-1``, and ``cprob+-2`` columns.

    Author:
        Jakub Vašák

    """
    result_df = df.copy()
    result_df["cprob"] = result_df["prob"].astype(int)
    result_df["cprob+-1"] = result_df["cprob"]
    result_df["cprob+-2"] = result_df["cprob"]

    for _, location_df in result_df.groupby("loc", sort=False):
        if location_df.empty:
            continue

        location_indices = location_df.index
        cprob_values = compute_cprob(location_df["prob"], window=window)
        annotation_values = location_df["annot"]

        result_df.loc[location_indices, "cprob"] = cprob_values
        result_df.loc[location_indices, "cprob+-1"] = compute_cprob_tol(
            cprob_values, annotation_values
        )
        result_df.loc[location_indices, "cprob+-2"] = compute_cprob_tolerance(
            cprob_values, annotation_values, tolerance_steps=2
        )

    for column_name in ["cprob", "cprob+-1", "cprob+-2"]:
        result_df[column_name] = result_df[column_name].astype(int)

    return result_df


def cleaning_part1(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the ``prob+-2`` column based on annotation and location patterns.

    Args:
        df: DataFrame containing ``annot``, ``loc``, and ``prob`` columns.

    Returns:
        DataFrame with cleaned ``prob+-2`` values.

    Author:
        Jakub Vašák

    """
    check_annot = 1
    check_name = None
    count = 1
    half = 1

    for i in range(len(df)):
        if count <= 0:
            half = df.loc[i, "prob"]
            current_annot = df.loc[i, "annot"]
            current_name = df.loc[i, "loc"]

            if current_annot != check_annot and current_name == check_name:
                half = df.loc[i, "annot"]
                count = 1
                df.loc[i, "prob+-2"] = half

            check_name = df.loc[i, "loc"]
            check_annot = df.loc[i, "annot"]

        else:
            half = df.loc[i, "annot"]
            df.loc[i, "prob+-2"] = half
            count -= 1

    return df


def add_around_one(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``prob+-1`` column based on previous-condition rules.

    Args:
        df: DataFrame containing ``annot``, ``loc``, and ``prob`` columns.

    Returns:
        DataFrame with ``prob+-1`` values populated.

    Author:
        Jakub Vašák

    """
    df["prob+-1"] = None
    prev_annot = 0
    prev_name = None
    count = 0

    for idx in df.index:
        if count <= 0:
            value = str(df.loc[idx, "prob"])
            curr_annot = df.loc[idx, "annot"]
            curr_name = df.loc[idx, "loc"]

            if curr_annot != prev_annot and curr_name == prev_name:
                value = str(df.loc[idx, "annot"])

            prev_name = curr_name
            prev_annot = curr_annot

            df.loc[idx, "prob+-1"] = value
        else:
            df.loc[idx, "prob+-1"] = str(df.loc[idx, "annot"])
            count -= 1

    return df


def add_around_one_rev(df: pd.DataFrame) -> pd.DataFrame:
    """Add reversed ``prob+-1`` values after the forward pass.

    Args:
        df: DataFrame containing ``annot``, ``loc``, and ``prob`` columns.

    Returns:
        DataFrame with reversed ``prob+-1`` values.

    Author:
        Jakub Vašák

    """
    check_annot = 1
    check_name = None
    half = 1
    count = 1

    for i in range(len(df)):
        if count <= 1:
            half = df.loc[i, "prob"]
            current_annot = df.loc[i, "annot"]
            current_name = df.loc[i, "loc"]

            if current_annot != check_annot and current_name == check_name:
                half = df.loc[i, "annot"]
                count = 1
                df.loc[i, "prob+-1"] = half

            check_name = current_name
            check_annot = current_annot

        else:
            half = df.loc[i, "annot"]
            df.loc[i, "prob+-1"] = half
            count -= 1

    return df


def box_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count correct and incorrect box-level classifications per prediction type.

    Args:
        df: DataFrame with columns ``names``, ``annot``, ``prob``, ``loc``,
            ``prob+-2``, ``cprob``, ``prob+-1``, ``cprob+-2``, and ``cprob+-1``.

    Returns:
        DataFrame with an added ``box`` column after printing the summary table.

    Author:
        Jakub Vašák

    """
    df["box"] = "x"
    box_rights = [0, 0, 0, 0]
    box_wrongs = [0, 0, 0, 0]
    prediction_columns = ["prob", "prob+-2", "cprob", "cprob+-2"]

    for _, location_df in df.groupby("loc", sort=False):
        annotations = location_df["annot"]
        for i, column_name in enumerate(prediction_columns):
            box_class = classify_box(location_df[column_name], annotations)
            if box_class in {"TP", "TN"}:
                box_rights[i] += 1
            else:
                box_wrongs[i] += 1

    # Create summary with exact same structure as original
    data = {
        "Boxes ground:": [
            box_rights[0],
            box_wrongs[0],
            box_rights[0] / (box_rights[0] + box_wrongs[0]),
        ],
        "Boxes +-2": [
            box_rights[1],
            box_wrongs[1],
            box_rights[1] / (box_rights[1] + box_wrongs[1]),
        ],
        "Boxes after clean": [
            box_rights[2],
            box_wrongs[2],
            box_rights[2] / (box_rights[2] + box_wrongs[2]),
        ],
        "Boxes after clean +-2": [
            box_rights[3],
            box_wrongs[3],
            box_rights[3] / (box_rights[3] + box_wrongs[3]),
        ],
        # "Boxes after clean True +-2": [
        #     box_rights[4],
        #     box_wrongs[4],
        #     box_rights[4] / (box_rights[4] + box_wrongs[4]),
        # ],
    }

    summary_df = pd.DataFrame(data, index=["True", "False", "%"]).T
    print(summary_df)

    return df


def switch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder evaluation columns to the desired output order.

    Args:
        df: DataFrame with prediction and annotation columns.

    Returns:
        DataFrame with columns reordered for export.

    Author:
        Jakub Vašák

    """
    # Změňte pořadí sloupců
    desired_columns_order = [
        "names",
        "loc",
        "annot",
        "prob",
        "prob+-1",
        "prob+-2",
        "cprob",
        "cprob+-1",
        "cprob+-2",
    ]  # Zde uveďte požadované pořadí sloupců
    return df.reindex(columns=desired_columns_order)


def generate_csv(df: pd.DataFrame, model_path: Path, save_path: Path) -> None:
    """Drop the location column and save evaluation results to CSV.

    Args:
        df: DataFrame containing the ``loc`` column to remove.
        model_path: Path to the model file used for naming the output CSV.
        save_path: Directory where the evaluation CSV is written.

    Author:
        Jakub Vašák

    """
    # Extract model name from path
    model_name = model_path.name

    # Remove location column
    df = df.drop(columns="loc")

    # Save to CSV
    save_path.mkdir(parents=True, exist_ok=True)
    dest_name = save_path / f"Evaluation_results_{model_name}.csv"
    df.to_csv(dest_name, index=False)


def locs(input_file_path: Path) -> list[str]:
    """Return unique location identifiers from an evaluation CSV.

    Args:
        input_file_path: Path to the input CSV file.

    Returns:
        Unique values from the ``loc`` column.

    Author:
        Jakub Vašák

    """
    df = pd.read_csv(input_file_path)

    unique_strings = df["loc"].unique()
    return unique_strings.tolist()


def conf_matrix_calc(file_path: Path, loca: str, row: str) -> str:
    """Calculate confusion-matrix outcome for one location and prediction column.

    Args:
        file_path: Path to the CSV file containing annotations and predictions.
        loca: Location identifier to filter rows by.
        row: Prediction column name (``cprob``, ``cprob+-1``, or ``cprob+-2``).

    Returns:
        Confusion matrix category (``TP``, ``TN``, ``FP``, or ``FN``).

    Author:
        Jakub Vašák

    """
    # Read CSV into DataFrame
    df = pd.read_csv(file_path)

    # Filter rows for specific location
    filtered_df = df[df["loc"] == loca]

    return classify_box(filtered_df[row], filtered_df["annot"])


def conf_matrix(input_file_path: Path, row: str) -> tuple[int, int, int, int]:
    """Calculate confusion matrix counts for the selected prediction column.

    Args:
        input_file_path: Path to CSV file containing predictions and annotations.
        row: Prediction column name (``cprob``, ``cprob+-1``, or ``cprob+-2``).

    Returns:
        Confusion matrix values in order ``(TP, TN, FP, FN)``.

    Author:
        Jakub Vašák

    """
    # Initialize confusion matrix counters from the CSV only.
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Get list of unique locations
    box_strings = locs(input_file_path)

    # Calculate confusion matrix for each location
    for box in box_strings:
        result = conf_matrix_calc(file_path=input_file_path, loca=box, row=row)

        # Update appropriate counter based on result
        if result == "TP":
            true_positives += 1
        elif result == "TN":
            true_negatives += 1
        elif result == "FP":
            false_positives += 1
        elif result == "FN":
            false_negatives += 1

    return true_positives, true_negatives, false_positives, false_negatives


def box_confusion(
    input_file_path: Path,
    model_path: Path,
    save_path: Path,
) -> pd.DataFrame:
    """Compute box-level confusion metrics and save them to Excel.

    Args:
        input_file_path: Path to the input CSV file containing predictions.
        model_path: Path to the model file used for Excel output naming.
        save_path: Directory where the Excel results are saved.

    Returns:
        DataFrame containing confusion matrix results with columns
        ``[BOX_CONFUSION_MATRIX, TP, TN, FP, FN, acc]``.

    Author:
        Jakub Vašák

    """
    # Calculate confusion matrix for each prediction type
    TP1, TN1, FP1, FN1 = conf_matrix(input_file_path, "cprob")
    acc1 = (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1)

    TP2, TN2, FP2, FN2 = conf_matrix(input_file_path, "cprob+-1")
    acc2 = (TP2 + TN2) / (TP2 + TN2 + FP2 + FN2)

    TP3, TN3, FP3, FN3 = conf_matrix(input_file_path, "cprob+-2")
    acc3 = (TP3 + TN3) / (TP3 + TN3 + FP3 + FN3)

    # Create results DataFrame
    data = {
        "BOX_CONFUSION_MATRIX": ["cprob", "cprob+-1", "cprob+-2"],
        "TP": [TP1, TP2, TP3],
        "TN": [TN1, TN2, TN3],
        "FP": [FP1, FP2, FP3],
        "FN": [FN1, FN2, FN3],
        "acc": [acc1, acc2, acc3],
    }

    result_df = pd.DataFrame(data)

    # Save results to Excel
    save_into_excel(
        model_path=model_path,
        sheet_name="BOX",
        df=result_df,
        save_path=save_path,
    )

    return result_df


def save_into_excel(
    model_path: Path,
    save_path: Path,
    sheet_name: str,
    df: pd.DataFrame,
) -> None:
    """Save a DataFrame into an evaluation Excel workbook.

    Args:
        model_path: Path to model file used to build the Excel filename.
        save_path: Directory where the Excel file is saved.
        sheet_name: Name of the worksheet to write.
        df: DataFrame containing data to save.

    Author:
        Jakub Vašák

    """
    # Suppress UserWarnings that can occur during Excel operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Extract model name and construct output path
        save_path.mkdir(parents=True, exist_ok=True)
        model_name = model_path.name
        excel_path = save_path / f"Evaluation_summary_{model_name}.xlsx"

        try:
            if excel_path.is_file():
                with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a") as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df.to_excel(excel_path, sheet_name=sheet_name, index=False)

        except Exception as exc:
            LOGGER.warning(
                "Could not save evaluation sheet %s to %s: %s",
                sheet_name,
                excel_path,
                exc,
            )


def evaluation_numb(
    true_positives: float,
    true_negatives: float,
    false_positives: float,
    false_negatives: float,
    correct_predictions: float,
    incorrect_predictions: float,
    task_name: str,
) -> pd.DataFrame:
    """Create a formatted DataFrame with evaluation metrics.

    Args:
        true_positives: Number of true positives.
        true_negatives: Number of true negatives.
        false_positives: Number of false positives.
        false_negatives: Number of false negatives.
        correct_predictions: Total number of correct predictions.
        incorrect_predictions: Total number of incorrect predictions.
        task_name: Name of the evaluation task.

    Returns:
        Transposed DataFrame containing metrics and accuracy.

    Author:
        Jakub Vašák

    """
    # Create data dictionary with metrics
    data = {
        "Value": [
            true_positives,
            true_negatives,
            false_positives,
            false_negatives,
            correct_predictions,
            incorrect_predictions,
            correct_predictions / (correct_predictions + incorrect_predictions),
        ]
    }

    # Create DataFrame with metrics as index
    df = pd.DataFrame(
        data, index=["TP", "TN", "FP", "FN", "True", "False", "Accuracy"]
    ).T  # Transpose DataFrame for better readability

    # Print task name and metrics DataFrame
    print(f"\n{task_name.upper()} EVALUATION:\n{df}\n{'-' * 50}")

    return df


def eva(
    df: pd.DataFrame,
    model_path: Path,
    task_name: str,
    task: str,
    save_path: Path,
    output_file_path: Path,
) -> None:
    """Calculate confusion-matrix metrics and save them to Excel.

    Args:
        df: DataFrame containing annotations and predictions.
        model_path: Path to model used for Excel output naming.
        task_name: Name of the evaluation task for the Excel sheet.
        task: Prediction column name to evaluate.
        save_path: Directory where Excel results will be saved.
        output_file_path: Path where the evaluated CSV is written.

    Author:
        Jakub Vašák

    """
    # Initialize confusion matrix counters
    true_predictions = false_predictions = 0
    true_positives = true_negatives = false_positives = false_negatives = 0

    # Get truth and prediction columns
    annotations = df["annot"].astype(int)
    predictions = df[task].astype(int)

    # Calculate metrics
    for truth, prediction in zip(annotations, predictions):  # noqa: B905
        # Count correct/incorrect predictions
        if truth == prediction:
            true_predictions += 1
        else:
            false_predictions += 1

        # Update confusion matrix
        if truth == 1 and prediction == 1:
            true_positives += 1
        elif truth == 0 and prediction == 0:
            true_negatives += 1
        elif truth == 0 and prediction == 1:
            false_positives += 1
        elif truth == 1 and prediction == 0:
            false_negatives += 1

    # Save original data
    df.to_csv(output_file_path, index=False)

    # Generate evaluation summary
    results_df = evaluation_numb(
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
        true_predictions,
        false_predictions,
        task_name,
    )

    # Save results to Excel
    save_into_excel(
        model_path=model_path,
        sheet_name=task_name,
        df=results_df,
        save_path=save_path,
    )


def eva_calc(
    df: pd.DataFrame,
    model_path: Path,
    save_path: Path,
    input_file_path: Path,
    output_file_path: Path,
) -> pd.DataFrame:
    """Calculate evaluation metrics for multiple prediction types.

    Args:
        df: DataFrame containing predictions and annotations.
        model_path: Path to the model file.
        save_path: Directory where results are saved.
        input_file_path: Path to the CSV used for box-level metrics.
        output_file_path: Path to the evaluated CSV output.

    Returns:
        Box confusion matrix results with columns
        ``[BOX_CONFUSION_MATRIX, TP, TN, FP, FN, acc]``.

    Author:
        Jakub Vašák

    """
    # Define evaluation tasks
    evaluation_tasks = [
        ("ground truth", "prob"),
        ("accuracy +-1", "prob+-1"),
        ("accuracy +-2", "prob+-2"),
        ("accuracy after clean", "cprob"),
        ("accuracy after clean +-1", "cprob+-1"),
        ("accuracy after clean +-2", "cprob+-2"),
    ]

    # Run evaluations
    for task_name, column in evaluation_tasks:
        eva(
            df=df,
            model_path=model_path,
            task_name=task_name,
            task=column,
            save_path=save_path,
            output_file_path=output_file_path,
        )

    # Generate and print box confusion matrix
    result_df = box_confusion(input_file_path, model_path, save_path)
    print(f"\n{result_df}")

    return result_df


def evaluation(
    data_path: Path,
    model_path: Path,
    names_path: Path,
    annotations_path: Path,
    save_path: Path,
    temp_path: Path = TEMP_PATH,
    input_file_path: Path = INPUT_FILE_PATH,
    output_file_path: Path = OUTPUT_FILE_PATH,
) -> None:
    """Run the full evaluation pipeline on the configured test data.

    Args:
        data_path: Path to the NPY file containing test images.
        model_path: Path to the saved model file to evaluate.
        names_path: Path to the NPY file containing image names.
        annotations_path: Path to the NPY file containing ground-truth labels.
        save_path: Directory where evaluation outputs are written.
        temp_path: Temporary workspace directory.
        input_file_path: Path to the temporary CSV for box-level metrics.
        output_file_path: Path to the evaluated CSV output.

    Author:
        Jakub Vašák

    """
    # Setup clean workspace
    delete_and_create(temp_path)

    # Import and process test data
    df = import_npy(
        data_path=data_path,
        model_path=model_path,
        names_path=names_path,
        annotations_path=annotations_path,
        save_path=save_path,
        temp_csv_path=input_file_path,
    )

    # Calculate evaluation metrics
    eva_calc(df, model_path, save_path, input_file_path, output_file_path)
    # eva_calc(input_file_path, model_path=model_path)


def set_paths() -> tuple[Path, Path, Path]:
    """Return configured paths for models, NumPy datasets, and outputs.

    Returns:
        Tuple of ``(model_dir, numpy_dataset_dir, output_dir)`` paths.

    Author:
        Jakub Vašák

    """
    return (
        EVALUATION_PATHS.model_dir,
        EVALUATION_PATHS.numpy_dataset_dir,
        EVALUATION_PATHS.output_dir,
    )


def format_model_name(paths: EvaluationPaths) -> str:
    """Create the configured model name for the selected evaluation parameters.

    Args:
        paths: Evaluation path settings and naming parameters.

    Returns:
        Formatted model filename for the current configuration.

    Author:
        Jakub Vašák

    """
    return paths.model_name_template.format(
        project_name=paths.project_name,
        image_size=paths.image_size,
        time_step=paths.time_step,
        data_range=paths.data_range,
        dataset_split=paths.dataset_split,
    )


def format_dataset_prefix(paths: EvaluationPaths) -> str:
    """Create the NumPy dataset prefix generated by stage 3 preprocessing.

    Args:
        paths: Evaluation path settings and naming parameters.

    Returns:
        Dataset filename prefix for the configured split.

    Author:
        Jakub Vašák

    """
    return (
        f"data_{paths.project_name}_{paths.image_size}_{paths.time_step}_"
        f"{paths.data_range}_{paths.dataset_split}"
    )


def main() -> None:
    """Run evaluation with the configured model and dataset paths.

    Author:
        Jakub Vašák
    """
    model_save_path, numpy_datasets, save_path = set_paths()
    paths = EVALUATION_PATHS
    dataset_prefix = format_dataset_prefix(paths)

    # Example evaluation call
    evaluation(
        data_path=numpy_datasets / f"{dataset_prefix}.npy",
        model_path=model_save_path / format_model_name(paths),
        names_path=numpy_datasets / f"{dataset_prefix}_img_names.npy",
        annotations_path=numpy_datasets / f"{dataset_prefix}_ann.npy",
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
