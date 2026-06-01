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
ROOT_PATH = Path("/home/vasakjakub/fenotypizace")
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
    """Delete and recreate the temporary evaluation workspace."""
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
    """
    Load NPY files, use model for predictions, save results to CSV.

    Args:
        data_path: Path to data NPY file
        names_path: Path to names NPY file
        model_path: Path to model file
        annotations_path: Path to annotations NPY file
        save_path: Path to save CSV file

    Returns:
        DataFrame with combined predictions
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
    """Add location column based on image names."""
    df["loc"] = (
        df["names"]
        .str.split("_")
        .apply(lambda shards: f"{shards[0]}-{shards[-1].replace('.png', '')}")
    )
    return df


def add_around_two(df: pd.DataFrame) -> pd.DataFrame:
    """Add prob+-2 column based on conditions."""
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
    """Reverse order of DataFrame rows."""
    return df.iloc[::-1].reset_index(drop=True)


def compute_cprob(
    predictions: np.ndarray | pd.Series, window: int = CPROB_WINDOW
) -> np.ndarray:
    """Smooth a binary prediction sequence without using annotations."""
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
    """Return annotation sequence only for explicit onset-tolerance evaluation."""
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
    """Compute cprob+-1 using the canonical analyze.py tolerance rule."""
    return compute_cprob_tolerance(cprob, annotations, tolerance_steps=1)


def classify_box(
    predictions: np.ndarray | pd.Series,
    annotations: np.ndarray | pd.Series,
) -> str:
    """Classify one location by first-emergence prediction semantics."""
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
    """Add cprob columns per location while preserving row order."""
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
    """Clean data based on annotation and location patterns."""
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
    """Add prob+-1 column based on previous conditions."""
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
    """Add reversed prob+-1 values."""
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
    """
    Counts correct and incorrect classifications for each location box in the dataset.

    Processing steps:
    1. Initializes counters for each prediction type
    2. Groups rows by location
    3. Classifies each location by first-emergence semantics
    4. Generates summary statistics for each prediction type

    Comparisons (in order):
    1. Ground truth (prob vs annot)
    2. Prob with ±2 tolerance
    3. Clean predictions (cprob)
    4. Clean predictions with ±2 tolerance

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns in order:
        names,annot,prob,loc,prob+-2,cprob,prob+-1,cprob+-2,cprob+-1

    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added 'box' column and printed accuracy summary
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
    """
    This function reorders the Dataframe columns according to the desired order.

    Args:
        df: DataFrame with columns [names,annot,prob,loc,prob+-2,cprob,prob+-1,cprob+-2,cprob+-1,box]

    Returns:
        DataFrame with added and reordered columns
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
    """
    Removes location column from DataFrame and saves it as CSV.

    Processing steps:
    1. Takes input DataFrame
    2. Removes 'loc' column
    3. Extracts model name from path
    4. Saves modified DataFrame to CSV

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the 'loc' column to be removed
    model_path : str
        Path to model, used to extract model name for output file
    model_save_path : str
        Directory where the output CSV will be saved

    Returns:
    --------
    None
        Saves modified DataFrame to CSV file named '{model_name}_test.csv'
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
    """
    This function reads a CSV file from the given input file path, extracts the unique
    values from the 'loc' column, and returns these unique values as a list.

    Parameters:
        input_file_path (str): The file path of the input CSV file

    Returns:
        list: A list containing the unique strings from the 'loc' column of the CSV file
    """

    df = pd.read_csv(input_file_path)

    unique_strings = df["loc"].unique()
    return unique_strings.tolist()


def conf_matrix_calc(file_path: Path, loca: str, row: str) -> str:
    """
    Calculate confusion matrix outcome for a specific location and prediction column.

    Processing steps:
    1. Read CSV file into DataFrame
    2. Filter rows for specific location
    3. Classify first-emergence match between annotations and predictions

    Logic for outcomes:
    - TP: germinating location with matching first positive step
    - TN: non-germinating location with no positive predictions
    - FP: positive prediction for NG or wrong first positive step
    - FN: germinating location with no positive prediction

    Parameters
    ----------
    file_path : str
        Path to CSV file containing annotations and predictions
    loca : str
        Location identifier to filter rows by
    row : str
        Name of prediction column to compare with annotations
        Supported values: ['cprob', 'cprob+-1', 'cprob+-2']

    Returns
    -------
    str
        Confusion matrix category:
        - 'TP': True Positive
        - 'TN': True Negative
        - 'FP': False Positive
        - 'FN': False Negative
    """
    # Read CSV into DataFrame
    df = pd.read_csv(file_path)

    # Filter rows for specific location
    filtered_df = df[df["loc"] == loca]

    return classify_box(filtered_df[row], filtered_df["annot"])


def conf_matrix(input_file_path: Path, row: str) -> tuple[int, int, int, int]:
    """
    Calculate confusion matrix values for predictions in specific column.

    Processing steps:
    1. Get list of unique locations from file
    2. Initialize confusion matrix counters
    3. For each location:
       - Calculate confusion matrix outcome
       - Update appropriate counter
    4. Return final counts

    Metrics calculated:
    - True Positives (TP): Correctly predicted positive cases
    - True Negatives (TN): Correctly predicted negative cases
    - False Positives (FP): Incorrectly predicted positive cases
    - False Negatives (FN): Incorrectly predicted negative cases

    Parameters
    ----------
    input_file_path : str
        Path to CSV file containing predictions and annotations
    row : str
        Name of column to compare with annotations column
        Supported values: ['cprob', 'cprob+-1', 'cprob+-2']

    Returns
    -------
    tuple[int, int, int, int]
        Confusion matrix values in order: (TP, TN, FP, FN)
        - TP: Number of true positives
        - TN: Number of true negatives
        - FP: Number of false positives
        - FN: Number of false negatives
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
    """
    Calculate confusion matrix metrics and accuracy for different prediction types.

    Processing steps:
    1. Calculate confusion matrix values for base predictions (cprob)
    2. Calculate confusion matrix values for ±1 tolerance (cprob+-1)
    3. Calculate confusion matrix values for ±2 tolerance (cprob+-2)
    4. Combine results into DataFrame
    5. Save results to Excel file

    Tasks evaluated:
    - Base predictions (cprob)
    - Predictions with ±1 frame tolerance (cprob+-1)
    - Predictions with ±2 frame tolerance (cprob+-2)

    Parameters
    ----------
    input_file_path : str
        Path to input CSV file containing predictions
    model_path : str
        Path to model file, used for Excel output naming
    save_path : str
        Directory where Excel results will be saved

    Returns
    -------
    pd.DataFrame
        DataFrame containing confusion matrix results with columns:
        [BOX_CONFUSION_MATRIX, TP, TN, FP, FN, acc]
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
    """
    Save DataFrame into Excel file, either creating new file or appending to existing one.

    Parameters
    ----------
    model_path : str
        Path to model file, used to extract model name for output file
    save_path : str
        Directory where Excel file will be saved
    sheet_name : str
        Name of sheet where DataFrame will be saved
    df : pd.DataFrame
        DataFrame containing data to be saved to Excel

    Returns
    -------
    None
        Saves DataFrame to Excel file but does not return value
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
    """
    Create and format a DataFrame containing evaluation metrics from confusion matrix values.

    Processing steps:
    1. Takes confusion matrix values and overall counts
    2. Creates dictionary with metrics
    3. Builds DataFrame with metrics as index
    4. Transposes DataFrame for better readability
    5. Prints task name and formatted metrics

    Parameters:
    -----------
    TP : float
        Number of true positives
    TN : float
        Number of true negatives
    FP : float
        Number of false positives
    FN : float
        Number of false negatives
    TRUE : float
        Total number of correct predictions
    FALSE : float
        Total number of incorrect predictions
    task_name : str
        Name of the evaluation task

    Returns:
    --------
    pd.DataFrame
        Transposed DataFrame containing evaluation metrics and accuracy
        Columns: [TP, TN, FP, FN, True, False, Accuracy]
        Single row with actual values
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
    """
    Calculates confusion matrix metrics and accuracy for predictions.

    Processing steps:
    1. Extracts truth values and predictions
    2. Calculates true/false predictions
    3. Computes confusion matrix values (TP, TN, FP, FN)
    4. Generates evaluation metrics
    5. Saves results to Excel

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing annotations and predictions
    model_path : str
        Path to model, used for Excel output
    task_name : str
        Name of evaluation task for Excel sheet
    task : str
        Column name containing predictions to evaluate
    save_path : str
        Directory where Excel results will be saved

    Returns:
    --------
    None
        Saves evaluation metrics to Excel file
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
    """
    Calculate and save multiple evaluation metrics for different prediction types.

    Processing steps:
    1. Evaluates ground truth predictions
    2. Evaluates predictions with ±1 tolerance
    3. Evaluates predictions with ±2 tolerance
    4. Evaluates cleaned predictions
    5. Evaluates cleaned predictions with ±1 tolerance
    6. Evaluates cleaned predictions with ±2 tolerance
    7. Generates box confusion matrix

    Evaluation tasks:
    - ground truth: raw model predictions
    - accuracy ±1: predictions with one frame tolerance
    - accuracy ±2: predictions with two frame tolerance
    - accuracy after clean: cleaned predictions
    - accuracy after clean ±1: cleaned predictions with one frame tolerance
    - accuracy after clean ±2: cleaned predictions with two frame tolerance

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing predictions and annotations
        Required columns: [names, annot, prob, prob+-1, prob+-2, cprob, cprob+-1, cprob+-2]
    model_path : str
        Path to model file
    save_path : str
        Directory where results will be saved

    Returns:
    --------
    pd.DataFrame
        Box confusion matrix results DataFrame with columns:
        [BOX_CONFUSION_MATRIX, TP, TN, FP, FN, acc]
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
    """
    Run complete evaluation pipeline for a model on test data.

    Processing steps:
    1. Create clean workspace by deleting and recreating temp directory
    2. Import and process test data:
       - Load test images
       - Load annotations
       - Load image names
       - Make model predictions
    3. Calculate evaluation metrics:
       - Confusion matrix values
       - Accuracy scores
       - Box-level metrics
    4. Save results to files

    Parameters
    ----------
    data_path : str
        Path to NPY file containing test images
    model_path : str
        Path to saved model file to evaluate
    names_path : str
        Path to NPY file containing image names
    annotations_path : str
        Path to NPY file containing ground truth annotations

    Returns
    -------
    None
        Results are saved to files:
        - CSV files with predictions
        - Excel files with evaluation metrics

    Error handling
    -------------
    - Checks if all input files exist
    - Validates input data formats
    - Creates output directories if missing
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
    """
    Set global paths for data, models, and results.

    Returns:
    --------
        tuple[str, str, str]
            Tuple containing paths for data, models, numpy datasets, and save locations
    """
    return (
        EVALUATION_PATHS.model_dir,
        EVALUATION_PATHS.numpy_dataset_dir,
        EVALUATION_PATHS.output_dir,
    )


def format_model_name(paths: EvaluationPaths) -> str:
    """Create the configured model name for the selected evaluation parameters."""
    return paths.model_name_template.format(
        project_name=paths.project_name,
        image_size=paths.image_size,
        time_step=paths.time_step,
        data_range=paths.data_range,
        dataset_split=paths.dataset_split,
    )


def format_dataset_prefix(paths: EvaluationPaths) -> str:
    """Create the NumPy dataset prefix generated by stage 3 preprocessing."""
    return (
        f"data_{paths.project_name}_{paths.image_size}_{paths.time_step}_"
        f"{paths.data_range}_{paths.dataset_split}"
    )


def main() -> None:
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
