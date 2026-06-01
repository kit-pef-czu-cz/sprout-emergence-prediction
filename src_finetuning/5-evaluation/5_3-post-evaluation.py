"""Generate emergence timing graphs from finetuning evaluation results.

The script reads a semicolon-separated ``Results.csv`` file, fits sigmoid curves
for each tray, exports per-tray plots, writes an all-trays comparison plot, and
saves the fitted EC50/slope/germination parameters.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.cm as cm  # noqa: PLR0402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import seaborn as sns
from matplotlib.ticker import MaxNLocator

LOGGER = logging.getLogger(__name__)


# Advanced users: edit these paths and parameters directly in this script.
ROOT_PATH = Path("sprout-emergence-prediction")
PROJECT_NAME = "nabila"
EVALUATION_ROOT_DIR = ROOT_PATH / "data" / "evaluation" / PROJECT_NAME

TIME_STEP = 3
DATA_RANGE = 9
INPUT_CSV = (
    EVALUATION_ROOT_DIR
    / f"Evaluation_results_ef2_tcn_{TIME_STEP}-{DATA_RANGE}_org_nab_relearned.csv"
)
OUTPUT_DIR = EVALUATION_ROOT_DIR / "emergence_graphs"

CSV_SEPARATOR = ";"
CSV_DECIMAL = "."
GENERATED_RESULTS_SEPARATOR = ","
GENERATED_RESULTS_NAME_COLUMN = "names"
GENERATED_RESULTS_EVENT_COLUMN = "cprob"
FILENAME_PATTERN = re.compile(r"(\d+)_(\d+-\d+-\d+-\d+-\d+-\d+)_(\d+-\d+)\.png")
FILENAME_DATETIME_FORMAT = "%d-%m-%y-%H-%M-%S"
OUTPUT_FORMAT = "png"
FIGURE_DPI = 300
SIGMOID_POINTS = 200
PREDICTOR_MAX_SLOPE = 0.5
HEATMAP_DEFAULT_COLUMNS = 10
HEATMAP_NINE_COLUMN_ROWS = {36, 72, 108}
HEATMAP_WINDOW = 20

SHOW_GRAPHS = False
SAVE_GRAPHS = True
SAVE_PARAMETERS_CSV = True
PARAMETERS_CSV = "Evaluated_data.csv"
ALL_TRAYS_FILENAME = "All_trays"


NumericArray = float | np.ndarray | pd.Series


def natural_sort_key(value: str) -> tuple[int, ...] | tuple[str]:
    """Return a numeric sort key for tray IDs and well IDs.

    Args:
        value: Identifier from the generated evaluation CSV.

    Returns:
        Numeric identifier parts when possible, otherwise the original string.

    Author:
        Jakub Vašák

    """
    try:
        return tuple(int(part) for part in value.split("-"))
    except ValueError:
        return (value,)


def sigmoid(
    x_value: NumericArray,
    log_ic50: float,
    hill_slope: float,
    maximum_response: float,
) -> NumericArray:
    """Calculate the sigmoid response curve used for emergence fitting.

    Args:
        x_value: Time point or array of time points.
        log_ic50: Time where the curve reaches half of ``maximum_response``.
        hill_slope: Slope parameter optimized in log-space.
        maximum_response: Maximum response value, expressed as emerged plants
            in percent.

    Returns:
        Sigmoid response for each input time point.

    Author:
        Jakub Vašák

    """
    return maximum_response / (1 + 10 ** ((log_ic50 - x_value) * hill_slope))


def is_generated_results_dataset(raw_dataset: pd.DataFrame) -> bool:
    """Check whether a DataFrame matches the CSV written by ``5_1-evaluation.py``.

    Args:
        raw_dataset: DataFrame loaded from the configured input CSV.

    Returns:
        ``True`` when the input has filename and prediction columns.

    Author:
        Jakub Vašák

    """
    return {
        GENERATED_RESULTS_NAME_COLUMN,
        GENERATED_RESULTS_EVENT_COLUMN,
    }.issubset(raw_dataset.columns)


def convert_generated_results(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    """Convert generated prediction rows into tray emergence-time columns.

    The CSV written by ``5_1-evaluation.py`` stores one row per tray/well/image.
    This function finds the first positive prediction for each tray/well series
    and converts its timestamp into hours from the first image timestamp in the
    file. The resulting matrix matches the legacy ``Results.csv`` structure used
    by the plotting code: one column per tray and one row per well.

    Args:
        raw_dataset: Comma-separated results from ``5_1-evaluation.py``.

    Returns:
        Matrix of first predicted emergence times in hours.

    Raises:
        ValueError: If required filename or prediction values are invalid.

    Author:
        Jakub Vašák

    """
    dataset = raw_dataset.copy()
    parsed_names = (
        dataset[GENERATED_RESULTS_NAME_COLUMN].astype(str).str.extract(FILENAME_PATTERN)
    )
    parsed_names.columns = ["tray", "timestamp_text", "well"]

    invalid_names = parsed_names.isna().any(axis=1)
    if invalid_names.any():
        examples = dataset.loc[invalid_names, GENERATED_RESULTS_NAME_COLUMN].head(3)
        msg = (
            "Could not parse tray, timestamp, and well from generated result "
            f"filename(s): {examples.tolist()}"
        )
        raise ValueError(msg)

    dataset[["tray", "timestamp_text", "well"]] = parsed_names
    dataset["timestamp"] = pd.to_datetime(
        dataset["timestamp_text"],
        format=FILENAME_DATETIME_FORMAT,
        errors="coerce",
    )
    if dataset["timestamp"].isna().any():
        msg = "Could not parse one or more timestamps from generated result names."
        raise ValueError(msg)

    event_values = pd.to_numeric(
        dataset[GENERATED_RESULTS_EVENT_COLUMN],
        errors="coerce",
    )
    if event_values.isna().any():
        msg = f"Column {GENERATED_RESULTS_EVENT_COLUMN!r} contains non-numeric values."
        raise ValueError(msg)
    dataset[GENERATED_RESULTS_EVENT_COLUMN] = event_values.astype(int)

    time_origin = dataset["timestamp"].min()
    emerged = (
        dataset.loc[dataset[GENERATED_RESULTS_EVENT_COLUMN].eq(1)]
        .groupby(["tray", "well"], as_index=False)["timestamp"]
        .min()
    )
    emerged["emergence_hours"] = (
        emerged["timestamp"] - time_origin
    ).dt.total_seconds() / 3600

    trays = sorted(dataset["tray"].unique(), key=natural_sort_key)
    wells = sorted(dataset["well"].unique(), key=natural_sort_key)
    emergence_matrix = (
        emerged.pivot(index="well", columns="tray", values="emergence_hours")  # noqa: PD010
        .reindex(index=wells, columns=trays)
        .rename(columns=lambda tray: f"Tray_{tray}")
    )

    LOGGER.info(
        "Converted generated results to %s tray column(s) and %s well row(s).",
        emergence_matrix.shape[1],
        emergence_matrix.shape[0],
    )
    return emergence_matrix


def prepare_legacy_dataset(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalize the legacy tray-matrix results table.

    Args:
        raw_dataset: Legacy results table where the first row contains tray
            names and following rows contain emergence timestamps.

    Returns:
        DataFrame with tray names as columns and emergence times as rows.

    Author:
        Jakub Vašák

    """
    dataset = raw_dataset.copy()
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop(columns="Unnamed: 0")

    column_names = [str(dataset[column].iloc[0]) for column in dataset.columns]
    dataset = dataset.drop(index=dataset.index[0]).reset_index(drop=True)
    dataset.columns = column_names
    return dataset


def prepare_dataset(raw_dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Normalize the raw results table before plotting.

    Args:
        raw_dataset: Results loaded from ``INPUT_CSV``. This can be either the
            generated long-format prediction CSV from ``5_1-evaluation.py`` or
            the legacy tray-matrix ``Results.csv`` format.

    Returns:
        A numeric dataset indexed by plant occurrence and a list with the plant
        count for each tray.

    Raises:
        ValueError: If the input dataset is empty.

    Author:
        Jakub Vašák

    """
    if raw_dataset.empty:
        msg = "The results dataset is empty."
        raise ValueError(msg)

    if is_generated_results_dataset(raw_dataset):
        dataset = convert_generated_results(raw_dataset)
    else:
        dataset = prepare_legacy_dataset(raw_dataset)

    dataset = dataset.apply(pd.to_numeric, errors="coerce", downcast="float")
    numbers_of_plants = [int(dataset[tray].count()) for tray in dataset.columns]
    if not np.isfinite(dataset.to_numpy(dtype=float, na_value=np.nan)).any():
        msg = (
            "The results dataset does not contain any numeric emergence times. "
            "Check INPUT_CSV and the configured CSV separator."
        )
        raise ValueError(msg)

    return dataset, numbers_of_plants


def fit_sigmoid(
    counts: pd.DataFrame,
    maximum_time: float,
    response_bound: float,
) -> np.ndarray:
    """Fit a sigmoid curve to cumulative emergence percentages.

    Args:
        counts: DataFrame with ``x`` time points and ``c%`` cumulative response.
        maximum_time: Maximum emergence time in the experiment.
        response_bound: Maximum cumulative emergence percentage for one tray.

    Returns:
        Optimized ``log_ic50``, ``hill_slope``, and maximum response values.

    Raises:
        ValueError: If bounds cannot be built from finite numeric values.

    Author:
        Jakub Vašák

    """
    if not np.isfinite(maximum_time) or not np.isfinite(response_bound):
        msg = (
            "Cannot fit sigmoid because maximum_time or response_bound is not "
            f"finite: maximum_time={maximum_time}, response_bound={response_bound}."
        )
        raise ValueError(msg)

    result = scipy.optimize.differential_evolution(
        lambda params: np.sum(
            (sigmoid(counts["x"], *params) - counts["c%"]) ** 2,
        ),
        [[0, maximum_time], [0, PREDICTOR_MAX_SLOPE], [0, response_bound]],
    )
    return result["x"]


def get_heatmap_columns(row_count: int) -> int:
    """Return the tray grid width expected by the heatmap layout.

    Args:
        row_count: Number of rows in the tray matrix.

    Returns:
        Heatmap grid width to use for reshaping tray values.

    Author:
        Jakub Vašák

    """
    if row_count in HEATMAP_NINE_COLUMN_ROWS:
        return 9
    return HEATMAP_DEFAULT_COLUMNS


def build_tray_counts(
    tray_values: pd.Series,
    number_of_plants: int,
) -> pd.DataFrame:
    """Build absolute and cumulative emergence percentages for a tray.

    Args:
        tray_values: Numeric emergence timestamps for one tray.
        number_of_plants: Count of valid plant observations in the tray.

    Returns:
        Counts with absolute percentage (``y%``) and cumulative percentage
        (``c%``) columns.

    Raises:
        ValueError: If the tray contains no valid emergence values.

    Author:
        Jakub Vašák

    """
    if number_of_plants == 0:
        msg = "Cannot build tray counts for a tray without emergence values."
        raise ValueError(msg)

    counts = tray_values.value_counts(sort=False)
    counts = pd.DataFrame({"x": counts.index, "y": counts.to_numpy()})
    counts = counts.sort_values("x")
    counts["y%"] = counts["y"] * 100 / number_of_plants
    counts["c"] = counts["y"].cumsum()
    counts["c%"] = counts["c"] * 100 / number_of_plants
    return counts


def plot_tray_graph(
    dataset: pd.DataFrame,
    tray: str,
    number_of_plants: int,
    minimum_time: float,
    maximum_time: float,
    output_dir: Path,
    output_format: str,
    save: bool,
    show: bool,
) -> list[float]:
    """Create the per-tray emergence curve and positional heatmap.

    Args:
        dataset: Numeric emergence times for all trays.
        tray: Name of the tray column being plotted.
        number_of_plants: Count of valid plant observations for ``tray``.
        minimum_time: Minimum emergence time across the full dataset.
        maximum_time: Maximum emergence time across the full dataset.
        output_dir: Directory where figures are saved.
        output_format: Matplotlib output format, such as ``png`` or ``jpg``.
        save: Whether to save generated figures.
        show: Whether to display figures interactively.

    Returns:
        Fitted sigmoid parameters as ``[EC50, slope, percent_germinated]``.

    Author:
        Jakub Vašák

    """
    LOGGER.info("Evaluating %s", tray)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = build_tray_counts(dataset[tray], number_of_plants)
    response_bound = float(counts["c%"].max())
    params = fit_sigmoid(counts, maximum_time, response_bound)
    LOGGER.info("Fitted parameters for %s: %s", tray, params)

    time_points = np.linspace(minimum_time, maximum_time, SIGMOID_POINTS)
    axes[0].bar(counts["x"], counts["y%"], color="mediumblue")
    axes[0].scatter(counts["x"], counts["c%"], color="b")
    axes[0].plot(
        time_points,
        sigmoid(time_points, *params),
        linewidth=2,
        linestyle="--",
        color="k",
    )
    axes[0].scatter(params[0], params[2] / 2, color="r", marker="x")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].text(
        0.02,
        0.90,
        f"EC50: {round(params[0], 1)} h     Emerged: {round(response_bound, 1)} %"
        f"\nSlope: {round(10 ** params[1], 3)}",
        transform=axes[0].transAxes,
    )
    axes[0].axhline(response_bound, 0, 1, linewidth=1, color="k", linestyle="dotted")
    axes[0].set_xlim(minimum_time - 2, maximum_time + 2)
    axes[0].set_ylim(0, 115)
    axes[0].set_title(tray)
    axes[0].set_xlabel("Emergence time [h]")
    axes[0].set_ylabel("Plants emerged [%]")

    columns = get_heatmap_columns(len(dataset))
    heatmap_data = dataset[tray].to_numpy().reshape(-1, columns)
    axes[1].set_facecolor("grey")
    sns.heatmap(
        heatmap_data,
        cmap="vlag_r",
        linewidth=0.5,
        linecolor="k",
        square=True,
        xticklabels=False,
        yticklabels=False,
        center=params[0],
        vmin=params[0] - HEATMAP_WINDOW,
        vmax=params[0] + HEATMAP_WINDOW,
        ax=axes[1],
    )

    if save:
        fig.savefig(output_dir / f"{tray}.{output_format}", dpi=FIGURE_DPI)
    if show:
        plt.show()
    plt.close(fig)

    return list(params)


def plot_all_trays(
    dataset: pd.DataFrame,
    parameters: list[list[float]],
    minimum_time: float,
    maximum_time: float,
    output_dir: Path,
    output_format: str,
    save: bool,
    show: bool,
) -> None:
    """Create a combined sigmoid-curve comparison for all trays.

    Args:
        dataset: Numeric emergence times for all trays.
        parameters: Fitted sigmoid parameters for each tray.
        minimum_time: Minimum emergence time across the full dataset.
        maximum_time: Maximum emergence time across the full dataset.
        output_dir: Directory where the figure is saved.
        output_format: Matplotlib output format, such as ``png`` or ``jpg``.
        save: Whether to save the generated figure.
        show: Whether to display the figure interactively.

    Author:
        Jakub Vašák

    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = cm.nipy_spectral(np.linspace(0, 1, dataset.shape[1]))
    sorted_parameters = sorted(parameters, key=lambda params: int(params[0]))
    sorted_indices = [
        index for index, _ in sorted(enumerate(parameters), key=lambda item: item[1])
    ]
    labels = [dataset.columns[index] for index in sorted_indices]

    x_values = np.linspace(minimum_time, maximum_time, SIGMOID_POINTS)
    response_x_values = np.linspace(0, maximum_time, SIGMOID_POINTS)

    for index, params in enumerate(sorted_parameters):
        ax.plot(
            x_values,
            sigmoid(response_x_values, *params),
            linewidth=2,
            linestyle="-",
            color=colors[index],
        )

    ax.legend(labels)
    ax.set_xlabel("Emergence time [h]")
    ax.set_ylabel("Plants emerged [%]")
    ax.grid(axis="y", color="silver", linestyle="--", linewidth=0.5)

    if save:
        fig.savefig(
            output_dir / f"{ALL_TRAYS_FILENAME}.{output_format}", dpi=FIGURE_DPI
        )
    if show:
        plt.show()
    plt.close(fig)


def export_parameters(
    parameters: list[list[float]],
    tray_names: pd.Index,
    output_dir: Path,
) -> pd.DataFrame:
    """Save fitted emergence parameters to CSV.

    Args:
        parameters: Fitted sigmoid parameters for each tray.
        tray_names: Column labels from the normalized dataset.
        output_dir: Directory where the parameters CSV is saved.

    Returns:
        DataFrame containing EC50, slope, and percent germinated per tray.

    Author:
        Jakub Vašák

    """
    export_data = pd.DataFrame(
        parameters,
        index=tray_names,
        columns=["EC50", "Slope", "%_germinated"],
    )
    export_data["Slope"] = 10 ** export_data["Slope"]
    export_data.to_csv(output_dir / PARAMETERS_CSV, sep=CSV_SEPARATOR)
    return export_data


def graphs(
    dataset: pd.DataFrame,
    output_format: str = OUTPUT_FORMAT,
    show: bool = SHOW_GRAPHS,
    save: bool = SAVE_GRAPHS,
    export_csv: bool = SAVE_PARAMETERS_CSV,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Plot tray-level and combined emergence graphs from a results dataset.

    Args:
        dataset: Raw results table loaded from the configured CSV file.
        output_format: Image format used when saving figures.
        show: Whether to display figures interactively.
        save: Whether to save generated figures.
        export_csv: Whether to save fitted EC50/slope/germination parameters.
        output_dir: Project-specific directory for generated artifacts.

    Returns:
        DataFrame with fitted emergence parameters for each tray.

    Author:
        Jakub Vašák

    """
    if save or export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    prepared_dataset, numbers_of_plants = prepare_dataset(dataset)
    maximum_time = float(prepared_dataset.max(numeric_only=True).max())
    minimum_time = float(prepared_dataset.min(numeric_only=True).min())

    parameters = [
        plot_tray_graph(
            dataset=prepared_dataset,
            tray=tray,
            number_of_plants=numbers_of_plants[index],
            minimum_time=minimum_time,
            maximum_time=maximum_time,
            output_dir=output_dir,
            output_format=output_format,
            save=save,
            show=show,
        )
        for index, tray in enumerate(prepared_dataset.columns)
    ]

    plot_all_trays(
        dataset=prepared_dataset,
        parameters=parameters,
        minimum_time=minimum_time,
        maximum_time=maximum_time,
        output_dir=output_dir,
        output_format=output_format,
        save=save,
        show=show,
    )

    if export_csv:
        return export_parameters(parameters, prepared_dataset.columns, output_dir)

    return pd.DataFrame(
        parameters,
        index=prepared_dataset.columns,
        columns=["EC50", "Slope", "%_germinated"],
    )


def read_results(input_csv: Path = INPUT_CSV) -> pd.DataFrame:
    """Read the configured emergence results CSV.

    Args:
        input_csv: Generated comma-separated CSV from ``5_1-evaluation.py`` or
            legacy semicolon-separated CSV with tray names in the first row.

    Returns:
        Raw results DataFrame ready for ``graphs``.

    Raises:
        FileNotFoundError: If ``input_csv`` does not exist.

    Author:
        Jakub Vašák

    """
    if not input_csv.is_file():
        msg = f"Input results CSV does not exist: {input_csv}"
        raise FileNotFoundError(msg)

    with input_csv.open(encoding="utf-8") as file:
        header_line = file.readline().strip()

    if (
        header_line.split(GENERATED_RESULTS_SEPARATOR)[0]
        == GENERATED_RESULTS_NAME_COLUMN
    ):
        return pd.read_csv(input_csv, sep=GENERATED_RESULTS_SEPARATOR)

    if header_line.split(CSV_SEPARATOR)[0] == GENERATED_RESULTS_NAME_COLUMN:
        return pd.read_csv(input_csv, sep=CSV_SEPARATOR)

    return pd.read_csv(input_csv, sep=CSV_SEPARATOR, header=None, decimal=CSV_DECIMAL)


def main() -> None:
    """Run the emergence graph evaluation with the configured paths.

    Author:
        Jakub Vašák
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    dataset = read_results()
    graphs(dataset)


if __name__ == "__main__":
    main()
