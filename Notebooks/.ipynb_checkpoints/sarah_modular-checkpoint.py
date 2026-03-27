
"""
Reusable demographic and experiment-group analysis module.

This module converts the notebook logic into reusable Python functions so the
same workflow can be applied to other datasets without keeping analysis steps
hardcoded inside a notebook.

Main capabilities
-----------------
1. Load configuration and source files from YAML or direct paths.
2. Clean the demographic dataset by removing unusable rows.
3. Explore logon and call behavior for specific user segments.
4. Generate descriptive statistics for age distributions.
5. Create reusable age-group and tenure-group boxplots.
6. Merge demographic data with experiment assignments.
7. Split the merged dataset into test and control groups.
8. Save cleaned and derived datasets when needed.

All functions include comments and docstrings in English to explain their
purpose, inputs, outputs, and intended reuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


# -------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------

# Default palette choices used in the original notebook.
DEFAULT_LOGON_PALETTE = "pastel"
DEFAULT_CALL_PALETTE = "bright"

# Default age bins used to group clients.
DEFAULT_AGE_BINS = list(range(20, 100, 10))

# Default tenure interval size used to create tenure groups.
DEFAULT_TENURE_BIN_WIDTH = 5


# -------------------------------------------------------------------
# Config and file loading helpers
# -------------------------------------------------------------------

def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file into a Python dictionary.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration values.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_csv_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(file_path)


def load_demo_data_from_config(
    config_path: str | Path,
    section: str = "input_data",
    file_key: str = "file1",
) -> pd.DataFrame:
    """
    Load the demographic dataset using a YAML config reference.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.
    section : str, default 'input_data'
        Top-level section containing the file path.
    file_key : str, default 'file1'
        Key for the demographic file.

    Returns
    -------
    pd.DataFrame
        Demographic dataset.
    """
    config = load_yaml_config(config_path)
    file_path = config[section][file_key]
    return load_csv_file(file_path)


def load_experiment_data_from_config(
    config_path: str | Path,
    section: str = "output_data",
    file_key: str = "file2",
) -> pd.DataFrame:
    """
    Load the experiment-assignment dataset using a YAML config reference.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.
    section : str, default 'output_data'
        Top-level section containing the file path.
    file_key : str, default 'file2'
        Key for the experiment file.

    Returns
    -------
    pd.DataFrame
        Experiment-assignment dataset.
    """
    config = load_yaml_config(config_path)
    file_path = config[section][file_key]
    return load_csv_file(file_path)


# -------------------------------------------------------------------
# Data cleaning and inspection helpers
# -------------------------------------------------------------------

def clean_demographic_data(
    demo_df: pd.DataFrame,
    id_col: str = "client_id",
) -> pd.DataFrame:
    """
    Remove rows that contain no usable information except the client ID.

    The notebook treated these rows as unusable because they do not add
    demographic or behavioral information to the analysis.

    Parameters
    ----------
    demo_df : pd.DataFrame
        Raw demographic dataset.
    id_col : str, default 'client_id'
        Identifier column that should be ignored when checking whether
        a row is fully empty.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with index reset.
    """
    cleaned_df = demo_df.dropna(how="all", subset=demo_df.columns.difference([id_col]))
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df


def summarize_dataframe_structure(df: pd.DataFrame) -> dict:
    """
    Build a lightweight structural summary of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Summary including shape, columns, and dtypes.
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def sort_by_column(
    df: pd.DataFrame,
    column: str,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Sort a DataFrame by a chosen column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Column used for sorting.
    ascending : bool, default True
        Sort direction.

    Returns
    -------
    pd.DataFrame
        Sorted copy of the data.
    """
    return df.sort_values(by=column, ascending=ascending).copy()


def get_sorted_unique_values(df: pd.DataFrame, column: str) -> list:
    """
    Return sorted unique values from a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Column whose unique values should be returned.

    Returns
    -------
    list
        Sorted list of unique values.
    """
    return sorted(df[column].dropna().unique().tolist())


# -------------------------------------------------------------------
# Segment extraction and descriptive analysis
# -------------------------------------------------------------------

def filter_clients_by_exact_value(
    df: pd.DataFrame,
    column: str,
    value,
) -> pd.DataFrame:
    """
    Filter rows where a column matches an exact value.

    This generalizes notebook logic such as selecting clients with exactly
    9 logons in the last 6 months.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Column used for filtering.
    value : any
        Exact value to match.

    Returns
    -------
    pd.DataFrame
        Filtered subset.
    """
    return df[df[column] == value].copy()


def age_distribution_table(
    df: pd.DataFrame,
    age_col: str = "clnt_age",
    count_col_name: str = "client_count",
) -> pd.DataFrame:
    """
    Count how many clients exist at each age.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset, typically already filtered to a segment of interest.
    age_col : str, default 'clnt_age'
        Age column.
    count_col_name : str, default 'client_count'
        Name for the output count column.

    Returns
    -------
    pd.DataFrame
        Age-frequency table.
    """
    age_dist = df.groupby(age_col).size().reset_index(name=count_col_name)
    return age_dist.sort_values(by=age_col).copy()


def calculate_age_statistics(
    df: pd.DataFrame,
    age_col: str = "clnt_age",
) -> dict:
    """
    Calculate descriptive statistics for the age distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    age_col : str, default 'clnt_age'
        Age column.

    Returns
    -------
    dict
        Mean, median, mode, and quartiles for the age variable.
    """
    series = df[age_col].dropna()

    return {
        "mean_age": series.mean(),
        "median_age": series.median(),
        "mode_age": series.mode().tolist(),
        "quartiles": series.quantile([0.25, 0.50, 0.75]).to_dict(),
    }


def filter_clients_with_no_activity(
    df: pd.DataFrame,
    logon_col: str = "logons_6_mnth",
    calls_col: str = "calls_6_mnth",
) -> pd.DataFrame:
    """
    Filter clients with zero logons and zero calls in the last 6 months.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    logon_col : str, default 'logons_6_mnth'
        Logon-count column.
    calls_col : str, default 'calls_6_mnth'
        Call-count column.

    Returns
    -------
    pd.DataFrame
        Subset of inactive clients.
    """
    return df[(df[logon_col] == 0) & (df[calls_col] == 0)].copy()


def extract_selected_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """
    Return a copy containing only selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : sequence[str]
        Columns to keep.

    Returns
    -------
    pd.DataFrame
        Reduced DataFrame.
    """
    return df[list(columns)].copy()


# -------------------------------------------------------------------
# Grouping helpers
# -------------------------------------------------------------------

def add_age_groups(
    df: pd.DataFrame,
    age_col: str = "clnt_age",
    bins: Sequence[int] = DEFAULT_AGE_BINS,
    output_col: str = "age_group",
) -> pd.DataFrame:
    """
    Add age-group categories to the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    age_col : str, default 'clnt_age'
        Age column.
    bins : sequence[int], default DEFAULT_AGE_BINS
        Bin edges used to create age groups.
    output_col : str, default 'age_group'
        Name of the output category column.

    Returns
    -------
    pd.DataFrame
        Copy with age-group column added.
    """
    result = df.copy()
    result[output_col] = pd.cut(result[age_col], bins=bins)
    return result


def create_tenure_groups(
    df: pd.DataFrame,
    tenure_col: str = "clnt_tenure_yr",
    bin_width: int = DEFAULT_TENURE_BIN_WIDTH,
    output_col: str = "tenure_group",
) -> pd.DataFrame:
    """
    Create tenure-group categories using fixed-width bins.

    The maximum tenure is rounded up so the final bin fully covers the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    tenure_col : str, default 'clnt_tenure_yr'
        Tenure column.
    bin_width : int, default 5
        Width of each tenure interval.
    output_col : str, default 'tenure_group'
        Name of the output category column.

    Returns
    -------
    pd.DataFrame
        Copy with tenure-group column added.
    """
    result = df.copy()
    max_tenure_numeric = np.ceil(result[tenure_col].max())
    bin_edges = np.arange(0, max_tenure_numeric + bin_width, bin_width)
    result[output_col] = pd.cut(result[tenure_col], bins=bin_edges)
    return result


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def _ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent directory for a file output if it does not exist.

    Parameters
    ----------
    path : str | Path
        Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_boxplot_by_group(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    palette: str = "pastel",
    figsize: tuple[int, int] = (12, 6),
    rotation: int = 0,
    save_path: str | Path | None = None,
) -> None:
    """
    Create a reusable boxplot for one grouped feature against one metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    x_col : str
        Categorical column displayed on the x-axis.
    y_col : str
        Numeric column displayed on the y-axis.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    palette : str, default 'pastel'
        Seaborn palette name.
    figsize : tuple[int, int], default (12, 6)
        Figure size.
    rotation : int, default 0
        Rotation for x tick labels.
    save_path : str | Path | None, default None
        Optional figure output path.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=x_col, y=y_col, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_single_variable_boxplot(
    df: pd.DataFrame,
    column: str,
    color: str = "violet",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 4),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a single-variable boxplot to inspect spread and outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Numeric column to plot.
    color : str, default 'violet'
        Box color.
    title : str | None, default None
        Optional plot title.
    figsize : tuple[int, int], default (10, 4)
        Figure size.
    save_path : str | Path | None, default None
        Optional figure output path.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=df[column], color=color)

    if title:
        plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_logons_by_age_group(
    df: pd.DataFrame,
    age_group_col: str = "age_group",
    logon_col: str = "logons_6_mnth",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot logons by age group.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that already contains an age-group column.
    age_group_col : str, default 'age_group'
        Age-group category column.
    logon_col : str, default 'logons_6_mnth'
        Logon metric column.
    save_path : str | Path | None, default None
        Optional figure output path.
    """
    plot_boxplot_by_group(
        df=df,
        x_col=age_group_col,
        y_col=logon_col,
        title="Boxplot of Logons by Age Group",
        xlabel="Age Group",
        ylabel="Logons in the Last 6 Months",
        palette=DEFAULT_LOGON_PALETTE,
        save_path=save_path,
    )


def plot_calls_by_age_group(
    df: pd.DataFrame,
    age_group_col: str = "age_group",
    calls_col: str = "calls_6_mnth",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot calls by age group.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that already contains an age-group column.
    age_group_col : str, default 'age_group'
        Age-group category column.
    calls_col : str, default 'calls_6_mnth'
        Calls metric column.
    save_path : str | Path | None, default None
        Optional figure output path.
    """
    plot_boxplot_by_group(
        df=df,
        x_col=age_group_col,
        y_col=calls_col,
        title="Boxplot of Calls by Age Group",
        xlabel="Age Group",
        ylabel="Calls in the Last 6 Months",
        palette=DEFAULT_CALL_PALETTE,
        save_path=save_path,
    )


def plot_logons_by_tenure_group(
    df: pd.DataFrame,
    tenure_group_col: str = "tenure_group",
    logon_col: str = "logons_6_mnth",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot logons by tenure group.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that already contains a tenure-group column.
    tenure_group_col : str, default 'tenure_group'
        Tenure-group category column.
    logon_col : str, default 'logons_6_mnth'
        Logon metric column.
    save_path : str | Path | None, default None
        Optional figure output path.
    """
    plot_boxplot_by_group(
        df=df,
        x_col=tenure_group_col,
        y_col=logon_col,
        title="Boxplot of Logons per Tenure Group",
        xlabel="Tenure Group",
        ylabel="Logons in the Last 6 Months",
        palette=DEFAULT_LOGON_PALETTE,
        figsize=(14, 6),
        save_path=save_path,
    )


def plot_calls_by_tenure_group(
    df: pd.DataFrame,
    tenure_group_col: str = "tenure_group",
    calls_col: str = "calls_6_mnth",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot calls by tenure group.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that already contains a tenure-group column.
    tenure_group_col : str, default 'tenure_group'
        Tenure-group category column.
    calls_col : str, default 'calls_6_mnth'
        Calls metric column.
    save_path : str | Path | None, default None
        Optional figure output path.
    """
    plot_boxplot_by_group(
        df=df,
        x_col=tenure_group_col,
        y_col=calls_col,
        title="Boxplot of Calls per Tenure Group",
        xlabel="Tenure Group",
        ylabel="Calls in the Last 6 Months",
        palette=DEFAULT_CALL_PALETTE,
        figsize=(14, 6),
        save_path=save_path,
    )


# -------------------------------------------------------------------
# Merge and split helpers
# -------------------------------------------------------------------

def merge_demo_with_experiment_data(
    demo_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
    on: str = "client_id",
) -> pd.DataFrame:
    """
    Merge demographic data with experiment assignments.

    Parameters
    ----------
    demo_df : pd.DataFrame
        Cleaned demographic dataset.
    experiment_df : pd.DataFrame
        Experiment-assignment dataset.
    on : str, default 'client_id'
        Merge key.

    Returns
    -------
    pd.DataFrame
        Merged dataset with column names standardized to lowercase.
    """
    merged_df = pd.merge(demo_df, experiment_df, on=on).copy()
    merged_df.columns = merged_df.columns.str.lower()
    return merged_df


def split_variation_groups(
    df: pd.DataFrame,
    variation_col: str = "variation",
    test_label: str = "test",
    control_label: str = "control",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the merged dataset into test and control subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset containing experiment labels.
    variation_col : str, default 'variation'
        Variation-assignment column.
    test_label : str, default 'test'
        Test-group label.
    control_label : str, default 'control'
        Control-group label.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (test_group_df, control_group_df)
    """
    test_group_df = df[df[variation_col] == test_label].copy()
    control_group_df = df[df[variation_col] == control_label].copy()
    return test_group_df, control_group_df


def count_group_sizes(
    test_group_df: pd.DataFrame,
    control_group_df: pd.DataFrame,
) -> dict:
    """
    Count the number of rows in the test and control datasets.

    Parameters
    ----------
    test_group_df : pd.DataFrame
        Test-group dataset.
    control_group_df : pd.DataFrame
        Control-group dataset.

    Returns
    -------
    dict
        Row counts for each variation group.
    """
    return {
        "test_rows": len(test_group_df),
        "control_rows": len(control_group_df),
    }


# -------------------------------------------------------------------
# Saving helpers
# -------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, output_path: str | Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to save.
    output_path : str | Path
        Destination CSV path.
    index : bool, default False
        Whether to save the DataFrame index.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


# -------------------------------------------------------------------
# End-to-end pipeline
# -------------------------------------------------------------------

def run_demographic_experiment_analysis(
    config_path: str | Path,
    age_bins: Sequence[int] = DEFAULT_AGE_BINS,
    tenure_bin_width: int = DEFAULT_TENURE_BIN_WIDTH,
    top_logon_value: int = 9,
) -> dict:
    """
    Run the full demographic + experiment-group workflow.

    This function orchestrates the major notebook steps in a reusable way.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.
    age_bins : sequence[int], default DEFAULT_AGE_BINS
        Age-group bin edges.
    tenure_bin_width : int, default 5
        Width of tenure bins.
    top_logon_value : int, default 9
        Exact logon value used to inspect a high-logon segment.

    Returns
    -------
    dict
        Main outputs generated by the analysis.
    """
    # Load input tables.
    demo_df = load_demo_data_from_config(config_path)
    experiment_df = load_experiment_data_from_config(config_path)

    # Clean the demographic file by removing rows that are empty except for client_id.
    demo_df_cleaned = clean_demographic_data(demo_df)

    # Explore logon values and the subset with the target logon count.
    unique_logons = get_sorted_unique_values(demo_df_cleaned, "logons_6_mnth")
    top_users = filter_clients_by_exact_value(demo_df_cleaned, "logons_6_mnth", top_logon_value)
    top_users_age = sort_by_column(top_users, "clnt_age")
    age_distribution = age_distribution_table(top_users_age)
    age_stats = calculate_age_statistics(top_users_age)

    # Identify users with no digital or phone activity.
    no_activity_df = filter_clients_with_no_activity(demo_df_cleaned)
    no_activity_selected = extract_selected_columns(no_activity_df, ["clnt_age", "clnt_tenure_yr"])

    # Add age and tenure groupings for plotting or grouped analysis.
    demo_with_age_groups = add_age_groups(demo_df_cleaned, bins=age_bins)
    demo_with_tenure_groups = create_tenure_groups(demo_with_age_groups, bin_width=tenure_bin_width)

    # Merge with experiment assignments and split into variation groups.
    joined_demo_expi_df = merge_demo_with_experiment_data(demo_df_cleaned, experiment_df)
    test_group_df, control_group_df = split_variation_groups(joined_demo_expi_df)
    group_sizes = count_group_sizes(test_group_df, control_group_df)

    return {
        "demo_df": demo_df,
        "demo_df_cleaned": demo_df_cleaned,
        "experiment_df": experiment_df,
        "unique_logons": unique_logons,
        "top_users": top_users,
        "top_users_age": top_users_age,
        "age_distribution": age_distribution,
        "age_stats": age_stats,
        "no_activity_df": no_activity_df,
        "no_activity_selected": no_activity_selected,
        "demo_with_age_groups": demo_with_age_groups,
        "demo_with_tenure_groups": demo_with_tenure_groups,
        "joined_demo_expi_df": joined_demo_expi_df,
        "test_group_df": test_group_df,
        "control_group_df": control_group_df,
        "group_sizes": group_sizes,
    }


def main() -> None:
    """
    Example entry point for running the module as a script.

    Update the file paths if your project structure differs.
    """
    config_path = "../config.yaml"

    results = run_demographic_experiment_analysis(config_path=config_path)

    print("Cleaned demographic dataset shape:", results["demo_df_cleaned"].shape)
    print("Unique logon values:", results["unique_logons"])
    print("Age statistics for top logon users:", results["age_stats"])
    print("Variation group sizes:", results["group_sizes"])

    # Example plot generation.
    plot_logons_by_age_group(results["demo_with_age_groups"])
    plot_calls_by_age_group(results["demo_with_age_groups"])
    plot_single_variable_boxplot(
        results["demo_df_cleaned"],
        column="clnt_tenure_yr",
        title="Tenure Distribution",
    )
    plot_logons_by_tenure_group(results["demo_with_tenure_groups"])
    plot_calls_by_tenure_group(results["demo_with_tenure_groups"])

    # Example saves.
    save_dataframe(results["demo_df_cleaned"], "demo_df_cleaned.csv")
    save_dataframe(results["joined_demo_expi_df"], "joined_demo_expi_df.csv")
    save_dataframe(results["test_group_df"], "test_group_df.csv")
    save_dataframe(results["control_group_df"], "control_group_df.csv")


if __name__ == "__main__":
    main()
