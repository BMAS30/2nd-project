
"""
Reusable tenure and funnel-segmentation analysis module.

This module converts the notebook logic into reusable Python functions so the
same workflow can be applied to other A/B test datasets without keeping the
analysis hardcoded inside a notebook.

Main capabilities
-----------------
1. Load configuration and source files from YAML or direct paths.
2. Compare tenure distributions between test and control groups.
3. Run normality, variance, and mean-comparison statistical tests.
4. Detect incomplete confirmers in the web journey data.
5. Build an error-free web dataset filtered to valid experiment clients.
6. Measure completion rates overall and by tenure group.
7. Analyze abandonment by funnel step and by variation group.
8. Detect clients who appear in web data without a 'start' event.
9. Build reusable plots for tenure distributions, completion rates,
   chi-square critical regions, tenure-group performance, and funnel views.
10. Export Tableau-ready summary tables when needed.

All functions include comments and docstrings in English to explain their
purpose, inputs, outputs, and intended reuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from scipy.stats import chi2_contingency


# -------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------

DEFAULT_STEPS = ["start", "step_1", "step_2", "step_3", "confirm"]

DEFAULT_BROAD_TENURE_BINS = [0, 5, 15, 30, 55]
DEFAULT_BROAD_TENURE_LABELS = ["New (0-5yr)", "Mid (6-15yr)", "Loyal (16-30yr)", "Veteran (30+yr)"]

DEFAULT_DETAIL_TENURE_BINS = [0, 5, 10, 15, 20, 25, 30, 55]
DEFAULT_DETAIL_TENURE_LABELS = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31+"]

DEFAULT_COLORS = {
    "test": "steelblue",
    "control": "coral",
}


# -------------------------------------------------------------------
# Config and loading helpers
# -------------------------------------------------------------------

def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file and return its parsed content.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_csv_file(file_path: str | Path, parse_dates: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Load a CSV file and optionally parse one or more datetime columns.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.
    parse_dates : sequence[str] | None, default None
        Columns to convert to datetime if present.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    df = pd.read_csv(file_path)

    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    return df


def load_analysis_inputs_from_config(
    config_path: str | Path,
    test_key: str = "file4",
    control_key: str = "file3",
    web_key: str = "file6",
    section: str = "output_data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three core datasets used in the notebook:
    test-group clients, control-group clients, and web-event data.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.
    test_key : str, default 'file4'
        Key for the test-group client file.
    control_key : str, default 'file3'
        Key for the control-group client file.
    web_key : str, default 'file6'
        Key for the web-event file.
    section : str, default 'output_data'
        Config section containing the file paths.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (clients_test, clients_control, df_web)
    """
    config = load_yaml_config(config_path)
    clients_test = load_csv_file(config[section][test_key])
    clients_control = load_csv_file(config[section][control_key])
    df_web = load_csv_file(config[section][web_key], parse_dates=["date_time"])
    return clients_test, clients_control, df_web


# -------------------------------------------------------------------
# Generic filtering helpers
# -------------------------------------------------------------------

def filter_by_exact_value(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """
    Return only rows where a column matches one exact value.

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


def split_clients_by_gender(
    df: pd.DataFrame,
    gender_col: str = "gendr",
    female_label: str = "F",
    male_label: str = "M",
    unknown_label: str = "U",
) -> dict[str, pd.DataFrame]:
    """
    Split a client table into female, male, and unknown-gender subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Client-level dataset.
    gender_col : str, default 'gendr'
        Gender column.
    female_label : str, default 'F'
        Female code.
    male_label : str, default 'M'
        Male code.
    unknown_label : str, default 'U'
        Unknown code.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing gender-specific subsets.
    """
    return {
        "female": filter_by_exact_value(df, gender_col, female_label),
        "male": filter_by_exact_value(df, gender_col, male_label),
        "unknown": filter_by_exact_value(df, gender_col, unknown_label),
    }


# -------------------------------------------------------------------
# Tenure-distribution analysis
# -------------------------------------------------------------------

def describe_tenure_distribution(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    tenure_col: str = "clnt_tenure_yr",
) -> pd.DataFrame:
    """
    Build descriptive statistics for tenure in test and control groups.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    tenure_col : str, default 'clnt_tenure_yr'
        Tenure column.

    Returns
    -------
    pd.DataFrame
        Side-by-side descriptive statistics.
    """
    test_desc = test_df[tenure_col].describe().rename("test")
    control_desc = control_df[tenure_col].describe().rename("control")
    return pd.concat([test_desc, control_desc], axis=1)


def run_tenure_normality_tests(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    tenure_col: str = "clnt_tenure_yr",
) -> dict:
    """
    Run D'Agostino-Pearson normality tests on tenure for both groups.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    tenure_col : str, default 'clnt_tenure_yr'
        Tenure column.

    Returns
    -------
    dict
        Test statistics and p-values for both groups.
    """
    stat_test, p_test = stats.normaltest(test_df[tenure_col].dropna())
    stat_control, p_control = stats.normaltest(control_df[tenure_col].dropna())

    return {
        "test_stat": stat_test,
        "test_p_value": p_test,
        "control_stat": stat_control,
        "control_p_value": p_control,
    }


def run_tenure_levene_test(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    tenure_col: str = "clnt_tenure_yr",
) -> dict:
    """
    Run Levene's test for equality of tenure variance across groups.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    tenure_col : str, default 'clnt_tenure_yr'
        Tenure column.

    Returns
    -------
    dict
        Levene statistic and p-value.
    """
    stat, p_value = stats.levene(
        test_df[tenure_col].dropna(),
        control_df[tenure_col].dropna(),
    )
    return {"levene_stat": stat, "levene_p_value": p_value}


def run_tenure_ttest(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    tenure_col: str = "clnt_tenure_yr",
    equal_var: bool = True,
) -> dict:
    """
    Run an independent-samples t-test on tenure.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    tenure_col : str, default 'clnt_tenure_yr'
        Tenure column.
    equal_var : bool, default True
        Whether to assume equal variance.

    Returns
    -------
    dict
        T-statistic and p-value.
    """
    stat, p_value = stats.ttest_ind(
        test_df[tenure_col].dropna(),
        control_df[tenure_col].dropna(),
        equal_var=equal_var,
    )
    return {"t_stat": stat, "p_value": p_value, "equal_var": equal_var}


# -------------------------------------------------------------------
# Funnel integrity helpers
# -------------------------------------------------------------------

def identify_confirmed_clients(
    df_web: pd.DataFrame,
    confirm_step: str = "confirm",
    client_col: str = "client_id",
    step_col: str = "process_step",
) -> np.ndarray:
    """
    Return the client IDs that reached the confirm step.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event dataset.
    confirm_step : str, default 'confirm'
        Final funnel step.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Step column.

    Returns
    -------
    np.ndarray
        Unique confirmed client IDs.
    """
    return df_web[df_web[step_col] == confirm_step][client_col].unique()


def find_incomplete_confirmers(
    df_web: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    client_col: str = "client_id",
    step_col: str = "process_step",
) -> tuple[pd.Series, pd.Index]:
    """
    Find clients who reached confirm but do not contain all required steps.

    The notebook used this logic to detect problematic journeys before
    computing completion and abandonment metrics.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event dataset.
    steps : sequence[str], default DEFAULT_STEPS
        Required ordered funnel steps.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Step column.

    Returns
    -------
    tuple[pd.Series, pd.Index]
        Boolean series per confirmed client indicating whether all steps exist,
        and the index of incomplete confirmers.
    """
    confirmed_clients = identify_confirmed_clients(df_web, client_col=client_col, step_col=step_col)

    complete_journeys = (
        df_web[df_web[client_col].isin(confirmed_clients)]
        .groupby(client_col)[step_col]
        .apply(set)
    )

    complete_journeys = complete_journeys.apply(lambda x: all(step in x for step in steps))
    incomplete_confirmers = complete_journeys[~complete_journeys].index
    return complete_journeys, incomplete_confirmers


def count_incomplete_confirmers_by_group(
    incomplete_confirmers: Sequence,
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    client_col: str = "client_id",
) -> dict:
    """
    Count how many incomplete confirmers belong to test and control groups.

    Parameters
    ----------
    incomplete_confirmers : sequence
        Client IDs flagged as incomplete confirmers.
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    client_col : str, default 'client_id'
        Client identifier column.

    Returns
    -------
    dict
        Counts by variation group.
    """
    incomplete_confirmers = pd.Index(incomplete_confirmers)
    return {
        "test_errors": int(incomplete_confirmers.isin(test_df[client_col]).sum()),
        "control_errors": int(incomplete_confirmers.isin(control_df[client_col]).sum()),
    }


def build_error_free_web_data(
    df_web: pd.DataFrame,
    incomplete_confirmers: Sequence,
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    client_col: str = "client_id",
) -> pd.DataFrame:
    """
    Remove incomplete confirmers and keep only clients that belong to
    the experiment test or control groups.

    Parameters
    ----------
    df_web : pd.DataFrame
        Original web-event dataset.
    incomplete_confirmers : sequence
        Problematic confirmed client IDs.
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    client_col : str, default 'client_id'
        Client identifier column.

    Returns
    -------
    pd.DataFrame
        Cleaned web-event dataset.
    """
    error_free_df = df_web[~df_web[client_col].isin(incomplete_confirmers)].copy()

    all_clients = pd.concat([test_df[[client_col]], control_df[[client_col]]], axis=0)
    error_free_df = error_free_df[error_free_df[client_col].isin(all_clients[client_col])].copy()

    return error_free_df


# -------------------------------------------------------------------
# Completion analysis
# -------------------------------------------------------------------

def add_completed_flag(
    clients_df: pd.DataFrame,
    completed_client_ids: Sequence,
    client_col: str = "client_id",
    output_col: str = "completed",
) -> pd.DataFrame:
    """
    Add a binary completed flag to a client-level dataset.

    Parameters
    ----------
    clients_df : pd.DataFrame
        Client-level dataset.
    completed_client_ids : sequence
        Client IDs that completed the funnel.
    client_col : str, default 'client_id'
        Client identifier column.
    output_col : str, default 'completed'
        Name of the binary output column.

    Returns
    -------
    pd.DataFrame
        Copy with completion flag added.
    """
    result = clients_df.copy()
    result[output_col] = result[client_col].isin(completed_client_ids).astype(int)
    return result


def build_completion_contingency_table(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    completed_client_ids: Sequence,
    client_col: str = "client_id",
) -> pd.DataFrame:
    """
    Build a completion vs non-completion contingency table for chi-square.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    completed_client_ids : sequence
        Client IDs that completed the funnel.
    client_col : str, default 'client_id'
        Client identifier column.

    Returns
    -------
    pd.DataFrame
        Contingency table with one row per group.
    """
    test_completed = int(test_df[client_col].isin(completed_client_ids).sum())
    control_completed = int(control_df[client_col].isin(completed_client_ids).sum())

    test_not_completed = int(len(test_df) - test_completed)
    control_not_completed = int(len(control_df) - control_completed)

    contingency_table = pd.DataFrame(
        {
            "Completed": [test_completed, control_completed],
            "Not Completed": [test_not_completed, control_not_completed],
        },
        index=["Test", "Control"],
    )
    return contingency_table


def calculate_group_completion_rates(
    contingency_table: pd.DataFrame,
) -> dict:
    """
    Calculate completion rates from a 2x2 contingency table.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Completion table with rows Test and Control.

    Returns
    -------
    dict
        Completion percentages for both groups.
    """
    test_rate = contingency_table.loc["Test", "Completed"] / contingency_table.loc["Test"].sum() * 100
    control_rate = contingency_table.loc["Control", "Completed"] / contingency_table.loc["Control"].sum() * 100
    return {
        "test_completion_rate_pct": test_rate,
        "control_completion_rate_pct": control_rate,
    }


def run_completion_chi_square_test(contingency_table: pd.DataFrame) -> dict:
    """
    Run a chi-square test on the completion contingency table.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        2x2 completion table.

    Returns
    -------
    dict
        Chi-square statistic, p-value, degrees of freedom, and expected values.
    """
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "expected": expected,
    }


# -------------------------------------------------------------------
# Tenure-group performance
# -------------------------------------------------------------------

def add_tenure_groups(
    df: pd.DataFrame,
    tenure_col: str = "clnt_tenure_yr",
    bins: Sequence[float] = DEFAULT_BROAD_TENURE_BINS,
    labels: Sequence[str] = DEFAULT_BROAD_TENURE_LABELS,
    output_col: str = "tenure_group",
) -> pd.DataFrame:
    """
    Add tenure-group labels to a client-level dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Client-level dataset.
    tenure_col : str, default 'clnt_tenure_yr'
        Tenure column.
    bins : sequence[float], default DEFAULT_BROAD_TENURE_BINS
        Bin edges.
    labels : sequence[str], default DEFAULT_BROAD_TENURE_LABELS
        Bin labels.
    output_col : str, default 'tenure_group'
        Output column name.

    Returns
    -------
    pd.DataFrame
        Copy with tenure groups added.
    """
    result = df.copy()
    result[output_col] = pd.cut(result[tenure_col], bins=bins, labels=labels)
    return result


def calculate_completion_rate_by_tenure_group(
    df: pd.DataFrame,
    tenure_group_col: str = "tenure_group",
    completed_col: str = "completed",
) -> pd.Series:
    """
    Calculate completion rate by tenure group.

    Parameters
    ----------
    df : pd.DataFrame
        Client-level dataset containing tenure groups and a completed flag.
    tenure_group_col : str, default 'tenure_group'
        Tenure-group column.
    completed_col : str, default 'completed'
        Binary completed flag.

    Returns
    -------
    pd.Series
        Completion rate in percent by tenure group.
    """
    return df.groupby(tenure_group_col, observed=False)[completed_col].mean() * 100


def count_clients_by_tenure_group(
    df: pd.DataFrame,
    tenure_group_col: str = "tenure_group",
) -> pd.Series:
    """
    Count clients by tenure group.

    Parameters
    ----------
    df : pd.DataFrame
        Client-level dataset containing tenure groups.
    tenure_group_col : str, default 'tenure_group'
        Tenure-group column.

    Returns
    -------
    pd.Series
        Client counts by tenure group.
    """
    return df[tenure_group_col].value_counts().sort_index()


def build_tenure_completion_table(
    test_rates: pd.Series,
    control_rates: pd.Series,
    tenure_group_name: str = "Tenure Group",
) -> pd.DataFrame:
    """
    Build a wide tenure-completion summary table.

    Parameters
    ----------
    test_rates : pd.Series
        Test-group completion rates by tenure group.
    control_rates : pd.Series
        Control-group completion rates by tenure group.
    tenure_group_name : str, default 'Tenure Group'
        Output label column name.

    Returns
    -------
    pd.DataFrame
        Wide comparison table.
    """
    df_out = pd.DataFrame(
        {
            tenure_group_name: test_rates.index.astype(str),
            "Test Rate": test_rates.values,
            "Control Rate": control_rates.reindex(test_rates.index).values,
        }
    )
    return df_out


def convert_tenure_table_to_long_format(
    tenure_df: pd.DataFrame,
    id_col: str = "Tenure Group",
) -> pd.DataFrame:
    """
    Convert the wide tenure-completion table to long format for Tableau.

    Parameters
    ----------
    tenure_df : pd.DataFrame
        Wide tenure comparison table.
    id_col : str, default 'Tenure Group'
        Identifier column.

    Returns
    -------
    pd.DataFrame
        Long-format table.
    """
    long_df = tenure_df.melt(
        id_vars=[id_col],
        value_vars=["Test Rate", "Control Rate"],
        var_name="Group",
        value_name="Completion Rate",
    )
    long_df["Group"] = long_df["Group"].str.replace(" Rate", "", regex=False)
    return long_df


# -------------------------------------------------------------------
# Abandonment and funnel-position analysis
# -------------------------------------------------------------------

def get_last_step_per_client(
    df_web: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    client_col: str = "client_id",
    step_col: str = "process_step",
) -> pd.Series:
    """
    Identify the furthest step reached by each client.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event dataset.
    steps : sequence[str], default DEFAULT_STEPS
        Ordered funnel steps.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Step column.

    Returns
    -------
    pd.Series
        Last step reached by each client.
    """
    step_index = {step: i for i, step in enumerate(steps)}

    return df_web.groupby(client_col)[step_col].apply(
        lambda x: x.iloc[x.map(step_index).argmax()]
    )


def calculate_abandonment_counts(
    df_web: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    client_col: str = "client_id",
    step_col: str = "process_step",
) -> dict:
    """
    Count how many clients abandoned after each funnel step.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event dataset.
    steps : sequence[str], default DEFAULT_STEPS
        Ordered funnel steps.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Step column.

    Returns
    -------
    dict
        Abandonment counts and completion totals.
    """
    last_step = get_last_step_per_client(df_web, steps=steps, client_col=client_col, step_col=step_col)
    dropped_at = last_step[last_step != "confirm"].value_counts()

    return {
        "abandoned_after_start": int(dropped_at.get("start", 0)),
        "abandoned_after_step_1": int(dropped_at.get("step_1", 0)),
        "abandoned_after_step_2": int(dropped_at.get("step_2", 0)),
        "abandoned_after_step_3": int(dropped_at.get("step_3", 0)),
        "total_abandoned": int(dropped_at.sum()),
        "total_completed": int((last_step == "confirm").sum()),
    }


def calculate_abandonment_counts_by_group(
    error_free_df_web: pd.DataFrame,
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    client_col: str = "client_id",
) -> dict[str, dict]:
    """
    Calculate abandonment counts separately for test and control groups.

    Parameters
    ----------
    error_free_df_web : pd.DataFrame
        Cleaned web-event dataset.
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    steps : sequence[str], default DEFAULT_STEPS
        Ordered funnel steps.
    client_col : str, default 'client_id'
        Client identifier column.

    Returns
    -------
    dict[str, dict]
        Abandonment metrics by group.
    """
    results = {}

    group_map = {
        "test": test_df[client_col],
        "control": control_df[client_col],
    }

    for group_name, group_clients in group_map.items():
        group_df = error_free_df_web[error_free_df_web[client_col].isin(group_clients)].copy()
        group_result = calculate_abandonment_counts(group_df, steps=steps, client_col=client_col)
        group_result["total_clients"] = int(len(group_clients))
        results[group_name] = group_result

    return results


def count_clients_reaching_each_step_by_group(
    df_web: pd.DataFrame,
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    client_col: str = "client_id",
    step_col: str = "process_step",
) -> dict[str, dict[str, int]]:
    """
    Count unique clients who reached each step for test and control groups.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event dataset.
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    steps : sequence[str], default DEFAULT_STEPS
        Ordered funnel steps.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Step column.

    Returns
    -------
    dict[str, dict[str, int]]
        Step reach counts for both groups.
    """
    result = {}

    for group_name, group_clients in {
        "test": test_df[client_col],
        "control": control_df[client_col],
    }.items():
        group_df = df_web[df_web[client_col].isin(group_clients)].copy()
        result[group_name] = {
            step: int(group_df[group_df[step_col] == step][client_col].nunique())
            for step in steps
        }

    return result


# -------------------------------------------------------------------
# Missing-start diagnostics
# -------------------------------------------------------------------

def find_clients_without_start(
    df_web: pd.DataFrame,
    client_col: str = "client_id",
    step_col: str = "process_step",
    start_step: str = "start",
) -> tuple[set, pd.DataFrame]:
    """
    Find clients who appear in web data but never have a start step.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event dataset.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Step column.
    start_step : str, default 'start'
        Start step label.

    Returns
    -------
    tuple[set, pd.DataFrame]
        Set of client IDs without start and their event subset.
    """
    clients_with_start = df_web[df_web[step_col] == start_step][client_col].unique()
    all_clients_in_web = df_web[client_col].unique()
    clients_without_start = set(all_clients_in_web) - set(clients_with_start)
    no_start_df = df_web[df_web[client_col].isin(clients_without_start)].copy()
    return clients_without_start, no_start_df


def find_clients_without_start_by_group(
    df_web: pd.DataFrame,
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    client_col: str = "client_id",
    step_col: str = "process_step",
    start_step: str = "start",
) -> dict[str, dict]:
    """
    Find missing-start clients separately for test and control groups.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event dataset.
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Step column.
    start_step : str, default 'start'
        Start step label.

    Returns
    -------
    dict[str, dict]
        Missing-start diagnostics by group.
    """
    results = {}

    for group_name, group_clients in {
        "test": test_df[client_col],
        "control": control_df[client_col],
    }.items():
        group_df = df_web[df_web[client_col].isin(group_clients)].copy()
        clients_without_start, no_start_df = find_clients_without_start(
            group_df,
            client_col=client_col,
            step_col=step_col,
            start_step=start_step,
        )

        results[group_name] = {
            "total_clients_in_web": int(group_df[client_col].nunique()),
            "clients_with_start": int(group_df[group_df[step_col] == start_step][client_col].nunique()),
            "clients_without_start": int(len(clients_without_start)),
            "step_counts_without_start": no_start_df[step_col].value_counts(),
            "no_start_df": no_start_df,
        }

    return results


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def _ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent directory for a file output if needed.

    Parameters
    ----------
    path : str | Path
        Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_tenure_histogram_comparison(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    tenure_col: str = "clnt_tenure_yr",
    bins: int = 30,
    test_color: str = "green",
    control_color: str = "purple",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot overlaid tenure histograms for test and control groups.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-group client dataset.
    control_df : pd.DataFrame
        Control-group client dataset.
    tenure_col : str, default 'clnt_tenure_yr'
        Tenure column.
    bins : int, default 30
        Number of histogram bins.
    test_color : str, default 'green'
        Test-group color.
    control_color : str, default 'purple'
        Control-group color.
    save_path : str | Path | None, default None
        Optional output path.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(test_df[tenure_col].dropna(), bins=bins, alpha=0.5, label="Test Group", color=test_color)
    plt.hist(control_df[tenure_col].dropna(), bins=bins, alpha=0.5, label="Control Group", color=control_color)
    plt.xlabel("Tenure (Years)")
    plt.ylabel("Number of Clients")
    plt.title("Tenure Distribution: Test vs Control Group")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_contingency_heatmap(
    contingency_table: pd.DataFrame,
    cmap: str = "Purples",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a heatmap of the completion contingency table.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        2x2 completion table.
    cmap : str, default 'Purples'
        Heatmap colormap.
    save_path : str | Path | None, default None
        Optional output path.
    """
    plt.figure(figsize=(8, 5))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap=cmap)
    plt.title("Contingency Table Heatmap: Completion by Group")
    plt.ylabel("Group")
    plt.xlabel("Outcome")
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_group_completion_rates(
    completion_rates: dict,
    groups: Sequence[str] = ("Test (New Process)", "Control (Old Process)"),
    colors: Sequence[str] = ("steelblue", "coral"),
    ylim: tuple[float, float] = (62, 70),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot overall completion rates for test and control groups.

    Parameters
    ----------
    completion_rates : dict
        Dictionary containing test and control completion percentages.
    groups : sequence[str], default (...)
        Labels for the x-axis.
    colors : sequence[str], default (...)
        Bar colors.
    ylim : tuple[float, float], default (62, 70)
        Y-axis bounds.
    save_path : str | Path | None, default None
        Optional output path.
    """
    rates = [
        completion_rates["test_completion_rate_pct"],
        completion_rates["control_completion_rate_pct"],
    ]

    plt.figure(figsize=(4, 6))
    bars = plt.bar(groups, rates, color=list(colors), width=0.3)
    plt.ylabel("Completion Rate (%)")
    plt.title("Completion Rates: Test vs Control Group")
    plt.ylim(*ylim)

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_chi_square_pdf(
    chi2_stat: float,
    df: int = 1,
    alpha: float = 0.05,
    x_max: float = 15,
    annotate_far_stat: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the chi-square probability density function and rejection region.

    Parameters
    ----------
    chi2_stat : float
        Observed chi-square statistic.
    df : int, default 1
        Degrees of freedom.
    alpha : float, default 0.05
        Significance level.
    x_max : float, default 15
        Maximum x-axis value for the chart.
    annotate_far_stat : bool, default True
        If True, annotate that the statistic may be far off the visible chart.
    save_path : str | Path | None, default None
        Optional output path.
    """
    critical_value = stats.chi2.ppf(1 - alpha, df)

    x = np.linspace(0.001, x_max, 1000)
    y = stats.chi2.pdf(x, df)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, "b-", linewidth=2)
    plt.fill_between(x, y, where=(x >= critical_value), color="red", alpha=0.5)
    plt.fill_between(x, y, where=(x < critical_value), color="blue", alpha=0.2)
    plt.axvline(x=critical_value, color="red", linewidth=2)

    if chi2_stat <= x_max:
        plt.axvline(x=chi2_stat, color="black", linewidth=1.5)
        plt.text(chi2_stat + 0.2, min(max(y) * 0.7, 0.2), f"Chi2 statistic\n{chi2_stat:.2f}", fontsize=9)
    elif annotate_far_stat:
        plt.text(
            x_max * 0.65,
            min(max(y) * 0.75, 0.2),
            f"Chi2 statistic = {chi2_stat:.2f}\n(way off the chart! →→→)",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.text(critical_value + 0.2, min(max(y) * 0.5, 0.1), f"Critical value\n{critical_value:.2f}", fontsize=9, color="red")
    plt.title("Chi-square Probability Density Function")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.ylim(0, 0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_tenure_group_completion_rates(
    test_rates: pd.Series,
    control_rates: pd.Series,
    test_label: str = "Test (New Process)",
    control_label: str = "Control (Old Process)",
    colors: dict[str, str] = DEFAULT_COLORS,
    ylim: tuple[float, float] = (58, 74),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot completion rates by tenure group for test and control.

    Parameters
    ----------
    test_rates : pd.Series
        Test-group completion rates by tenure group.
    control_rates : pd.Series
        Control-group completion rates by tenure group.
    test_label : str, default 'Test (New Process)'
        Test legend label.
    control_label : str, default 'Control (Old Process)'
        Control legend label.
    colors : dict[str, str], default DEFAULT_COLORS
        Plot colors keyed by group name.
    ylim : tuple[float, float], default (58, 74)
        Y-axis limits.
    save_path : str | Path | None, default None
        Optional output path.
    """
    labels = test_rates.index.astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, test_rates.values, width, label=test_label, color=colors["test"])
    bars2 = ax.bar(x + width / 2, control_rates.reindex(test_rates.index).values, width, label=control_label, color=colors["control"])

    ax.set_xlabel("Tenure Group (Years)")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Completion Rates by Tenure Group: Test vs Control")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(*ylim)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_client_journey_remaining(
    test_step_counts: dict[str, int],
    control_step_counts: dict[str, int],
    steps: Sequence[str] = DEFAULT_STEPS,
    test_base: int | None = None,
    control_base: int | None = None,
    test_label: str = "Test (New UI)",
    control_label: str = "Control (Old UI)",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the percentage of clients remaining at each funnel step.

    Parameters
    ----------
    test_step_counts : dict[str, int]
        Test unique-client counts by step.
    control_step_counts : dict[str, int]
        Control unique-client counts by step.
    steps : sequence[str], default DEFAULT_STEPS
        Ordered funnel steps.
    test_base : int | None, default None
        Base denominator for test percentages. If omitted, use start count.
    control_base : int | None, default None
        Base denominator for control percentages. If omitted, use start count.
    test_label : str, default 'Test (New UI)'
        Test legend label.
    control_label : str, default 'Control (Old UI)'
        Control legend label.
    save_path : str | Path | None, default None
        Optional output path.
    """
    test_base = test_base or test_step_counts[steps[0]]
    control_base = control_base or control_step_counts[steps[0]]

    test_pct = [test_step_counts[step] / test_base * 100 for step in steps]
    control_pct = [control_step_counts[step] / control_base * 100 for step in steps]

    x = np.arange(len(steps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, test_pct, width, label=test_label, color=DEFAULT_COLORS["test"])
    bars2 = ax.bar(x + width / 2, control_pct, width, label=control_label, color=DEFAULT_COLORS["control"])

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Process Step")
    ax.set_ylabel("% of Clients Remaining")
    ax.set_title("Client Journey: Test vs Control Group")
    ax.set_xticks(x)
    ax.set_xticklabels([step.title().replace("_", " ") for step in steps])
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def draw_funnel(
    ax,
    data: Sequence[float],
    labels: Sequence[str],
    color: str,
    title: str,
) -> None:
    """
    Draw a funnel-style chart on a provided matplotlib axis.

    Parameters
    ----------
    ax : matplotlib axis
        Target axis.
    data : sequence[float]
        Percentages per step.
    labels : sequence[str]
        Step labels.
    color : str
        Funnel color.
    title : str
        Subplot title.
    """
    max_width = 0.8

    for i, (pct, label) in enumerate(zip(data, labels)):
        width = max_width * pct / 100
        left = (max_width - width) / 2

        if i < len(data) - 1:
            next_width = max_width * data[i + 1] / 100
            next_left = (max_width - next_width) / 2
            coords = [
                [left, -i],
                [left + width, -i],
                [next_left + next_width, -i - 1],
                [next_left, -i - 1],
            ]
        else:
            coords = [
                [left, -i],
                [left + width, -i],
                [left + width, -i - 0.8],
                [left, -i - 0.8],
            ]

        polygon = plt.Polygon(
            coords,
            closed=True,
            facecolor=color,
            alpha=0.7,
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(polygon)

        ax.text(
            0.4,
            -i - 0.4,
            f"{label}\n{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    ax.set_xlim(0, 0.8)
    ax.set_ylim(-len(data), 1)
    ax.axis("off")
    ax.set_title(title)


def plot_funnel_comparison(
    test_pct: Sequence[float],
    control_pct: Sequence[float],
    labels: Sequence[str] = ("Start", "Step 1", "Step 2", "Step 3", "Confirm"),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot side-by-side funnel charts for test and control groups.

    Parameters
    ----------
    test_pct : sequence[float]
        Test percentages by funnel step.
    control_pct : sequence[float]
        Control percentages by funnel step.
    labels : sequence[str], default (...)
        Step labels.
    save_path : str | Path | None, default None
        Optional output path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    draw_funnel(ax1, test_pct, labels, DEFAULT_COLORS["test"], "Test Funnel")
    draw_funnel(ax2, control_pct, labels, DEFAULT_COLORS["control"], "Control Funnel")
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def draw_reverse_funnel(
    ax,
    data: Sequence[float],
    labels: Sequence[str],
    color: str,
    title: str,
) -> None:
    """
    Draw a reverse-funnel chart showing drop-off magnitudes.

    Parameters
    ----------
    ax : matplotlib axis
        Target axis.
    data : sequence[float]
        Drop-off percentages.
    labels : sequence[str]
        Stage labels.
    color : str
        Funnel color.
    title : str
        Subplot title.
    """
    max_width = 0.8

    for i, (pct, label) in enumerate(zip(data, labels)):
        width = max_width * pct / max(data)
        left = (max_width - width) / 2

        if i < len(data) - 1:
            next_width = max_width * data[i + 1] / max(data)
            next_left = (max_width - next_width) / 2
            coords = [
                [left, -i],
                [left + width, -i],
                [next_left + next_width, -i - 1],
                [next_left, -i - 1],
            ]
        else:
            coords = [
                [left, -i],
                [left + width, -i],
                [left + width, -i - 0.8],
                [left, -i - 0.8],
            ]

        polygon = plt.Polygon(
            coords,
            closed=True,
            facecolor=color,
            alpha=0.7,
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(polygon)

        ax.text(
            0.4,
            -i - 0.4,
            f"{label}\n{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    ax.set_xlim(0, 0.8)
    ax.set_ylim(-len(data), 1)
    ax.axis("off")
    ax.set_title(title)


def plot_reverse_funnel_dropoff_comparison(
    test_dropoff: Sequence[float],
    control_dropoff: Sequence[float],
    labels: Sequence[str] = ("After Start", "After Step 1", "After Step 2", "After Step 3"),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot side-by-side reverse funnels for step-level drop-offs.

    Parameters
    ----------
    test_dropoff : sequence[float]
        Test drop-off percentages.
    control_dropoff : sequence[float]
        Control drop-off percentages.
    labels : sequence[str], default (...)
        Drop-off labels.
    save_path : str | Path | None, default None
        Optional output path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    draw_reverse_funnel(ax1, test_dropoff, labels, DEFAULT_COLORS["test"], "Test Drop-off Funnel")
    draw_reverse_funnel(ax2, control_dropoff, labels, DEFAULT_COLORS["control"], "Control Drop-off Funnel")
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# -------------------------------------------------------------------
# Saving helpers
# -------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, output_path: str | Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : str | Path
        Output file path.
    index : bool, default False
        Whether to save the index.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


# -------------------------------------------------------------------
# End-to-end pipeline
# -------------------------------------------------------------------

def run_tenure_funnel_analysis(
    config_path: str | Path,
    steps: Sequence[str] = DEFAULT_STEPS,
    broad_tenure_bins: Sequence[float] = DEFAULT_BROAD_TENURE_BINS,
    broad_tenure_labels: Sequence[str] = DEFAULT_BROAD_TENURE_LABELS,
    detail_tenure_bins: Sequence[float] = DEFAULT_DETAIL_TENURE_BINS,
    detail_tenure_labels: Sequence[str] = DEFAULT_DETAIL_TENURE_LABELS,
) -> dict:
    """
    Run the full tenure and funnel-analysis workflow.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.
    steps : sequence[str], default DEFAULT_STEPS
        Ordered funnel steps.
    broad_tenure_bins : sequence[float], default DEFAULT_BROAD_TENURE_BINS
        Coarse tenure bin edges.
    broad_tenure_labels : sequence[str], default DEFAULT_BROAD_TENURE_LABELS
        Coarse tenure labels.
    detail_tenure_bins : sequence[float], default DEFAULT_DETAIL_TENURE_BINS
        Detailed tenure bin edges.
    detail_tenure_labels : sequence[str], default DEFAULT_DETAIL_TENURE_LABELS
        Detailed tenure labels.

    Returns
    -------
    dict
        Main outputs generated by the analysis.
    """
    clients_test, tenure_control, df_web = load_analysis_inputs_from_config(config_path)
    tenure_test = clients_test.copy()

    # Basic gender split used at the top of the notebook.
    gender_subsets = split_clients_by_gender(clients_test)

    # Tenure comparison tests.
    tenure_describe = describe_tenure_distribution(tenure_test, tenure_control)
    normality_results = run_tenure_normality_tests(tenure_test, tenure_control)
    levene_results = run_tenure_levene_test(tenure_test, tenure_control)
    ttest_results = run_tenure_ttest(
        tenure_test,
        tenure_control,
        equal_var=levene_results["levene_p_value"] >= 0.05,
    )

    # Detect problematic confirmed journeys.
    complete_journeys, incomplete_confirmers = find_incomplete_confirmers(df_web, steps=steps)
    incomplete_counts_by_group = count_incomplete_confirmers_by_group(
        incomplete_confirmers,
        tenure_test,
        tenure_control,
    )

    # Remove problematic journeys and restrict to experiment clients only.
    error_free_df_web = build_error_free_web_data(
        df_web=df_web,
        incomplete_confirmers=incomplete_confirmers,
        test_df=tenure_test,
        control_df=tenure_control,
    )

    # Completion analysis.
    completed_client_ids = identify_confirmed_clients(error_free_df_web)
    contingency_table = build_completion_contingency_table(
        tenure_test,
        tenure_control,
        completed_client_ids,
    )
    completion_rates = calculate_group_completion_rates(contingency_table)
    chi_square_results = run_completion_chi_square_test(contingency_table)

    # Tenure-group completion analysis using coarse and detailed bins.
    tenure_test_broad = add_tenure_groups(
        add_completed_flag(tenure_test, completed_client_ids),
        bins=broad_tenure_bins,
        labels=broad_tenure_labels,
    )
    tenure_control_broad = add_tenure_groups(
        add_completed_flag(tenure_control, completed_client_ids),
        bins=broad_tenure_bins,
        labels=broad_tenure_labels,
    )

    broad_test_rates = calculate_completion_rate_by_tenure_group(tenure_test_broad)
    broad_control_rates = calculate_completion_rate_by_tenure_group(tenure_control_broad)

    tenure_test_detail = add_tenure_groups(
        add_completed_flag(tenure_test, completed_client_ids),
        bins=detail_tenure_bins,
        labels=detail_tenure_labels,
    )
    tenure_control_detail = add_tenure_groups(
        add_completed_flag(tenure_control, completed_client_ids),
        bins=detail_tenure_bins,
        labels=detail_tenure_labels,
    )

    detail_test_rates = calculate_completion_rate_by_tenure_group(tenure_test_detail)
    detail_control_rates = calculate_completion_rate_by_tenure_group(tenure_control_detail)

    detail_test_counts = count_clients_by_tenure_group(tenure_test_detail)
    detail_control_counts = count_clients_by_tenure_group(tenure_control_detail)

    tenure_completion_wide = build_tenure_completion_table(detail_test_rates, detail_control_rates)
    tenure_completion_long = convert_tenure_table_to_long_format(tenure_completion_wide)

    # Abandonment and reach analysis.
    abandonment_overall = calculate_abandonment_counts(error_free_df_web, steps=steps)
    abandonment_by_group = calculate_abandonment_counts_by_group(
        error_free_df_web=error_free_df_web,
        test_df=tenure_test,
        control_df=tenure_control,
        steps=steps,
    )

    step_counts_by_group = count_clients_reaching_each_step_by_group(
        df_web=df_web,
        test_df=tenure_test,
        control_df=tenure_control,
        steps=steps,
    )

    # Missing-start diagnostics.
    all_clients_without_start, no_start_df = find_clients_without_start(error_free_df_web)
    clients_without_start_by_group = find_clients_without_start_by_group(
        error_free_df_web,
        tenure_test,
        tenure_control,
    )

    return {
        "clients_test": clients_test,
        "tenure_test": tenure_test,
        "tenure_control": tenure_control,
        "df_web": df_web,
        "gender_subsets": gender_subsets,
        "tenure_describe": tenure_describe,
        "normality_results": normality_results,
        "levene_results": levene_results,
        "ttest_results": ttest_results,
        "complete_journeys": complete_journeys,
        "incomplete_confirmers": incomplete_confirmers,
        "incomplete_counts_by_group": incomplete_counts_by_group,
        "error_free_df_web": error_free_df_web,
        "completed_client_ids": completed_client_ids,
        "contingency_table": contingency_table,
        "completion_rates": completion_rates,
        "chi_square_results": chi_square_results,
        "tenure_test_broad": tenure_test_broad,
        "tenure_control_broad": tenure_control_broad,
        "broad_test_rates": broad_test_rates,
        "broad_control_rates": broad_control_rates,
        "tenure_test_detail": tenure_test_detail,
        "tenure_control_detail": tenure_control_detail,
        "detail_test_rates": detail_test_rates,
        "detail_control_rates": detail_control_rates,
        "detail_test_counts": detail_test_counts,
        "detail_control_counts": detail_control_counts,
        "tenure_completion_wide": tenure_completion_wide,
        "tenure_completion_long": tenure_completion_long,
        "abandonment_overall": abandonment_overall,
        "abandonment_by_group": abandonment_by_group,
        "step_counts_by_group": step_counts_by_group,
        "all_clients_without_start": all_clients_without_start,
        "no_start_df": no_start_df,
        "clients_without_start_by_group": clients_without_start_by_group,
    }


def main() -> None:
    """
    Example entry point for running the module as a script.

    Update the config path to match your project structure.
    """
    config_path = "../config.yaml"

    results = run_tenure_funnel_analysis(config_path=config_path)

    print("Tenure descriptive statistics:")
    print(results["tenure_describe"])
    print()

    print("Normality results:")
    print(results["normality_results"])
    print()

    print("Levene results:")
    print(results["levene_results"])
    print()

    print("T-test results:")
    print(results["ttest_results"])
    print()

    print("Incomplete confirmers by group:")
    print(results["incomplete_counts_by_group"])
    print()

    print("Overall completion rates:")
    print(results["completion_rates"])
    print()

    print("Chi-square results:")
    print(results["chi_square_results"])
    print()

    print("Detailed tenure completion table:")
    print(results["tenure_completion_wide"])
    print()

    print("Overall abandonment:")
    print(results["abandonment_overall"])
    print()

    print("Missing-start diagnostics by group:")
    for group, info in results["clients_without_start_by_group"].items():
        print(f"{group}: {info['clients_without_start']} clients without start")

    # Example plots
    plot_tenure_histogram_comparison(results["tenure_test"], results["tenure_control"])
    plot_contingency_heatmap(results["contingency_table"])
    plot_group_completion_rates(results["completion_rates"])
    plot_chi_square_pdf(results["chi_square_results"]["chi2"])
    plot_tenure_group_completion_rates(results["detail_test_rates"], results["detail_control_rates"])

    # Example funnel plots based on actual step counts.
    steps = DEFAULT_STEPS
    test_counts = results["step_counts_by_group"]["test"]
    control_counts = results["step_counts_by_group"]["control"]

    plot_client_journey_remaining(test_counts, control_counts, steps=steps)

    test_pct = [test_counts[step] / test_counts[steps[0]] * 100 for step in steps]
    control_pct = [control_counts[step] / control_counts[steps[0]] * 100 for step in steps]
    plot_funnel_comparison(test_pct, control_pct)

    test_dropoff = [
        (test_counts["start"] - test_counts["step_1"]) / test_counts["start"] * 100,
        (test_counts["step_1"] - test_counts["step_2"]) / test_counts["start"] * 100,
        (test_counts["step_2"] - test_counts["step_3"]) / test_counts["start"] * 100,
        (test_counts["step_3"] - test_counts["confirm"]) / test_counts["start"] * 100,
    ]
    control_dropoff = [
        (control_counts["start"] - control_counts["step_1"]) / control_counts["start"] * 100,
        (control_counts["step_1"] - control_counts["step_2"]) / control_counts["start"] * 100,
        (control_counts["step_2"] - control_counts["step_3"]) / control_counts["start"] * 100,
        (control_counts["step_3"] - control_counts["confirm"]) / control_counts["start"] * 100,
    ]
    plot_reverse_funnel_dropoff_comparison(test_dropoff, control_dropoff)

    # Example exports
    save_dataframe(results["error_free_df_web"], "error_free_df_web.csv")
    save_dataframe(results["tenure_completion_wide"], "tenure_completion_rates.csv")
    save_dataframe(results["tenure_completion_long"], "tenure_completion_rates_long.csv")


if __name__ == "__main__":
    main()
