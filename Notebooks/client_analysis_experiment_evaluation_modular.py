
"""
Reusable client analysis and experiment evaluation module.

Main capabilities
-----------------
1. Load raw experiment, demographic, and web-event datasets.
2. Clean and standardize the experiment-assignment table.
3. Profile online-process users versus the full client base.
4. Build demographic summaries and segment distributions.
5. Evaluate experiment balance and baseline comparability.
6. Run t-tests and chi-square tests across key variables.
7. Measure web coverage by experimental group.
8. Analyze weekly completion rates and temporal stability.
9. Fit a regression model to assess effect decay over time.
10. Generate reusable charts and export-ready summary tables.

"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yaml
from scipy.stats import chi2_contingency, ttest_ind


# -------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------

DEFAULT_NUMERIC_COLS = [
    "clnt_tenure_yr",
    "clnt_tenure_mnth",
    "clnt_age",
    "num_accts",
    "bal",
    "calls_6_mnth",
    "logons_6_mnth",
]

DEFAULT_AGE_BINS = [0, 30, 40, 50, 60, 70, 120]
DEFAULT_AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]

DEFAULT_FIGURE_COLORS = {
    "primary": "#96151D",
    "secondary": "#D97A7F",
    "control": "#D97A7F",
    "test": "#96151D",
}


# -------------------------------------------------------------------
# Config and file loading helpers
# -------------------------------------------------------------------

def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    config_path : str | Path
        Path to the config file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_csv_file(file_path: str | Path, parse_dates: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Load a CSV or TXT file into a pandas DataFrame and optionally parse dates.

    Parameters
    ----------
    file_path : str | Path
        Input file path.
    parse_dates : sequence[str] | None, default None
        Column names to convert to datetime if present.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """
    df = pd.read_csv(file_path)

    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    return df


def load_client_analysis_inputs(
    config_path: str | Path,
    experiment_raw_path: str | Path = "../data/raw/df_final_experiment_clients.txt",
    demo_key: str = "file1",
    experiment_clean_key: str = "file2",
    web_key: str = "file7",
    config_section: str = "output_data",
) -> dict[str, pd.DataFrame]:
    """
    Load the raw experiment file plus cleaned demographic, experiment, and web files.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config.
    experiment_raw_path : str | Path, default '../data/raw/df_final_experiment_clients.txt'
        Raw experiment-assignment file.
    demo_key : str, default 'file1'
        Key for the cleaned demographic file in the config.
    experiment_clean_key : str, default 'file2'
        Key for the cleaned experiment file in the config.
    web_key : str, default 'file7'
        Key for the cleaned web-event file in the config.
    config_section : str, default 'output_data'
        Config section name.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with loaded input tables.
    """
    config = load_yaml_config(config_path)

    return {
        "df_exp_raw": load_csv_file(experiment_raw_path),
        "demo_df_cleaned": load_csv_file(config[config_section][demo_key]),
        "df_ec": load_csv_file(config[config_section][experiment_clean_key]),
        "df_web": load_csv_file(config[config_section][web_key], parse_dates=["date_time"]),
    }


# -------------------------------------------------------------------
# Generic utility helpers
# -------------------------------------------------------------------

def _ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent directory of an output file if it does not exist.

    Parameters
    ----------
    path : str | Path
        Output path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, output_path: str | Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save.
    output_path : str | Path
        Destination file path.
    index : bool, default False
        Whether to save the index.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


# -------------------------------------------------------------------
# Experiment table cleaning
# -------------------------------------------------------------------

def clean_experiment_assignment_table(
    df_exp: pd.DataFrame,
    client_col: str = "client_id",
    variation_col: str = "Variation",
) -> pd.DataFrame:
    """
    Clean the raw experiment-assignment table.

    The cleaning steps reproduce the notebook logic:
    - remove rows with missing client ID or variation
    - remove duplicated clients
    - strip whitespace and lowercase the variation label
    - cast client_id to integer

    Parameters
    ----------
    df_exp : pd.DataFrame
        Raw experiment-assignment table.
    client_col : str, default 'client_id'
        Client ID column.
    variation_col : str, default 'Variation'
        Variation label column.

    Returns
    -------
    pd.DataFrame
        Cleaned experiment table.
    """
    clean_df = df_exp.dropna(subset=[client_col, variation_col]).copy()
    clean_df = clean_df.drop_duplicates(subset=[client_col]).copy()
    clean_df[variation_col] = clean_df[variation_col].astype(str).str.strip().str.lower()
    clean_df[client_col] = clean_df[client_col].astype(int)
    return clean_df


def summarize_experiment_assignment_quality(
    df_exp: pd.DataFrame,
    variation_col: str = "Variation",
) -> dict:
    """
    Summarize missingness, duplicates, and variation distribution.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Experiment-assignment table.
    variation_col : str, default 'Variation'
        Variation label column.

    Returns
    -------
    dict
        Quality checks and variation counts.
    """
    return {
        "missing_values": df_exp.isna().sum(),
        "duplicate_rows": int(df_exp.duplicated().sum()),
        "variation_unique_values": df_exp[variation_col].unique().tolist(),
        "variation_counts": df_exp[variation_col].value_counts(),
        "variation_percent": df_exp[variation_col].value_counts(normalize=True) * 100,
    }


# -------------------------------------------------------------------
# Online user profiling
# -------------------------------------------------------------------

def get_online_client_ids(
    df_web: pd.DataFrame,
    client_col: str = "client_id",
) -> np.ndarray:
    """
    Extract unique client IDs that appear in the online process data.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event table.
    client_col : str, default 'client_id'
        Client ID column.

    Returns
    -------
    np.ndarray
        Unique online client IDs.
    """
    return df_web[client_col].unique()


def filter_online_clients(
    demo_df: pd.DataFrame,
    online_client_ids: Sequence,
    client_col: str = "client_id",
) -> pd.DataFrame:
    """
    Keep only demographic rows belonging to clients who used the online process.

    Parameters
    ----------
    demo_df : pd.DataFrame
        Demographic table.
    online_client_ids : sequence
        Client IDs found in the web-event table.
    client_col : str, default 'client_id'
        Client ID column.

    Returns
    -------
    pd.DataFrame
        Demographic subset for online users.
    """
    return demo_df[demo_df[client_col].isin(online_client_ids)].copy()


def summarize_numeric_profiles(
    all_clients_demo: pd.DataFrame,
    online_clients_demo: pd.DataFrame,
    numeric_cols: Sequence[str] = DEFAULT_NUMERIC_COLS,
) -> dict[str, pd.DataFrame]:
    """
    Produce descriptive summaries for all clients and online clients.

    Parameters
    ----------
    all_clients_demo : pd.DataFrame
        Full demographic table.
    online_clients_demo : pd.DataFrame
        Online-user demographic subset.
    numeric_cols : sequence[str], default DEFAULT_NUMERIC_COLS
        Numeric columns to summarize.

    Returns
    -------
    dict[str, pd.DataFrame]
        Descriptive summaries and mean-comparison tables.
    """
    all_numeric_summary = all_clients_demo[list(numeric_cols)].describe()
    online_numeric_summary = online_clients_demo[list(numeric_cols)].describe()

    comparison_means = pd.DataFrame({
        "all_clients_mean": all_clients_demo[list(numeric_cols)].mean(),
        "online_clients_mean": online_clients_demo[list(numeric_cols)].mean(),
    })
    comparison_means["difference"] = comparison_means["online_clients_mean"] - comparison_means["all_clients_mean"]
    comparison_means["pct_difference"] = (
        comparison_means["difference"] / comparison_means["all_clients_mean"] * 100
    )

    summary_table = pd.DataFrame({
        "metric": list(numeric_cols),
        "all_clients_mean": [all_clients_demo[col].mean() for col in numeric_cols],
        "online_clients_mean": [online_clients_demo[col].mean() for col in numeric_cols],
    })
    summary_table["difference"] = summary_table["online_clients_mean"] - summary_table["all_clients_mean"]
    summary_table["pct_difference"] = (
        summary_table["difference"] / summary_table["all_clients_mean"] * 100
    )

    return {
        "all_numeric_summary": all_numeric_summary,
        "online_numeric_summary": online_numeric_summary,
        "comparison_means": comparison_means.sort_values(by="pct_difference", ascending=False),
        "summary_table": summary_table.sort_values(by="pct_difference", ascending=False),
    }


def build_client_profile_with_online_flag(
    demo_df: pd.DataFrame,
    online_client_ids: Sequence,
    client_col: str = "client_id",
    output_col: str = "used_online_process",
) -> pd.DataFrame:
    """
    Add a boolean flag indicating whether each client used the online process.

    Parameters
    ----------
    demo_df : pd.DataFrame
        Demographic table.
    online_client_ids : sequence
        Client IDs found online.
    client_col : str, default 'client_id'
        Client ID column.
    output_col : str, default 'used_online_process'
        Output flag name.

    Returns
    -------
    pd.DataFrame
        Demographic table with online-usage flag.
    """
    client_profile = demo_df.copy()
    client_profile[output_col] = client_profile[client_col].isin(online_client_ids)
    return client_profile


def compare_online_vs_offline_means(
    client_profile: pd.DataFrame,
    numeric_cols: Sequence[str] = DEFAULT_NUMERIC_COLS,
    online_flag_col: str = "used_online_process",
) -> pd.DataFrame:
    """
    Compare mean client characteristics for online users versus non-users.

    Parameters
    ----------
    client_profile : pd.DataFrame
        Demographic table with an online-usage flag.
    numeric_cols : sequence[str], default DEFAULT_NUMERIC_COLS
        Numeric columns to compare.
    online_flag_col : str, default 'used_online_process'
        Online-usage flag column.

    Returns
    -------
    pd.DataFrame
        Mean profile comparison table.
    """
    grouped_profile = client_profile.groupby(online_flag_col)[list(numeric_cols)].mean().T
    grouped_profile.columns = ["did_not_use_online", "used_online"]
    grouped_profile["difference"] = grouped_profile["used_online"] - grouped_profile["did_not_use_online"]
    return grouped_profile


def add_age_groups(
    client_profile: pd.DataFrame,
    age_col: str = "clnt_age",
    bins: Sequence[float] = DEFAULT_AGE_BINS,
    labels: Sequence[str] = DEFAULT_AGE_LABELS,
    output_col: str = "age_group",
) -> pd.DataFrame:
    """
    Add age-group categories to the client profile table.

    Parameters
    ----------
    client_profile : pd.DataFrame
        Client table.
    age_col : str, default 'clnt_age'
        Age column.
    bins : sequence[float], default DEFAULT_AGE_BINS
        Bin edges.
    labels : sequence[str], default DEFAULT_AGE_LABELS
        Bin labels.
    output_col : str, default 'age_group'
        Output column name.

    Returns
    -------
    pd.DataFrame
        Table with age groups.
    """
    result = client_profile.copy()
    result[output_col] = pd.cut(result[age_col], bins=bins, labels=labels)
    return result


def build_age_distribution_comparison(
    client_profile: pd.DataFrame,
    online_flag_col: str = "used_online_process",
    age_group_col: str = "age_group",
) -> pd.DataFrame:
    """
    Compare age-group distributions for all clients versus online users.

    Parameters
    ----------
    client_profile : pd.DataFrame
        Client profile with age groups and online-usage flag.
    online_flag_col : str, default 'used_online_process'
        Online-usage flag column.
    age_group_col : str, default 'age_group'
        Age-group column.

    Returns
    -------
    pd.DataFrame
        Distribution comparison table.
    """
    age_dist_online = (
        client_profile[client_profile[online_flag_col]][age_group_col]
        .value_counts(normalize=True)
        .sort_index()
    )
    age_dist_all = (
        client_profile[age_group_col]
        .value_counts(normalize=True)
        .sort_index()
    )

    age_comparison = pd.concat(
        [
            age_dist_all.rename("all_clients_pct"),
            age_dist_online.rename("online_clients_pct"),
        ],
        axis=1,
    )
    return age_comparison


def add_client_segment_by_tenure(
    client_profile: pd.DataFrame,
    tenure_col: str = "clnt_tenure_yr",
    output_col: str = "client_segment",
    threshold_years: float = 5,
) -> pd.DataFrame:
    """
    Classify clients as new or long-standing based on tenure.

    Parameters
    ----------
    client_profile : pd.DataFrame
        Client profile table.
    tenure_col : str, default 'clnt_tenure_yr'
        Tenure column.
    output_col : str, default 'client_segment'
        Output segment column.
    threshold_years : float, default 5
        Cutoff separating new from long-standing clients.

    Returns
    -------
    pd.DataFrame
        Profile table with tenure segment.
    """
    result = client_profile.copy()
    result[output_col] = result[tenure_col].apply(
        lambda x: "New" if x < threshold_years else "Long-standing"
    )
    return result


def summarize_client_segments(
    client_profile: pd.DataFrame,
    segment_col: str = "client_segment",
) -> dict:
    """
    Build counts and percentages for client tenure segments.

    Parameters
    ----------
    client_profile : pd.DataFrame
        Profile table with segment column.
    segment_col : str, default 'client_segment'
        Segment column.

    Returns
    -------
    dict
        Counts and percentage distributions.
    """
    return {
        "segment_counts": client_profile[segment_col].value_counts(),
        "segment_pct": client_profile[segment_col].value_counts(normalize=True),
    }


# -------------------------------------------------------------------
# Experiment balance and randomization analysis
# -------------------------------------------------------------------

def merge_experiment_with_demographics(
    experiment_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    client_col: str = "client_id",
) -> pd.DataFrame:
    """
    Merge experiment assignment with client demographics.

    Parameters
    ----------
    experiment_df : pd.DataFrame
        Experiment-assignment table.
    demo_df : pd.DataFrame
        Demographic table.
    client_col : str, default 'client_id'
        Merge key.

    Returns
    -------
    pd.DataFrame
        Merged experiment-demographic table.
    """
    return experiment_df.merge(demo_df, on=client_col, how="left")


def summarize_group_balance(
    df_merged: pd.DataFrame,
    variation_col: str = "Variation",
) -> dict:
    """
    Summarize experiment group sizes and proportions.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Experiment-demographic merged table.
    variation_col : str, default 'Variation'
        Variation column.

    Returns
    -------
    dict
        Counts and percentages by variation.
    """
    return {
        "group_counts": df_merged[variation_col].value_counts(),
        "group_pct": df_merged[variation_col].value_counts(normalize=True),
    }


def summarize_missingness_by_group(
    df_merged: pd.DataFrame,
    numeric_cols: Sequence[str] = DEFAULT_NUMERIC_COLS,
    variation_col: str = "Variation",
) -> pd.DataFrame:
    """
    Measure missing-value rates by group for selected numeric columns.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged experiment table.
    numeric_cols : sequence[str], default DEFAULT_NUMERIC_COLS
        Numeric columns to inspect.
    variation_col : str, default 'Variation'
        Variation column.

    Returns
    -------
    pd.DataFrame
        Missing-rate table by variation and variable.
    """
    rows = []
    for col in numeric_cols:
        tmp = df_merged.groupby(variation_col)[col].apply(lambda x: x.isna().mean())
        for variation, missing_rate in tmp.items():
            rows.append({"variable": col, "Variation": variation, "missing_rate": missing_rate})
    return pd.DataFrame(rows)


def summarize_global_missingness(df_merged: pd.DataFrame) -> pd.Series:
    """
    Compute overall missing-value rates across the merged experiment table.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged experiment table.

    Returns
    -------
    pd.Series
        Missing-value rate by column.
    """
    return df_merged.isna().mean().sort_values(ascending=False)


def check_experiment_coverage_in_demographics(
    df_ec: pd.DataFrame,
    demo_df: pd.DataFrame,
    client_col: str = "client_id",
    variation_col: str = "Variation",
) -> pd.Series:
    """
    Check what share of experiment clients appears in the demographic table.

    Parameters
    ----------
    df_ec : pd.DataFrame
        Experiment-assignment table.
    demo_df : pd.DataFrame
        Demographic table.
    client_col : str, default 'client_id'
        Client ID column.
    variation_col : str, default 'Variation'
        Variation column.

    Returns
    -------
    pd.Series
        Coverage rate by variation.
    """
    tmp = df_ec.copy()
    tmp["in_demo"] = tmp[client_col].isin(demo_df[client_col])
    return tmp.groupby(variation_col)["in_demo"].mean()


def run_numeric_balance_ttests(
    df_merged: pd.DataFrame,
    numeric_cols: Sequence[str] = DEFAULT_NUMERIC_COLS,
    variation_col: str = "Variation",
    control_label: str = "control",
    test_label: str = "test",
) -> pd.DataFrame:
    """
    Run independent-samples t-tests for numeric baseline variables.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged experiment table.
    numeric_cols : sequence[str], default DEFAULT_NUMERIC_COLS
        Numeric columns to test.
    variation_col : str, default 'Variation'
        Variation column.
    control_label : str, default 'control'
        Control label.
    test_label : str, default 'test'
        Test label.

    Returns
    -------
    pd.DataFrame
        T-test results by variable.
    """
    results = []

    for col in numeric_cols:
        control = df_merged[df_merged[variation_col] == control_label][col].dropna()
        test = df_merged[df_merged[variation_col] == test_label][col].dropna()

        if len(control) == 0 or len(test) == 0:
            results.append({
                "variable": col,
                "t_stat": np.nan,
                "p_value": np.nan,
                "status": "skipped_empty_group",
            })
            continue

        stat, p = ttest_ind(control, test)
        results.append({
            "variable": col,
            "t_stat": stat,
            "p_value": p,
            "status": "ok",
        })

    return pd.DataFrame(results).sort_values("p_value", na_position="last")


def run_categorical_balance_chi_square(
    df_merged: pd.DataFrame,
    category_col: str,
    variation_col: str = "Variation",
) -> dict:
    """
    Run a chi-square test for a categorical baseline variable by variation.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged experiment table.
    category_col : str
        Categorical variable to test.
    variation_col : str, default 'Variation'
        Variation column.

    Returns
    -------
    dict
        Contingency table and chi-square test output.
    """
    contingency = pd.crosstab(df_merged[variation_col], df_merged[category_col])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    return {
        "contingency": contingency,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "expected": expected,
    }


def check_web_coverage_by_group(
    df_merged: pd.DataFrame,
    df_web: pd.DataFrame,
    client_col: str = "client_id",
    variation_col: str = "Variation",
) -> pd.Series:
    """
    Check whether test and control clients appear equally in web data.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged experiment table.
    df_web : pd.DataFrame
        Web-event table.
    client_col : str, default 'client_id'
        Client ID column.
    variation_col : str, default 'Variation'
        Variation column.

    Returns
    -------
    pd.Series
        Share appearing in web data by variation.
    """
    web_clients = df_web[client_col].unique()
    tmp = df_merged.copy()
    tmp["appeared_in_web"] = tmp[client_col].isin(web_clients)
    return tmp.groupby(variation_col)["appeared_in_web"].mean()


# -------------------------------------------------------------------
# Weekly completion and duration assessment
# -------------------------------------------------------------------

def build_visit_completion_table(
    df_web: pd.DataFrame,
    df_ec: pd.DataFrame,
    visit_col: str = "visit_id",
    step_col: str = "process_step",
    confirm_step: str = "confirm",
    client_col: str = "client_id",
    datetime_col: str = "date_time",
    variation_col: str = "Variation",
) -> pd.DataFrame:
    """
    Build a visit-level completion table with variation and weekly buckets.

    Parameters
    ----------
    df_web : pd.DataFrame
        Web-event table.
    df_ec : pd.DataFrame
        Experiment-assignment table.
    visit_col : str, default 'visit_id'
        Visit ID column.
    step_col : str, default 'process_step'
        Process-step column.
    confirm_step : str, default 'confirm'
        Completion step.
    client_col : str, default 'client_id'
        Client ID column.
    datetime_col : str, default 'date_time'
        Datetime column.
    variation_col : str, default 'Variation'
        Variation column.

    Returns
    -------
    pd.DataFrame
        Visit-level completion table with variation and weekly fields.
    """
    web_df = df_web.copy()
    web_df[datetime_col] = pd.to_datetime(web_df[datetime_col])
    web_df["week"] = web_df[datetime_col].dt.to_period("W").astype(str)

    df_completion = (
        web_df.groupby(visit_col)[step_col]
        .apply(lambda x: int(confirm_step in x.values))
        .reset_index(name="completed")
    )

    visit_meta = (
        web_df.groupby(visit_col)
        .agg({
            client_col: "first",
            "week": "first",
        })
        .reset_index()
    )

    df_completion = df_completion.merge(visit_meta, on=visit_col, how="left")
    df_completion = df_completion.merge(
        df_ec[[client_col, variation_col]],
        on=client_col,
        how="left",
    )
    return df_completion


def calculate_weekly_completion_rates(
    df_completion: pd.DataFrame,
    week_col: str = "week",
    variation_col: str = "Variation",
    completed_col: str = "completed",
) -> pd.DataFrame:
    """
    Calculate weekly completion rates by variation.

    Parameters
    ----------
    df_completion : pd.DataFrame
        Visit-level completion table.
    week_col : str, default 'week'
        Weekly bucket column.
    variation_col : str, default 'Variation'
        Variation column.
    completed_col : str, default 'completed'
        Completion flag.

    Returns
    -------
    pd.DataFrame
        Weekly completion rates.
    """
    return (
        df_completion
        .groupby([week_col, variation_col])[completed_col]
        .mean()
        .reset_index()
        .sort_values(week_col)
    )


def build_weekly_effect_table(
    df_completion: pd.DataFrame,
    week_col: str = "week",
    variation_col: str = "Variation",
    completed_col: str = "completed",
) -> pd.DataFrame:
    """
    Build a weekly effect table where effect = test completion rate - control rate.

    Parameters
    ----------
    df_completion : pd.DataFrame
        Visit-level completion table.
    week_col : str, default 'week'
        Weekly bucket column.
    variation_col : str, default 'Variation'
        Variation column.
    completed_col : str, default 'completed'
        Completion flag.

    Returns
    -------
    pd.DataFrame
        Weekly effect table with a sequential time index.
    """
    weekly_effect = (
        df_completion
        .groupby([week_col, variation_col])[completed_col]
        .mean()
        .unstack()
    )
    weekly_effect["effect"] = weekly_effect["test"] - weekly_effect["control"]
    weekly_effect = weekly_effect.reset_index()
    weekly_effect["time_index"] = range(len(weekly_effect))
    return weekly_effect


def run_weekly_chi_square_tests(
    df_completion: pd.DataFrame,
    week_col: str = "week",
    variation_col: str = "Variation",
    completed_col: str = "completed",
) -> pd.DataFrame:
    """
    Run a chi-square test independently for each week.

    Parameters
    ----------
    df_completion : pd.DataFrame
        Visit-level completion table.
    week_col : str, default 'week'
        Weekly bucket column.
    variation_col : str, default 'Variation'
        Variation column.
    completed_col : str, default 'completed'
        Completion flag.

    Returns
    -------
    pd.DataFrame
        Weekly p-values and chi-square statistics.
    """
    tmp = df_completion.copy()
    tmp["week_start"] = pd.to_datetime(tmp[week_col].str.split("/").str[0])

    results = []
    for week in sorted(tmp["week_start"].dropna().unique()):
        subset = tmp[tmp["week_start"] == week]
        table = pd.crosstab(subset[variation_col], subset[completed_col])

        if table.shape[0] < 2 or table.shape[1] < 2:
            results.append({"date": pd.Timestamp(week), "chi2": np.nan, "p_value": np.nan})
            continue

        chi2, p_value, _, _ = chi2_contingency(table)
        results.append({"date": pd.Timestamp(week), "chi2": chi2, "p_value": p_value})

    return pd.DataFrame(results).sort_values("date")


def fit_effect_decay_regression(
    weekly_effect: pd.DataFrame,
    time_col: str = "time_index",
    effect_col: str = "effect",
) -> dict:
    """
    Fit an OLS regression of weekly effect on time index.

    Parameters
    ----------
    weekly_effect : pd.DataFrame
        Weekly effect table.
    time_col : str, default 'time_index'
        Time-index column.
    effect_col : str, default 'effect'
        Effect column.

    Returns
    -------
    dict
        Regression inputs and fitted model.
    """
    X = sm.add_constant(weekly_effect[time_col])
    y = weekly_effect[effect_col]
    model = sm.OLS(y, X).fit()
    return {"X": X, "y": y, "model": model}


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def plot_age_group_distribution(
    client_profile: pd.DataFrame,
    age_group_col: str = "age_group",
    online_flag_col: str = "used_online_process",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the count of clients by age group and online-usage status.

    Parameters
    ----------
    client_profile : pd.DataFrame
        Profile table with age groups and online-usage flag.
    age_group_col : str, default 'age_group'
        Age-group column.
    online_flag_col : str, default 'used_online_process'
        Online-usage flag.
    save_path : str | Path | None, default None
        Optional output path.
    """
    age_group_df = (
        client_profile
        .groupby([age_group_col, online_flag_col], observed=False)
        .size()
        .reset_index(name="count")
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=age_group_df,
        x=age_group_col,
        y="count",
        hue=online_flag_col,
        palette=[DEFAULT_FIGURE_COLORS["secondary"], DEFAULT_FIGURE_COLORS["primary"]],
    )
    plt.title("Age Group Distribution: Online vs Offline Clients")
    plt.xlabel("Age Group")
    plt.ylabel("Client Count")
    plt.legend(title="Used Online Process", labels=["Offline", "Online"])
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_online_offline_share(
    client_profile: pd.DataFrame,
    online_flag_col: str = "used_online_process",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the percentage share of online versus offline clients.

    Parameters
    ----------
    client_profile : pd.DataFrame
        Profile table with online-usage flag.
    online_flag_col : str, default 'used_online_process'
        Online-usage flag column.
    save_path : str | Path | None, default None
        Optional output path.
    """
    usage_counts = client_profile[online_flag_col].value_counts(normalize=True).reindex([True, False])

    plt.figure(figsize=(6, 4))
    usage_counts.plot(
        kind="bar",
        color=[DEFAULT_FIGURE_COLORS["primary"], DEFAULT_FIGURE_COLORS["secondary"]],
        edgecolor="black",
    )
    plt.title("Share of Clients Using the Online Process")
    plt.xlabel("Client Type")
    plt.ylabel("Percentage")
    plt.xticks(ticks=[0, 1], labels=["Online", "Offline"], rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_client_segment_pie(
    segment_pct: pd.Series,
    colors: Sequence[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the client composition by new versus long-standing segments.

    Parameters
    ----------
    segment_pct : pd.Series
        Segment percentage distribution.
    colors : sequence[str] | None, default None
        Pie colors.
    save_path : str | Path | None, default None
        Optional output path.
    """
    if colors is None:
        colors = [DEFAULT_FIGURE_COLORS["primary"], DEFAULT_FIGURE_COLORS["secondary"]]

    plt.figure()
    segment_pct.plot(
        kind="pie",
        autopct="%1.1f%%",
        colors=list(colors),
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    plt.title("Client Composition: New vs Long-standing")
    plt.ylabel("")
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_ttest_pvalues(
    ttest_results: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot t-test p-values across numeric variables using a log-scale x-axis.

    Parameters
    ----------
    ttest_results : pd.DataFrame
        Output of `run_numeric_balance_ttests`.
    save_path : str | Path | None, default None
        Optional output path.
    """
    plot_df = ttest_results.dropna(subset=["p_value"]).sort_values("p_value")

    plt.figure(figsize=(10, 5))
    plt.barh(plot_df["variable"], plot_df["p_value"], color=DEFAULT_FIGURE_COLORS["primary"])
    plt.axvline(x=0.05, linestyle="--", color="black", label="0.05 threshold")
    plt.xscale("log")
    plt.xlabel("p-value (log scale)")
    plt.title("T-test Results Across Variables")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_weekly_pvalues(
    weekly_chi_square_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot weekly chi-square p-values over time.

    Parameters
    ----------
    weekly_chi_square_df : pd.DataFrame
        Weekly p-value table.
    save_path : str | Path | None, default None
        Optional output path.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(
        weekly_chi_square_df["date"],
        weekly_chi_square_df["p_value"],
        color=DEFAULT_FIGURE_COLORS["primary"],
        alpha=0.8,
    )
    plt.axhline(
        y=0.05,
        linestyle="--",
        color="black",
        label="Significance threshold (0.05)",
    )
    plt.yscale("log")
    plt.title("P-values Over Time")
    plt.xlabel("Date")
    plt.ylabel("p-value (log scale)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_effect_over_time(
    weekly_effect: pd.DataFrame,
    regression_output: dict,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot weekly treatment effect with fitted regression trend.

    Parameters
    ----------
    weekly_effect : pd.DataFrame
        Weekly effect table.
    regression_output : dict
        Output of `fit_effect_decay_regression`.
    save_path : str | Path | None, default None
        Optional output path.
    """
    plt.figure()
    plt.scatter(
        weekly_effect["time_index"],
        weekly_effect["effect"],
        color=DEFAULT_FIGURE_COLORS["primary"],
        alpha=0.7,
    )
    plt.plot(
        weekly_effect["time_index"],
        regression_output["model"].predict(regression_output["X"]),
        linestyle="--",
        color="black",
    )
    plt.axhline(0, linestyle=":", color="gray")
    plt.title("Effect Over Time with Regression Trend")
    plt.xlabel("Time Index")
    plt.ylabel("Effect")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# -------------------------------------------------------------------
# Text conclusion helpers
# -------------------------------------------------------------------

def build_experiment_balance_conclusion(
    ttest_results: pd.DataFrame,
    gender_p_value: float,
    alpha: float = 0.05,
) -> str:
    """
    Build a reusable conclusion about experiment balance and randomization.

    Parameters
    ----------
    ttest_results : pd.DataFrame
        Numeric t-test results.
    gender_p_value : float
        Gender chi-square p-value.
    alpha : float, default 0.05
        Significance threshold.

    Returns
    -------
    str
        Written conclusion.
    """
    significant_vars = ttest_results.loc[
        (ttest_results["p_value"] < alpha) & (ttest_results["status"] == "ok"),
        "variable",
    ].tolist()

    if significant_vars or gender_p_value < alpha:
        return (
            "The experiment appears balanced in size, but the baseline comparability is weak. "
            "Statistical tests indicate significant differences between control and test in one or more "
            "pre-existing client characteristics, which suggests the randomization was not fully effective. "
            "As a result, part of the observed experimental outcome may be driven by baseline client differences "
            "rather than by the design change alone."
        )

    return (
        "The experiment appears balanced in both group size and baseline characteristics. "
        "Statistical testing does not indicate meaningful pre-existing differences between control and test, "
        "which supports the validity of the randomization process."
    )


def build_duration_assessment_conclusion(
    weekly_chi_square_df: pd.DataFrame,
    regression_output: dict,
    alpha: float = 0.05,
) -> str:
    """
    Build a reusable conclusion about temporal stability of the treatment effect.

    Parameters
    ----------
    weekly_chi_square_df : pd.DataFrame
        Weekly chi-square results.
    regression_output : dict
        Regression output from `fit_effect_decay_regression`.
    alpha : float, default 0.05
        Significance threshold.

    Returns
    -------
    str
        Written duration assessment.
    """
    n_significant = int((weekly_chi_square_df["p_value"] < alpha).sum())
    slope = regression_output["model"].params.get("time_index", np.nan)
    slope_p = regression_output["model"].pvalues.get("time_index", np.nan)

    if slope < 0 and slope_p < alpha:
        decay_text = (
            "The regression trend indicates a statistically significant decay in the treatment effect over time."
        )
    elif slope < 0:
        decay_text = (
            "The regression trend is negative, suggesting possible decay, but the evidence is not statistically strong."
        )
    else:
        decay_text = (
            "The regression trend does not indicate a clear decay pattern over time."
        )

    return (
        f"The experiment timeframe was long enough to observe the treatment effect across multiple weeks. "
        f"However, the weekly chi-square analysis shows that the effect was only significant in {n_significant} weeks, "
        f"which indicates temporal instability. {decay_text} "
        f"This suggests that aggregate results may overstate the long-term impact if the early gains were not sustained."
    )


# -------------------------------------------------------------------
# End-to-end pipeline
# -------------------------------------------------------------------

def run_client_analysis_experiment_evaluation(
    config_path: str | Path,
    experiment_raw_path: str | Path = "../data/raw/df_final_experiment_clients.txt",
    numeric_cols: Sequence[str] = DEFAULT_NUMERIC_COLS,
) -> dict:
    """
    Run the full client analysis and experiment evaluation workflow.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.
    experiment_raw_path : str | Path, default '../data/raw/df_final_experiment_clients.txt'
        Raw experiment file path.
    numeric_cols : sequence[str], default DEFAULT_NUMERIC_COLS
        Numeric columns used in the profile and balance analysis.

    Returns
    -------
    dict
        Main outputs generated by the workflow.
    """
    inputs = load_client_analysis_inputs(
        config_path=config_path,
        experiment_raw_path=experiment_raw_path,
    )

    df_exp_raw = inputs["df_exp_raw"]
    demo_df_cleaned = inputs["demo_df_cleaned"]
    df_ec = inputs["df_ec"]
    df_web = inputs["df_web"]

    # Clean raw experiment assignment.
    df_exp_clean = clean_experiment_assignment_table(df_exp_raw)
    experiment_quality = summarize_experiment_assignment_quality(df_exp_clean)

    # Profile online users.
    online_client_ids = get_online_client_ids(df_web)
    online_clients_demo = filter_online_clients(demo_df_cleaned, online_client_ids)
    numeric_summaries = summarize_numeric_profiles(
        all_clients_demo=demo_df_cleaned,
        online_clients_demo=online_clients_demo,
        numeric_cols=numeric_cols,
    )

    client_profile = build_client_profile_with_online_flag(demo_df_cleaned, online_client_ids)
    grouped_profile = compare_online_vs_offline_means(client_profile, numeric_cols=numeric_cols)

    client_profile = add_age_groups(client_profile)
    age_comparison = build_age_distribution_comparison(client_profile)

    client_profile = add_client_segment_by_tenure(client_profile)
    segment_summary = summarize_client_segments(client_profile)

    # Evaluate experiment structure and baseline comparability.
    df_merged = merge_experiment_with_demographics(df_ec, demo_df_cleaned)
    group_balance = summarize_group_balance(df_merged)
    missingness_by_group = summarize_missingness_by_group(df_merged, numeric_cols=numeric_cols)
    global_missingness = summarize_global_missingness(df_merged)
    demographic_coverage = check_experiment_coverage_in_demographics(df_ec, demo_df_cleaned)
    ttest_results = run_numeric_balance_ttests(df_merged, numeric_cols=numeric_cols)
    gender_test = run_categorical_balance_chi_square(df_merged, category_col="gendr")
    web_coverage = check_web_coverage_by_group(df_merged, df_web)

    experiment_balance_conclusion = build_experiment_balance_conclusion(
        ttest_results=ttest_results,
        gender_p_value=gender_test["p_value"],
    )

    # Duration assessment.
    df_completion = build_visit_completion_table(df_web, df_ec)
    weekly_rates = calculate_weekly_completion_rates(df_completion)
    weekly_effect = build_weekly_effect_table(df_completion)
    weekly_chi_square_df = run_weekly_chi_square_tests(df_completion)
    regression_output = fit_effect_decay_regression(weekly_effect)

    duration_assessment_conclusion = build_duration_assessment_conclusion(
        weekly_chi_square_df=weekly_chi_square_df,
        regression_output=regression_output,
    )

    return {
        "df_exp_raw": df_exp_raw,
        "df_exp_clean": df_exp_clean,
        "experiment_quality": experiment_quality,
        "demo_df_cleaned": demo_df_cleaned,
        "df_ec": df_ec,
        "df_web": df_web,
        "online_client_ids": online_client_ids,
        "online_clients_demo": online_clients_demo,
        "numeric_summaries": numeric_summaries,
        "client_profile": client_profile,
        "grouped_profile": grouped_profile,
        "age_comparison": age_comparison,
        "segment_summary": segment_summary,
        "df_merged": df_merged,
        "group_balance": group_balance,
        "missingness_by_group": missingness_by_group,
        "global_missingness": global_missingness,
        "demographic_coverage": demographic_coverage,
        "ttest_results": ttest_results,
        "gender_test": gender_test,
        "web_coverage": web_coverage,
        "experiment_balance_conclusion": experiment_balance_conclusion,
        "df_completion": df_completion,
        "weekly_rates": weekly_rates,
        "weekly_effect": weekly_effect,
        "weekly_chi_square_df": weekly_chi_square_df,
        "regression_output": regression_output,
        "duration_assessment_conclusion": duration_assessment_conclusion,
    }


def main() -> None:
    """
    Example entry point for running the module as a script.

    Update paths if needed to match your project structure.
    """
    config_path = "../config.yaml"

    results = run_client_analysis_experiment_evaluation(config_path=config_path)

    print("Experiment table quality checks:")
    print(results["experiment_quality"]["variation_counts"])
    print()

    print("Online vs all-clients mean comparison:")
    print(results["numeric_summaries"]["comparison_means"])
    print()

    print("Group balance:")
    print(results["group_balance"]["group_counts"])
    print()

    print("T-test results:")
    print(results["ttest_results"])
    print()

    print("Gender chi-square p-value:", results["gender_test"]["p_value"])
    print()

    print("Weekly chi-square results:")
    print(results["weekly_chi_square_df"])
    print()

    print("Regression summary:")
    print(results["regression_output"]["model"].summary())
    print()

    print("Experiment balance conclusion:")
    print(results["experiment_balance_conclusion"])
    print()

    print("Duration assessment conclusion:")
    print(results["duration_assessment_conclusion"])

    # Example figure outputs.
    plot_age_group_distribution(results["client_profile"])
    plot_online_offline_share(results["client_profile"])
    plot_client_segment_pie(results["segment_summary"]["segment_pct"])
    plot_ttest_pvalues(results["ttest_results"])
    plot_weekly_pvalues(results["weekly_chi_square_df"])
    plot_effect_over_time(results["weekly_effect"], results["regression_output"])

    # Example exports.
    save_dataframe(results["df_exp_clean"], "df_final_experiment_clients_clean.csv")
    save_dataframe(results["numeric_summaries"]["summary_table"], "online_vs_all_clients_summary.csv")
    save_dataframe(results["ttest_results"], "experiment_balance_ttests.csv")
    save_dataframe(results["weekly_chi_square_df"], "weekly_chi_square_results.csv")


if __name__ == "__main__":
    main()
