
"""
Reusable A/B test funnel analysis pipeline.

This module converts the notebook logic into reusable functions so the same
workflow can be applied to other experiments with minimal changes.

Main capabilities
-----------------
1. Load and combine multiple web-event files.
2. Clean event sequences by removing consecutive duplicated steps.
3. Build funnel metrics per visit/client.
4. Compare control vs test conversion rates with a two-proportion z-test.
5. Compute transition durations between consecutive funnel steps.
6. Filter transition outliers using configurable limits.
7. Run Welch's t-tests on transition durations.
8. Create reusable plots for completion rates and step durations.

"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import yaml


# -------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------

# Default funnel order used across the analysis.
DEFAULT_STEPS = ["start", "step_1", "step_2", "step_3", "confirm"]

# Default comparison groups.
DEFAULT_VARIATION_ORDER = ["control", "test"]

# Default colors used in the plots.
DEFAULT_COLORS = {
    "control": "#333333",
    "test": "#96151D",
}

# Default transition outlier limits in seconds.
DEFAULT_TRANSITION_LIMITS = {
    "start -> step_1": 120,
    "step_1 -> step_2": 90,
    "step_2 -> step_3": 240,
    "step_3 -> confirm": 320,
}


# -------------------------------------------------------------------
# File loading helpers
# -------------------------------------------------------------------

def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration content.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_event_files(file_paths: Sequence[str | Path], parse_dates: bool = True) -> pd.DataFrame:
    """
    Load multiple event files and concatenate them into a single DataFrame.

    This replaces the hardcoded notebook logic where each file was loaded
    manually into a separate variable.

    Parameters
    ----------
    file_paths : sequence of str or Path
        Paths to the raw event files.
    parse_dates : bool, default True
        If True, convert the 'date_time' column to pandas datetime.

    Returns
    -------
    pd.DataFrame
        Combined event dataset.
    """
    frames = [pd.read_csv(path) for path in file_paths]
    df = pd.concat(frames, axis=0, ignore_index=True)

    if parse_dates and "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"])

    return df


def load_experiment_data_from_config(config_path: str | Path, config_key: str = "output_data", file_key: str = "file2") -> pd.DataFrame:
    """
    Load the experiment metadata file using keys stored in a YAML config.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.
    config_key : str, default 'output_data'
        Top-level YAML key.
    file_key : str, default 'file2'
        Nested key containing the CSV path.

    Returns
    -------
    pd.DataFrame
        Experiment metadata table, typically with client-level variation labels.
    """
    config = load_yaml_config(config_path)
    file_path = config[config_key][file_key]
    return pd.read_csv(file_path)


# -------------------------------------------------------------------
# Data preparation helpers
# -------------------------------------------------------------------

def sort_events(
    df: pd.DataFrame,
    sort_cols: Sequence[str] = ("client_id", "date_time"),
) -> pd.DataFrame:
    """
    Sort event data so the user journey is in chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially cleaned event dataset.
    sort_cols : sequence of str
        Columns used to sort the events.

    Returns
    -------
    pd.DataFrame
        Sorted copy of the dataset.
    """
    return df.sort_values(list(sort_cols)).copy()


def remove_consecutive_duplicate_steps(
    df: pd.DataFrame,
    visit_col: str = "visit_id",
    step_col: str = "process_step",
) -> pd.DataFrame:
    """
    Remove repeated consecutive process steps within the same visit.

    Example:
    If a visit has ['step_1', 'step_1', 'step_2'], the second 'step_1'
    is removed because it does not represent a real transition.

    Parameters
    ----------
    df : pd.DataFrame
        Event dataset sorted by visit and time.
    visit_col : str, default 'visit_id'
        Visit identifier column.
    step_col : str, default 'process_step'
        Funnel step column.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset without consecutive repeated steps.
    """
    clean_df = df.copy()
    clean_df["prev_step"] = clean_df.groupby(visit_col)[step_col].shift()
    clean_df = clean_df[clean_df[step_col] != clean_df["prev_step"]].copy()
    return clean_df


def merge_events_with_experiment_data(
    events_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
    on: str = "client_id",
) -> pd.DataFrame:
    """
    Merge event-level data with experiment metadata.

    This is usually where the Variation column gets added to the event table.

    Parameters
    ----------
    events_df : pd.DataFrame
        Clean event log data.
    experiment_df : pd.DataFrame
        Experiment metadata, such as test/control assignment.
    on : str, default 'client_id'
        Merge key.

    Returns
    -------
    pd.DataFrame
        Merged dataset.
    """
    return pd.merge(events_df, experiment_df, on=on).copy()


# -------------------------------------------------------------------
# Funnel metrics
# -------------------------------------------------------------------

def build_funnel_table(
    df: pd.DataFrame,
    index_cols: Sequence[str] = ("visit_id", "client_id"),
    step_col: str = "process_step",
    time_col: str = "date_time",
) -> pd.DataFrame:
    """
    Build a funnel table with one row per visit/client and one column per step.

    Each cell stores the first timestamp at which that step was reached.

    Parameters
    ----------
    df : pd.DataFrame
        Clean event data.
    index_cols : sequence of str
        Columns that define a unique journey unit.
    step_col : str, default 'process_step'
        Funnel step column.
    time_col : str, default 'date_time'
        Timestamp column.

    Returns
    -------
    pd.DataFrame
        Pivoted funnel table.
    """
    return df.pivot_table(
        index=list(index_cols),
        columns=step_col,
        values=time_col,
        aggfunc="min",
    )


def add_conversion_flag(funnel_df: pd.DataFrame, target_step: str = "confirm") -> pd.DataFrame:
    """
    Add a binary conversion flag based on whether the target step was reached.

    Parameters
    ----------
    funnel_df : pd.DataFrame
        Funnel table created by `build_funnel_table`.
    target_step : str, default 'confirm'
        Final conversion step.

    Returns
    -------
    pd.DataFrame
        Funnel table with an added 'converted' column.
    """
    result = funnel_df.copy()
    result["converted"] = result[target_step].notna()
    return result


def compute_conversion_rate(funnel_df: pd.DataFrame, converted_col: str = "converted") -> float:
    """
    Compute the overall conversion rate.

    Parameters
    ----------
    funnel_df : pd.DataFrame
        Funnel table with a converted flag.
    converted_col : str, default 'converted'
        Conversion indicator column.

    Returns
    -------
    float
        Mean conversion rate.
    """
    return float(funnel_df[converted_col].mean())


def compute_step_reach_rates(funnel_df: pd.DataFrame, steps: Sequence[str]) -> pd.Series:
    """
    Compute the share of journeys that reached each funnel step.

    Parameters
    ----------
    funnel_df : pd.DataFrame
        Funnel table with step columns.
    steps : sequence of str
        Ordered funnel steps to measure.

    Returns
    -------
    pd.Series
        Reach rate for each step.
    """
    return funnel_df[list(steps)].notna().mean()


# -------------------------------------------------------------------
# Group comparison helpers
# -------------------------------------------------------------------

def split_by_variation(
    df: pd.DataFrame,
    variation_col: str = "Variation",
    control_label: str = "control",
    test_label: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into control and test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing a variation column.
    variation_col : str, default 'Variation'
        Group assignment column.
    control_label : str, default 'control'
        Label used for the control group.
    test_label : str, default 'test'
        Label used for the test group.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (control_df, test_df)
    """
    control_df = df[df[variation_col] == control_label].copy()
    test_df = df[df[variation_col] == test_label].copy()
    return control_df, test_df


def count_unique_clients_at_step(
    df: pd.DataFrame,
    step: str,
    client_col: str = "client_id",
    step_col: str = "process_step",
) -> int:
    """
    Count how many unique clients reached a specific funnel step.

    Parameters
    ----------
    df : pd.DataFrame
        Event dataset.
    step : str
        Funnel step to count.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Process step column.

    Returns
    -------
    int
        Number of unique clients that reached the step.
    """
    return int(df.loc[df[step_col] == step, client_col].nunique())


def compute_step_rate(
    df: pd.DataFrame,
    step: str,
    client_col: str = "client_id",
    step_col: str = "process_step",
) -> float:
    """
    Compute the unique-client reach rate for one funnel step.

    Parameters
    ----------
    df : pd.DataFrame
        Event dataset for a single variation.
    step : str
        Funnel step to evaluate.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Process step column.

    Returns
    -------
    float
        Step reach rate.
    """
    total_clients = df[client_col].nunique()
    if total_clients == 0:
        return np.nan
    reached_clients = count_unique_clients_at_step(df, step, client_col, step_col)
    return reached_clients / total_clients


def two_proportion_z_test(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    step: str,
    threshold: float = 0.05,
    client_col: str = "client_id",
    step_col: str = "process_step",
    alpha: float = 0.05,
) -> dict:
    """
    Run a one-sided two-proportion z-test for a given funnel step.

    The default notebook logic tested whether the test group improvement
    reaches a minimum threshold over the control group.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-group events.
    control_df : pd.DataFrame
        Control-group events.
    step : str
        Funnel step to test.
    threshold : float, default 0.05
        Minimum improvement target.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Process step column.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Statistical results and interpretation.
    """
    completed_test = count_unique_clients_at_step(test_df, step, client_col, step_col)
    completed_control = count_unique_clients_at_step(control_df, step, client_col, step_col)

    total_test = test_df[client_col].nunique()
    total_control = control_df[client_col].nunique()

    rate_test = completed_test / total_test if total_test else np.nan
    rate_control = completed_control / total_control if total_control else np.nan

    p_pool = (completed_test + completed_control) / (total_test + total_control)
    std_error = math.sqrt(p_pool * (1 - p_pool) * ((1 / total_test) + (1 / total_control)))
    z_score = (rate_test - rate_control - threshold) / std_error

    critical_value = st.norm.ppf(alpha)
    p_value = st.norm.cdf(z_score)

    return {
        "step": step,
        "rate_test": rate_test,
        "rate_control": rate_control,
        "difference": rate_test - rate_control,
        "threshold": threshold,
        "z_score": z_score,
        "critical_value": critical_value,
        "p_value": p_value,
        "reject_h0": z_score < critical_value,
        "interpretation": (
            "Improvement does NOT reach the threshold"
            if z_score < critical_value
            else "Improvement reaches the threshold"
        ),
    }


def run_stepwise_z_tests(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    threshold: float = 0.05,
    client_col: str = "client_id",
    step_col: str = "process_step",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run the same two-proportion z-test across all steps in the funnel.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-group events.
    control_df : pd.DataFrame
        Control-group events.
    steps : sequence of str
        Funnel steps to test.
    threshold : float, default 0.05
        Improvement target.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Process step column.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        One row per step with all test results.
    """
    results = [
        two_proportion_z_test(
            test_df=test_df,
            control_df=control_df,
            step=step,
            threshold=threshold,
            client_col=client_col,
            step_col=step_col,
            alpha=alpha,
        )
        for step in steps
    ]
    return pd.DataFrame(results)


def build_step_rate_comparison(
    merged_df: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    variation_col: str = "Variation",
    client_col: str = "client_id",
    step_col: str = "process_step",
) -> pd.DataFrame:
    """
    Create a long table of step reach rates for each variation.

    This function replaces hardcoded percentage lists such as
    [99.41, 85.64, ...] and calculates everything directly from the data.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Event-level dataset containing a variation column.
    steps : sequence of str
        Ordered funnel steps.
    variation_col : str, default 'Variation'
        Variation assignment column.
    client_col : str, default 'client_id'
        Client identifier column.
    step_col : str, default 'process_step'
        Funnel step column.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per variation and step.
    """
    rows = []
    for variation, group in merged_df.groupby(variation_col):
        for step in steps:
            rows.append(
                {
                    "Variation": variation,
                    "process_step": step,
                    "rate": compute_step_rate(group, step, client_col, step_col) * 100,
                }
            )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Time and transition analysis
# -------------------------------------------------------------------

def add_time_differences(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("client_id", "visit_id"),
    time_col: str = "date_time",
    output_col: str = "time_diff",
) -> pd.DataFrame:
    """
    Add a time difference column between consecutive events in each journey.

    Parameters
    ----------
    df : pd.DataFrame
        Ordered event dataset.
    group_cols : sequence of str
        Columns defining each user journey.
    time_col : str, default 'date_time'
        Timestamp column.
    output_col : str, default 'time_diff'
        Name of the resulting timedelta column.

    Returns
    -------
    pd.DataFrame
        Copy of the dataset with an added timedelta column.
    """
    result = df.copy()
    result[output_col] = result.groupby(list(group_cols))[time_col].diff()
    return result


def add_time_diff_seconds(
    df: pd.DataFrame,
    time_diff_col: str = "time_diff",
    output_col: str = "time_diff_seconds",
) -> pd.DataFrame:
    """
    Convert a timedelta column into seconds.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a timedelta column.
    time_diff_col : str, default 'time_diff'
        Timedelta column.
    output_col : str, default 'time_diff_seconds'
        Name of the resulting numeric column.

    Returns
    -------
    pd.DataFrame
        Copy of the dataset with a seconds column.
    """
    result = df.copy()
    result[output_col] = result[time_diff_col].dt.total_seconds()
    return result


def make_transition_labels(
    steps: Sequence[str] = DEFAULT_STEPS,
) -> list[str]:
    """
    Build the list of logical step-to-step transitions from an ordered funnel.

    Parameters
    ----------
    steps : sequence of str
        Ordered funnel steps.

    Returns
    -------
    list[str]
        Transition labels such as 'start -> step_1'.
    """
    return [f"{steps[i]} -> {steps[i+1]}" for i in range(len(steps) - 1)]


def build_transition_table(
    df: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    group_cols: Sequence[str] = ("client_id", "visit_id"),
    step_col: str = "process_step",
    time_col: str = "date_time",
) -> pd.DataFrame:
    """
    Create a transition-level table with durations between consecutive steps.

    Parameters
    ----------
    df : pd.DataFrame
        Ordered event dataset.
    steps : sequence of str
        Ordered funnel steps.
    group_cols : sequence of str
        Columns defining each user journey.
    step_col : str, default 'process_step'
        Process step column.
    time_col : str, default 'date_time'
        Timestamp column.

    Returns
    -------
    pd.DataFrame
        Transition-level dataset restricted to logical funnel transitions.
    """
    result = df.copy()

    # Create the next step within each journey.
    result["next_step"] = result.groupby(list(group_cols))[step_col].shift(-1)

    # Calculate absolute time until the next step.
    result["interval_duration"] = (
        result.groupby(list(group_cols))[time_col]
        .diff(-1)
        .dt.total_seconds()
        .abs()
    )

    # Build a readable transition label.
    result["transition"] = result[step_col] + " -> " + result["next_step"]

    # Keep only the expected logical transitions from the funnel definition.
    valid_transitions = make_transition_labels(steps)
    return result[result["transition"].isin(valid_transitions)].copy()


def summarize_transition_durations(
    transitions_df: pd.DataFrame,
    variation_col: str = "Variation",
    transition_col: str = "transition",
    duration_col: str = "interval_duration",
) -> pd.DataFrame:
    """
    Summarize transition durations using mean and median by group.

    Parameters
    ----------
    transitions_df : pd.DataFrame
        Transition-level dataset.
    variation_col : str, default 'Variation'
        Group assignment column.
    transition_col : str, default 'transition'
        Transition label column.
    duration_col : str, default 'interval_duration'
        Transition duration column in seconds.

    Returns
    -------
    pd.DataFrame
        Pivot-style summary table.
    """
    return (
        transitions_df
        .groupby([variation_col, transition_col])[duration_col]
        .agg(["mean", "median"])
        .unstack(level=0)
    )


def filter_transition_outliers(
    transitions_df: pd.DataFrame,
    limits: dict[str, float],
    transition_col: str = "transition",
    duration_col: str = "interval_duration",
    default_limit: float = np.inf,
) -> pd.DataFrame:
    """
    Filter transition records using a per-transition duration threshold.

    This replaces the notebook's hardcoded lambda and dictionary logic
    with a reusable function.

    Parameters
    ----------
    transitions_df : pd.DataFrame
        Transition-level dataset.
    limits : dict[str, float]
        Maximum allowed duration per transition.
    transition_col : str, default 'transition'
        Transition label column.
    duration_col : str, default 'interval_duration'
        Duration column.
    default_limit : float, default np.inf
        Fallback limit for transitions not present in the dictionary.

    Returns
    -------
    pd.DataFrame
        Filtered transition dataset.
    """
    max_allowed = transitions_df[transition_col].map(limits).fillna(default_limit)
    return transitions_df.loc[transitions_df[duration_col] <= max_allowed].copy()


def compute_max_transition_time(
    transitions_df: pd.DataFrame,
    duration_col: str = "interval_duration",
) -> float:
    """
    Return the maximum transition duration observed in the data.

    Parameters
    ----------
    transitions_df : pd.DataFrame
        Transition-level dataset.
    duration_col : str, default 'interval_duration'
        Transition duration column.

    Returns
    -------
    float
        Maximum duration in seconds.
    """
    return float(transitions_df[duration_col].max())


def run_transition_ttests(
    transitions_df: pd.DataFrame,
    transition_col: str = "transition",
    variation_col: str = "Variation",
    duration_col: str = "interval_duration",
    control_label: str = "control",
    test_label: str = "test",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run Welch's t-test for each transition comparing control vs test durations.

    Parameters
    ----------
    transitions_df : pd.DataFrame
        Transition-level dataset.
    transition_col : str, default 'transition'
        Transition label column.
    variation_col : str, default 'Variation'
        Variation assignment column.
    duration_col : str, default 'interval_duration'
        Duration column in seconds.
    control_label : str, default 'control'
        Label used for the control group.
    test_label : str, default 'test'
        Label used for the test group.
    alpha : float, default 0.05
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        T-test results per transition.
    """
    results = []

    for transition in transitions_df[transition_col].dropna().unique():
        group_control = transitions_df[
            (transitions_df[transition_col] == transition)
            & (transitions_df[variation_col] == control_label)
        ][duration_col]

        group_test = transitions_df[
            (transitions_df[transition_col] == transition)
            & (transitions_df[variation_col] == test_label)
        ][duration_col]

        t_stat, p_val = st.ttest_ind(group_control, group_test, equal_var=False)

        results.append(
            {
                "Transition": transition,
                "Control Mean": round(group_control.mean(), 4),
                "Test Mean": round(group_test.mean(), 4),
                "T-Statistic": round(t_stat, 4),
                "P-Value": round(p_val, 4),
                "Significant?": "Yes" if p_val < alpha else "No",
            }
        )

    return pd.DataFrame(results)


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def _ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent folder of an output file if it does not exist.

    Parameters
    ----------
    path : str | Path
        Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_step_completion_rates(
    rates_df: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    variation_order: Sequence[str] = DEFAULT_VARIATION_ORDER,
    colors: dict[str, str] = DEFAULT_COLORS,
    title: str = "Completion Rate per Step: Control vs Test",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot grouped bars comparing step reach rates by variation.

    Parameters
    ----------
    rates_df : pd.DataFrame
        Output of `build_step_rate_comparison`.
    steps : sequence of str
        Ordered funnel steps for the x-axis.
    variation_order : sequence of str
        Order of the bars inside each step.
    colors : dict[str, str]
        Color palette keyed by variation label.
    title : str
        Plot title.
    save_path : str | Path | None, default None
        If provided, save the figure to disk.
    """
    plot_df = rates_df.copy()
    plot_df["process_step"] = pd.Categorical(plot_df["process_step"], categories=steps, ordered=True)
    plot_df = plot_df.sort_values(["process_step", "Variation"])

    x = np.arange(len(steps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, variation in enumerate(variation_order):
        sub = plot_df[plot_df["Variation"] == variation]
        heights = sub.set_index("process_step").reindex(steps)["rate"].values
        offset = (-width / 2) if i == 0 else (width / 2)

        bars = ax.bar(
            x + offset,
            heights,
            width,
            label=variation.capitalize(),
            color=colors.get(variation, None),
        )

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Process Step")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.set_ylim(0, max(105, plot_df["rate"].max() + 5))
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_transition_duration_boxplot(
    transitions_df: pd.DataFrame,
    variation_order: Sequence[str] = DEFAULT_VARIATION_ORDER,
    colors: dict[str, str] = DEFAULT_COLORS,
    title: str = "Checking for Outliers: Control vs Test",
    y_limit: tuple[float, float] | None = (0, 500),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot transition duration distributions to inspect outliers by group.

    Parameters
    ----------
    transitions_df : pd.DataFrame
        Transition-level dataset.
    variation_order : sequence of str
        Order of hue levels.
    colors : dict[str, str]
        Plot color palette.
    title : str
        Plot title.
    y_limit : tuple[float, float] | None, default (0, 500)
        Optional y-axis limits.
    save_path : str | Path | None, default None
        If provided, save the figure to disk.
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    sns.boxplot(
        data=transitions_df,
        x="transition",
        y="interval_duration",
        hue="Variation",
        hue_order=list(variation_order),
        palette=colors,
    )

    plt.title(title)
    plt.ylabel("Seconds")

    if y_limit is not None:
        plt.ylim(*y_limit)

    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_clean_mean_durations(
    summary_df: pd.DataFrame,
    colors: dict[str, str] = DEFAULT_COLORS,
    title: str = "Cleaned Average Duration per Step (Filtered Data)",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the mean duration per transition after outlier filtering.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of `summarize_transition_durations`.
    colors : dict[str, str]
        Plot color palette.
    title : str
        Plot title.
    save_path : str | Path | None, default None
        If provided, save the figure to disk.
    """
    plot_df = (
        summary_df["mean"]
        .reset_index()
        .melt(id_vars="transition", var_name="Variation", value_name="mean_seconds")
    )

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=plot_df,
        x="transition",
        y="mean_seconds",
        hue="Variation",
        hue_order=list(DEFAULT_VARIATION_ORDER),
        palette=colors,
    )

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Seconds (Mean)")
    plt.xlabel("User Journey")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    plt.tight_layout()

    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# -------------------------------------------------------------------
# End-to-end pipeline
# -------------------------------------------------------------------

def run_full_analysis(
    web_file_paths: Sequence[str | Path],
    experiment_df: pd.DataFrame,
    steps: Sequence[str] = DEFAULT_STEPS,
    transition_limits: dict[str, float] = DEFAULT_TRANSITION_LIMITS,
) -> dict[str, pd.DataFrame | float]:
    """
    Run the full analysis pipeline and return all major outputs.

    This is the orchestration layer that chains together the reusable
    functions defined above.

    Parameters
    ----------
    web_file_paths : sequence of str or Path
        Raw event files to concatenate.
    experiment_df : pd.DataFrame
        Experiment metadata containing control/test labels.
    steps : sequence of str, default DEFAULT_STEPS
        Ordered funnel steps.
    transition_limits : dict[str, float], default DEFAULT_TRANSITION_LIMITS
        Outlier thresholds per transition.

    Returns
    -------
    dict
        Dictionary with the most important analysis outputs.
    """
    # Load and clean the event data.
    web_df = load_event_files(web_file_paths)
    web_df = sort_events(web_df)
    web_clean = remove_consecutive_duplicate_steps(web_df)

    # Funnel analysis.
    funnel_df = build_funnel_table(web_clean)
    funnel_df = add_conversion_flag(funnel_df)
    conversion_rate = compute_conversion_rate(funnel_df)
    step_reach_rates = compute_step_reach_rates(funnel_df, steps)

    # Merge with experiment metadata.
    merged_df = merge_events_with_experiment_data(web_clean, experiment_df)
    merged_df = sort_events(merged_df)

    # Group split and proportion tests.
    control_df, test_df = split_by_variation(merged_df)
    z_test_results = run_stepwise_z_tests(test_df=test_df, control_df=control_df, steps=steps)

    # Completion-rate comparison for plotting.
    step_rate_comparison = build_step_rate_comparison(merged_df, steps=steps)

    # Time and transition analysis.
    merged_df = add_time_differences(merged_df)
    merged_df = add_time_diff_seconds(merged_df)

    transitions_df = build_transition_table(merged_df, steps=steps)
    max_transition_time = compute_max_transition_time(transitions_df)
    transition_summary = summarize_transition_durations(transitions_df)

    filtered_transitions_df = filter_transition_outliers(
        transitions_df=transitions_df,
        limits=transition_limits,
    )
    filtered_transition_summary = summarize_transition_durations(filtered_transitions_df)

    ttest_df = run_transition_ttests(filtered_transitions_df)

    return {
        "web_df": web_df,
        "web_clean": web_clean,
        "funnel_df": funnel_df,
        "conversion_rate": conversion_rate,
        "step_reach_rates": step_reach_rates,
        "merged_df": merged_df,
        "control_df": control_df,
        "test_df": test_df,
        "z_test_results": z_test_results,
        "step_rate_comparison": step_rate_comparison,
        "transitions_df": transitions_df,
        "max_transition_time": max_transition_time,
        "transition_summary": transition_summary,
        "filtered_transitions_df": filtered_transitions_df,
        "filtered_transition_summary": filtered_transition_summary,
        "ttest_df": ttest_df,
    }


def main() -> None:
    """
    Example execution entry point.

    Update the paths below to match your project structure before running
    this module as a script.
    """
    web_paths = [
        "../data/raw/df_final_web_data_pt_1.txt",
        "../data/raw/df_final_web_data_pt_2.txt",
    ]
    config_path = "../config.yaml"

    experiment_df = load_experiment_data_from_config(config_path)

    results = run_full_analysis(
        web_file_paths=web_paths,
        experiment_df=experiment_df,
    )

    print(f"Overall conversion rate: {results['conversion_rate']:.2%}")
    print(f"Slowest transition: {results['max_transition_time'] / 60:.2f} minutes")

    print("\nStepwise z-tests:")
    print(results["z_test_results"])

    print("\nTransition t-tests:")
    print(results["ttest_df"])

    # Example plots
    plot_step_completion_rates(results["step_rate_comparison"])
    plot_transition_duration_boxplot(results["transitions_df"])
    plot_clean_mean_durations(results["filtered_transition_summary"])


if __name__ == "__main__":
    main()
