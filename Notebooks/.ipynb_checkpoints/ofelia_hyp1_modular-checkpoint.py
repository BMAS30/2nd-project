
"""
Reusable completion-rate hypothesis testing module.

This module converts the notebook logic into reusable functions so the
analysis can be applied to other A/B tests without rewriting the same code.

Main capabilities
-----------------
1. Load and prepare experiment and web-event datasets.
2. Merge event data with experiment assignment data.
3. Calculate completion rates by variation.
4. Run a chi-square test on completion vs non-completion.
5. Calculate absolute lift and relative uplift.
6. Print a reusable text summary of the main findings.

The code is modular and each function includes comments and docstrings
explaining its role and expected inputs/outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from scipy import stats


# -------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------

# Default final step used to define a completed journey.
DEFAULT_CONFIRM_STEP = "confirm"

# Default variation labels for A/B testing.
DEFAULT_CONTROL_LABEL = "control"
DEFAULT_TEST_LABEL = "test"


# -------------------------------------------------------------------
# File loading and preparation
# -------------------------------------------------------------------

def load_csv_file(file_path: str | Path, parse_date_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Load a CSV file and optionally parse one or more date columns.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.
    parse_date_cols : sequence[str] | None, default None
        List of columns that should be converted to datetime.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    df = pd.read_csv(file_path)

    if parse_date_cols:
        for col in parse_date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    return df


def sort_event_data(
    df: pd.DataFrame,
    sort_cols: Sequence[str] = ("visit_id", "date_time"),
) -> pd.DataFrame:
    """
    Sort the event log in the correct chronological order.

    This is useful before any journey or funnel analysis because the order
    of events matters when reconstructing the user path.

    Parameters
    ----------
    df : pd.DataFrame
        Raw event dataset.
    sort_cols : sequence[str], default ('visit_id', 'date_time')
        Columns used to sort the data.

    Returns
    -------
    pd.DataFrame
        Sorted copy of the event dataset.
    """
    return df.sort_values(list(sort_cols)).copy()


def merge_web_and_experiment_data(
    web_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
    on: str = "client_id",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge the web-event data with the experiment assignment data.

    In this analysis, the merge is used to attach the Variation column
    to the event-level table.

    Parameters
    ----------
    web_df : pd.DataFrame
        Event-level dataset.
    experiment_df : pd.DataFrame
        Experiment metadata dataset.
    on : str, default 'client_id'
        Merge key.
    how : str, default 'inner'
        Merge strategy.

    Returns
    -------
    pd.DataFrame
        Merged dataset.
    """
    return pd.merge(web_df, experiment_df, on=on, how=how).copy()


# -------------------------------------------------------------------
# Core completion metrics
# -------------------------------------------------------------------

def count_total_unique_entities(
    df: pd.DataFrame,
    variation_col: str = "Variation",
    entity_col: str = "visitor_id",
) -> pd.DataFrame:
    """
    Count the total number of unique entities per variation.

    In the original notebook, the entity of interest is visitor_id.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    variation_col : str, default 'Variation'
        A/B group column.
    entity_col : str, default 'visitor_id'
        Column representing the unique unit of analysis.

    Returns
    -------
    pd.DataFrame
        Table with one row per variation and total unique entities.
    """
    result = df.groupby(variation_col)[entity_col].nunique().reset_index()
    result.columns = [variation_col, "total_entities"]
    return result


def count_completed_unique_entities(
    df: pd.DataFrame,
    completion_step: str = DEFAULT_CONFIRM_STEP,
    variation_col: str = "Variation",
    entity_col: str = "visitor_id",
    step_col: str = "process_step",
) -> pd.DataFrame:
    """
    Count how many unique entities completed the process in each variation.

    Completion is defined as reaching a specific final process step.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    completion_step : str, default 'confirm'
        Final step that marks completion.
    variation_col : str, default 'Variation'
        A/B group column.
    entity_col : str, default 'visitor_id'
        Unit of analysis.
    step_col : str, default 'process_step'
        Funnel-step column.

    Returns
    -------
    pd.DataFrame
        Table with one row per variation and completed unique entities.
    """
    result = (
        df[df[step_col] == completion_step]
        .groupby(variation_col)[entity_col]
        .nunique()
        .reset_index()
    )
    result.columns = [variation_col, "completed_entities"]
    return result


def build_completion_summary(
    df: pd.DataFrame,
    completion_step: str = DEFAULT_CONFIRM_STEP,
    variation_col: str = "Variation",
    entity_col: str = "visitor_id",
    step_col: str = "process_step",
) -> pd.DataFrame:
    """
    Build the full completion summary table by variation.

    The output includes:
    - total unique entities
    - completed unique entities
    - non-completed unique entities
    - completion rate as a proportion
    - completion rate as a percentage

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    completion_step : str, default 'confirm'
        Final process step used to define completion.
    variation_col : str, default 'Variation'
        A/B group column.
    entity_col : str, default 'visitor_id'
        Unit of analysis.
    step_col : str, default 'process_step'
        Funnel-step column.

    Returns
    -------
    pd.DataFrame
        Completion summary by variation.
    """
    total = count_total_unique_entities(df, variation_col=variation_col, entity_col=entity_col)
    completed = count_completed_unique_entities(
        df,
        completion_step=completion_step,
        variation_col=variation_col,
        entity_col=entity_col,
        step_col=step_col,
    )

    result = total.merge(completed, on=variation_col, how="left")
    result["completed_entities"] = result["completed_entities"].fillna(0).astype(int)
    result["not_completed"] = result["total_entities"] - result["completed_entities"]
    result["completion_rate"] = result["completed_entities"] / result["total_entities"]
    result["completion_rate_%"] = (result["completion_rate"] * 100).round(2)

    return result


def calculate_completion_rate(
    df: pd.DataFrame,
    completion_step: str = DEFAULT_CONFIRM_STEP,
    variation_col: str = "Variation",
    entity_col: str = "visitor_id",
    step_col: str = "process_step",
) -> pd.DataFrame:
    """
    Calculate completion rates by variation.

    This is the modular version of the original notebook function, but it
    now relies on a reusable helper that builds the full summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    completion_step : str, default 'confirm'
        Final process step used to define completion.
    variation_col : str, default 'Variation'
        A/B group column.
    entity_col : str, default 'visitor_id'
        Unit of analysis.
    step_col : str, default 'process_step'
        Funnel-step column.

    Returns
    -------
    pd.DataFrame
        Completion summary focused on completion metrics.
    """
    summary = build_completion_summary(
        df=df,
        completion_step=completion_step,
        variation_col=variation_col,
        entity_col=entity_col,
        step_col=step_col,
    )

    return summary[[variation_col, "total_entities", "completed_entities", "completion_rate_%"]].copy()


# -------------------------------------------------------------------
# Statistical testing
# -------------------------------------------------------------------

def build_chi_square_contingency_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the contingency table used in the chi-square test.

    The expected structure is:
    [[completed, not_completed],
     [completed, not_completed]]

    Parameters
    ----------
    summary_df : pd.DataFrame
        Completion summary returned by `build_completion_summary`.

    Returns
    -------
    pd.DataFrame
        Two-column contingency table with completed vs not completed counts.
    """
    return summary_df[["completed_entities", "not_completed"]].copy()


def run_completion_chi_square_test(summary_df: pd.DataFrame) -> dict:
    """
    Run a chi-square test on completion vs non-completion by variation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Completion summary returned by `build_completion_summary`.

    Returns
    -------
    dict
        Dictionary containing chi-square statistic, p-value, degrees of
        freedom, and expected frequencies.
    """
    contingency_table = build_chi_square_contingency_table(summary_df).values
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "expected": expected,
    }


def calculate_completion_rate_with_pvalue(
    df: pd.DataFrame,
    completion_step: str = DEFAULT_CONFIRM_STEP,
    variation_col: str = "Variation",
    entity_col: str = "visitor_id",
    step_col: str = "process_step",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate completion rates and run a chi-square significance test.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    completion_step : str, default 'confirm'
        Final process step used to define completion.
    variation_col : str, default 'Variation'
        A/B group column.
    entity_col : str, default 'visitor_id'
        Unit of analysis.
    step_col : str, default 'process_step'
        Funnel-step column.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Completion summary and test results.
    """
    summary = build_completion_summary(
        df=df,
        completion_step=completion_step,
        variation_col=variation_col,
        entity_col=entity_col,
        step_col=step_col,
    )

    test_result = run_completion_chi_square_test(summary)
    test_result["significant"] = test_result["p_value"] < alpha
    test_result["alpha"] = alpha

    return summary, test_result


# -------------------------------------------------------------------
# Uplift analysis
# -------------------------------------------------------------------

def extract_group_completion_rates(
    summary_df: pd.DataFrame,
    control_label: str = DEFAULT_CONTROL_LABEL,
    test_label: str = DEFAULT_TEST_LABEL,
    variation_col: str = "Variation",
) -> tuple[float, float]:
    """
    Extract control and test completion rates from the summary table.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Completion summary.
    control_label : str, default 'control'
        Label used for the control group.
    test_label : str, default 'test'
        Label used for the test group.
    variation_col : str, default 'Variation'
        Variation column name.

    Returns
    -------
    tuple[float, float]
        Control completion rate, test completion rate.
    """
    control_rate = summary_df.loc[
        summary_df[variation_col] == control_label, "completion_rate"
    ].values[0]

    test_rate = summary_df.loc[
        summary_df[variation_col] == test_label, "completion_rate"
    ].values[0]

    return control_rate, test_rate


def calculate_relative_uplift(
    df: pd.DataFrame,
    completion_step: str = DEFAULT_CONFIRM_STEP,
    control_label: str = DEFAULT_CONTROL_LABEL,
    test_label: str = DEFAULT_TEST_LABEL,
    variation_col: str = "Variation",
    entity_col: str = "visitor_id",
    step_col: str = "process_step",
) -> dict:
    """
    Calculate absolute uplift and relative uplift in completion rate.

    Formulas
    --------
    Absolute difference (pp) = (test_rate - control_rate) * 100
    Relative uplift (%)      = ((test_rate - control_rate) / control_rate) * 100

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    completion_step : str, default 'confirm'
        Final process step used to define completion.
    control_label : str, default 'control'
        Label used for the control group.
    test_label : str, default 'test'
        Label used for the test group.
    variation_col : str, default 'Variation'
        A/B group column.
    entity_col : str, default 'visitor_id'
        Unit of analysis.
    step_col : str, default 'process_step'
        Funnel-step column.

    Returns
    -------
    dict
        Control rate, test rate, absolute difference, and relative uplift.
    """
    summary = build_completion_summary(
        df=df,
        completion_step=completion_step,
        variation_col=variation_col,
        entity_col=entity_col,
        step_col=step_col,
    )

    control_rate, test_rate = extract_group_completion_rates(
        summary_df=summary,
        control_label=control_label,
        test_label=test_label,
        variation_col=variation_col,
    )

    absolute_diff_pp = (test_rate - control_rate) * 100
    relative_uplift_pct = ((test_rate - control_rate) / control_rate) * 100

    return {
        "control_rate": control_rate,
        "test_rate": test_rate,
        "absolute_difference_pp": absolute_diff_pp,
        "relative_uplift_pct": relative_uplift_pct,
    }


# -------------------------------------------------------------------
# Reporting helpers
# -------------------------------------------------------------------

def print_completion_report(
    summary_df: pd.DataFrame,
    test_result: dict,
    alpha: float = 0.05,
    variation_col: str = "Variation",
) -> None:
    """
    Print a formatted completion-rate significance report.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Completion summary.
    test_result : dict
        Output of `run_completion_chi_square_test`.
    alpha : float, default 0.05
        Significance level.
    variation_col : str, default 'Variation'
        Variation column name.
    """
    print("=" * 45)
    print(summary_df[[variation_col, "total_entities", "completed_entities", "completion_rate_%"]].to_string(index=False))
    print("=" * 45)
    print(f"Chi-square statistic : {test_result['chi2']:.4f}")
    print(f"P-value              : {test_result['p_value']:.4f}")
    print(f"Degrees of freedom   : {test_result['dof']}")
    print("-" * 45)

    if test_result["p_value"] < alpha:
        print("✅ Result is SIGNIFICANT (p < 0.05)")
        print("   The difference in completion rates is unlikely to be due to chance.")
    else:
        print("❌ Result is NOT significant (p >= 0.05)")
        print("   The observed difference may be explained by random variation.")


def print_relative_uplift_report(uplift_result: dict) -> None:
    """
    Print a formatted uplift report.

    Parameters
    ----------
    uplift_result : dict
        Output of `calculate_relative_uplift`.
    """
    print("=" * 45)
    print("        RELATIVE UPLIFT ANALYSIS")
    print("=" * 45)
    print(f"Control completion rate : {uplift_result['control_rate'] * 100:.2f}%")
    print(f"Test completion rate    : {uplift_result['test_rate'] * 100:.2f}%")
    print("-" * 45)
    print(f"Absolute difference     : {uplift_result['absolute_difference_pp']:+.2f} pp")
    print(f"Relative uplift         : {uplift_result['relative_uplift_pct']:+.2f}%")
    print("=" * 45)


def build_conclusion_text(
    summary_df: pd.DataFrame,
    test_result: dict,
    uplift_result: dict,
    control_label: str = DEFAULT_CONTROL_LABEL,
    test_label: str = DEFAULT_TEST_LABEL,
    variation_col: str = "Variation",
) -> str:
    """
    Build a reusable written conclusion for the hypothesis test.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Completion summary table.
    test_result : dict
        Chi-square test output.
    uplift_result : dict
        Uplift metrics output.
    control_label : str, default 'control'
        Control-group label.
    test_label : str, default 'test'
        Test-group label.
    variation_col : str, default 'Variation'
        Variation column name.

    Returns
    -------
    str
        Human-readable conclusion.
    """
    control_rate_pct = summary_df.loc[
        summary_df[variation_col] == control_label, "completion_rate_%"
    ].values[0]

    test_rate_pct = summary_df.loc[
        summary_df[variation_col] == test_label, "completion_rate_%"
    ].values[0]

    if test_result["p_value"] < 0.05:
        significance_text = (
            "The chi-square test confirms that this difference is statistically significant."
        )
        recommendation = (
            "Recommendation: Based on both statistical significance and practical effect size, "
            "the test variation should be adopted as the new default experience."
        )
    else:
        significance_text = (
            "The chi-square test does not provide enough evidence to conclude that the difference is statistically significant."
        )
        recommendation = (
            "Recommendation: Do not roll out the test variation yet. More data or a new experiment may be needed."
        )

    conclusion = (
        "Completion Rate Analysis\n"
        f"The test group achieved a completion rate of {test_rate_pct:.2f}% compared to "
        f"{control_rate_pct:.2f}% in the control group, representing an absolute improvement of "
        f"{uplift_result['absolute_difference_pp']:+.2f} percentage points and a relative uplift of "
        f"{uplift_result['relative_uplift_pct']:+.2f}%.\n"
        f"{significance_text}\n"
        f"The p-value was {test_result['p_value']:.6f}, which indicates whether the observed difference "
        f"is likely to be explained by chance alone.\n"
        f"{recommendation}"
    )

    return conclusion


# -------------------------------------------------------------------
# End-to-end pipeline
# -------------------------------------------------------------------

def run_completion_hypothesis_analysis(
    web_file_path: str | Path,
    experiment_file_path: str | Path,
    completion_step: str = DEFAULT_CONFIRM_STEP,
) -> dict:
    """
    Run the full completion-rate analysis from raw files to final outputs.

    Parameters
    ----------
    web_file_path : str | Path
        Path to the cleaned web-event file.
    experiment_file_path : str | Path
        Path to the cleaned experiment-clients file.
    completion_step : str, default 'confirm'
        Final process step used to define completion.

    Returns
    -------
    dict
        Main outputs of the analysis.
    """
    # Load the two source tables used in the original notebook.
    web_df = load_csv_file(web_file_path, parse_date_cols=["date_time"])
    experiment_df = load_csv_file(experiment_file_path)

    # Sort the event data before merging so the journey remains chronological.
    web_df = sort_event_data(web_df)

    # Merge events with variation assignment.
    merged_df = merge_web_and_experiment_data(web_df, experiment_df, on="client_id", how="inner")

    # Build summary and run significance test.
    summary_df, test_result = calculate_completion_rate_with_pvalue(
        merged_df,
        completion_step=completion_step,
    )

    # Compute uplift metrics.
    uplift_result = calculate_relative_uplift(
        merged_df,
        completion_step=completion_step,
    )

    # Build reusable conclusion text.
    conclusion = build_conclusion_text(
        summary_df=summary_df,
        test_result=test_result,
        uplift_result=uplift_result,
    )

    return {
        "web_df": web_df,
        "experiment_df": experiment_df,
        "merged_df": merged_df,
        "summary_df": summary_df,
        "test_result": test_result,
        "uplift_result": uplift_result,
        "conclusion": conclusion,
    }


def main() -> None:
    """
    Example entry point for running this module as a script.

    Update the file paths below to match your project structure.
    """
    web_file = "../Data/clean/df_final_web_data.csv"
    experiment_file = "../Data/clean/df_final_experiment_clients_clean.csv"

    results = run_completion_hypothesis_analysis(
        web_file_path=web_file,
        experiment_file_path=experiment_file,
    )

    print_completion_report(results["summary_df"], results["test_result"])
    print()
    print_relative_uplift_report(results["uplift_result"])
    print()
    print(results["conclusion"])


if __name__ == "__main__":
    main()
