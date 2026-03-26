import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# 1. LOAD CLEANED DATA VIA CONFIG
# ─────────────────────────────────────────────
print("Loading data...")

# Resolve paths relative to config.yaml's location, not the working directory
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Base directory = where config.yaml lives (i.e. Notebooks/)
base = CONFIG_PATH.parent / "Data"
op = config["output_data"]

demo = pd.read_csv(base / "clean/demo_df_cleaned.csv")
exp  = pd.read_csv(base / "clean/df_final_experiment_clients_clean.csv")
web  = pd.read_csv(base / "clean/df_web.csv", parse_dates=["date_time"])

print(f"  web rows:   {len(web):,}")
print(f"  exp rows:   {len(exp):,}")
print(f"  demo rows:  {len(demo):,}")
# ─────────────────────────────────────────────
# 2. STANDARDISE process_step VALUES
# ─────────────────────────────────────────────
step_order = {"start": 1, "step_1": 2, "step_2": 3, "step_3": 4, "confirm": 5}

web["process_step"] = web["process_step"].str.strip().str.lower()

# Report any unexpected step names
unexpected = web[~web["process_step"].isin(step_order.keys())]["process_step"].unique()
if len(unexpected):
    print(f"  ⚠️  Unexpected step values found: {unexpected}")

web["step_order"] = web["process_step"].map(step_order)

# ─────────────────────────────────────────────
# 3. SORT & COMPUTE TIME ON STEP (per visit_id)
# ─────────────────────────────────────────────
web = web.sort_values(["visit_id", "date_time"]).reset_index(drop=True)

# ── Only keep forward-moving steps ──────────────────────
web["prev_step_order"] = web.groupby("visit_id")["step_order"].shift(1).fillna(0)
web = web[web["step_order"] >= web["prev_step_order"]].copy()
web = web.drop(columns=["prev_step_order"])

# ── Time on step (time until next step in same visit) ───
web["time_on_step_seconds"] = (
    web.groupby("visit_id")["date_time"]
    .diff()
    .shift(-1)
    .dt.total_seconds()
)

# ── Cap per step based on boxplot whiskers ───────────────
step_caps = {
    "start":  110,
    "step_1": 90,
    "step_2": 240,
    "step_3": 320,
}
for step, cap in step_caps.items():
    web.loc[web["process_step"] == step, "time_on_step_seconds"] = (
        web.loc[web["process_step"] == step, "time_on_step_seconds"].clip(upper=cap)
    )

# ─────────────────────────────────────────────
# 4. FLAG COMPLETION & DROP-OFF PER VISIT
# ─────────────────────────────────────────────
# Did this visit reach 'confirm'?
web["visit_completed"] = web["process_step"] == "confirm"

# Furthest step reached per visit (for drop-off analysis)
furthest = (
    web.groupby("visit_id")["step_order"]
    .max()
    .reset_index()
    .rename(columns={"step_order": "furthest_step"})
)
web = web.merge(furthest, on="visit_id", how="left")

# ─────────────────────────────────────────────
# 5. JOIN EXPERIMENT GROUP (inner join = keep only experiment participants)
# ─────────────────────────────────────────────
web = web.merge(exp, on="client_id", how="inner")
print(f"  After joining experiment clients: {len(web):,} rows")

# ─────────────────────────────────────────────
# 6. JOIN DEMOGRAPHICS (left join = keep all experiment users)
# ─────────────────────────────────────────────
web = web.merge(demo, on="client_id", how="left")

# ─────────────────────────────────────────────
# 7. CREATE AGE GROUPS
# ─────────────────────────────────────────────
def age_group(age):
    if pd.isna(age):
        return "Unknown"
    elif age < 30:
        return "Under 30"
    elif age < 45:
        return "30–44"
    elif age < 60:
        return "45–59"
    else:
        return "60+"

web["age_group"] = web["clnt_age"].apply(age_group)
# Remove Unknown age group
web = web[web["age_group"] != "Unknown"]
# Make age_group an ordered category so Tableau sorts it correctly
age_order = ["Under 30", "30–44", "45–59", "60+"]
web["age_group"] = pd.Categorical(web["age_group"], categories=age_order, ordered=True)
# ─────────────────────────────────────────────
# 8. CLEAN GENDER
# ─────────────────────────────────────────────
gender_map = {"M": "Male", "F": "Female", "U": "Unknown", "X": "Unknown"}
web["gender"] = web["gendr"].map(gender_map).fillna("Unknown")

# ─────────────────────────────────────────────
# 9. ADD DATE HELPERS (for Tableau date filters)
# ─────────────────────────────────────────────
web["date"]         = web["date_time"].dt.date
web["week"]         = web["date_time"].dt.to_period("W").astype(str)
web["month"]        = web["date_time"].dt.to_period("M").astype(str)

# ─────────────────────────────────────────────
# 10. EXPORT: MAIN EVENTS TABLE (for Tableau)
# ─────────────────────────────────────────────
cols_to_export = [
    "client_id", "visitor_id", "visit_id",
    "process_step", "step_order",
    "date_time", "date", "week", "month",
    "time_on_step_seconds",
    "visit_completed", "furthest_step",
    "Variation",                            # test / control
    "clnt_age", "age_group", "gender",
    "clnt_tenure_yr", "clnt_tenure_mnth",
    "num_accts", "bal",
    "calls_6_mnth", "logons_6_mnth",
]
# Only keep columns that actually exist (handles missing demo data gracefully)
cols_to_export = [c for c in cols_to_export if c in web.columns]

web[cols_to_export].to_csv(base / "clean/ab_test_tableau_ready.csv", index=False)
print(f"\nExported: ab_test_tableau_ready.csv ({len(web):,} rows, {len(cols_to_export)} columns)")

# ─────────────────────────────────────────────
# 11. EXPORT: FUNNEL SUMMARY (one row per group × step)
# ─────────────────────────────────────────────
funnel = (
    web.groupby(["Variation", "process_step", "step_order"])
    .agg(
        users=("client_id", "nunique"),
        visits=("visit_id", "nunique"),
        avg_time_on_step=("time_on_step_seconds", "mean"),
    )
    .reset_index()
    .sort_values(["Variation", "step_order"])
)

# Total users per group (denominator for completion rate)
total_per_group = web.groupby("Variation")["client_id"].nunique().rename("total_users")
funnel = funnel.merge(total_per_group, on="Variation")
funnel["pct_of_starters"] = (funnel["users"] / funnel["total_users"] * 100).round(2)

funnel.to_csv("ab_test_funnel_summary.csv", index=False)
print(f"Exported: ab_test_funnel_summary.csv ({len(funnel)} rows)")

# ─────────────────────────────────────────────
# 12. EXPORT: COMPLETION RATE BY DEMOGRAPHICS
# ─────────────────────────────────────────────
demo_summary = (
    web.groupby(["Variation", "age_group", "gender"])
    .agg(
        total_visits=("visit_id", "nunique"),
        completed_visits=("visit_id", lambda x: web.loc[x.index, "visit_completed"].sum()),
    )
    .reset_index()
)
demo_summary["completion_rate_pct"] = (
    demo_summary["completed_visits"] / demo_summary["total_visits"] * 100
).round(2)

demo_summary.to_csv(base / "clean/ab_test_demographics_summary.csv", index=False)
print(f"Exported: ab_test_demographics_summary.csv ({len(demo_summary)} rows)")

# Completion rate per client (matches notebook methodology)
completed_clients = (
    web[web["process_step"] == "confirm"]["client_id"]
    .unique()
)

client_level = (
    web.drop_duplicates("client_id")[
        ["client_id", "Variation", "age_group", "gender",
         "clnt_tenure_yr", "clnt_age"]
    ].copy()
)

client_level["completed"] = client_level["client_id"].isin(completed_clients).astype(int)

completion_by_age = (
    client_level.groupby(["Variation", "age_group"])["completed"]
    .mean()
    .reset_index()
    .rename(columns={"completed": "completion_rate"})
)
completion_by_age = completion_by_age[completion_by_age["age_group"] != "Unknown"]
completion_by_age.to_csv(base / "clean/ab_test_completion_by_age.csv", index=False)
completion_by_age["age_group"] = pd.Categorical(
    completion_by_age["age_group"], 
    categories=age_order, 
    ordered=True
)
completion_by_age = completion_by_age.sort_values("age_group")
print("Exported: ab_test_completion_by_age.csv")

# ─────────────────────────────────────────────
# EXPORT: Overall KPI summary
# ─────────────────────────────────────────────
completed_clients = web[web["process_step"] == "confirm"]["client_id"].unique()

client_level = web.drop_duplicates("client_id")[["client_id", "Variation"]].copy()
client_level["completed"] = client_level["client_id"].isin(completed_clients).astype(int)

kpi_summary = (
    client_level.groupby("Variation")["completed"]
    .mean()
    .reset_index()
    .rename(columns={"completed": "completion_rate"})
)

kpi_summary["completion_rate_pct"] = (kpi_summary["completion_rate"] * 100).round(2)

print("\nKPI Summary:")
print(kpi_summary)

kpi_summary.to_csv(base / "clean/ab_test_kpi_summary.csv", index=False)
print("Exported: ab_test_kpi_summary.csv")

# ─────────────────────────────────────────────
# EXPORT: Error rates for Tableau
# ─────────────────────────────────────────────
steps = ['start', 'step_1', 'step_2', 'step_3', 'confirm']
step_order_map = {s: i for i, s in enumerate(steps)}

def detect_errors(group):
    group = group.sort_values('date_time')
    sequence = group['process_step'].tolist()
    errors = []

    if 'start' not in sequence:
        errors.append('missing_start')
    if 'confirm' not in sequence:
        errors.append('missing_confirm')

    visited = [s for s in steps if s in sequence]
    for i in range(len(visited) - 1):
        expected_next = steps[steps.index(visited[i]) + 1] if steps.index(visited[i]) + 1 < len(steps) else None
        if expected_next and visited[i+1] != expected_next:
            errors.append('skipped_step')

    for i in range(len(sequence) - 1):
        curr = sequence[i]
        nxt = sequence[i+1]
        if curr in step_order_map and nxt in step_order_map:
            if step_order_map[nxt] < step_order_map[curr]:
                errors.append(f'went_back:{curr}_to_{nxt}')

    for step in steps:
        if sequence.count(step) > 1:
            errors.append(f'repeated:{step}')

    return errors

# Apply to all clients in experiment
error_records = []
for client_id, group in web.groupby('client_id'):
    errors = detect_errors(group)
    variation = group['Variation'].iloc[0]
    for e in errors:
        error_records.append({
            'client_id': client_id,
            'error': e,
            'Variation': variation
        })

error_df = pd.DataFrame(error_records)

# Simplify error categories for Tableau
error_df['error_category'] = error_df['error'].apply(lambda x:
    'Went Back' if x.startswith('went_back') else
    'Repeated Step' if x.startswith('repeated') else
    'Skipped Step' if x == 'skipped_step' else
    'Missing Start' if x == 'missing_start' else
    'Never Completed' if x == 'missing_confirm' else x
)

# Error rate per variation and category
total_clients = web.groupby('Variation')['client_id'].nunique().reset_index()
total_clients.columns = ['Variation', 'total_clients']

error_summary = (
    error_df.groupby(['Variation', 'error_category'])['client_id']
    .nunique()
    .reset_index()
    .rename(columns={'client_id': 'clients_with_error'})
)

error_summary = error_summary.merge(total_clients, on='Variation')
error_summary['error_rate'] = (
    error_summary['clients_with_error'] / error_summary['total_clients']
).round(4)

print("\nError Summary:")
print(error_summary.to_string())

error_summary.to_csv(base / "clean/ab_test_error_summary.csv", index=False)
print("Exported: ab_test_error_summary.csv")
# ─────────────────────────────────────────────
# 13. QUICK SANITY CHECK PRINTOUT
# ─────────────────────────────────────────────
print("\n──────────────────────────────────────")
print("SANITY CHECK")
print("──────────────────────────────────────")
print("\nGroup split:")
print(web.groupby("Variation")["client_id"].nunique())

print("\nFunnel (% reaching each step):")
print(funnel[["Variation","process_step","pct_of_starters"]].to_string(index=False))

print("\nOverall completion rate by group:")
completion = (
    web.groupby("Variation")
    .agg(total=("visit_id","nunique"), completed=("visit_completed","sum"))
    .assign(rate=lambda d: (d["completed"]/d["total"]*100).round(2))
)
print(completion)