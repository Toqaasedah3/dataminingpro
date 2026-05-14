import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_PATH = os.path.join(BASE_DIR, "data", "processed", "V5_risk_scored_dataset.csv")

# Features used to build each employee's content profile vector
PROFILE_FEATURES = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "WorkLifeBalance",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "JobInvolvement",
    "PerformanceRating",
    "OverTime",
    "NumCompaniesWorked",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "TrainingTimesLastYear",
    "attrition_score",
    "burnout_score",
]

# Output columns to return
OUTPUT_COLS = [
    "EmployeeNumber",
    "Age",
    "JobRole",
    "Department",
    "MonthlyIncome",
    "risk_category",
    "burnout_category",
    "anomaly_status",
    "attrition_score",
    "burnout_score",
    "priority_score",
    "priority",
    "retention_strategy",
    "hr_actions",
    "peer_gap_warnings",
    "action_count",
    "gap_count",
    "peer_avg_income",
    "peer_avg_satisfaction",
    "peer_avg_wlb",
]


# STEP 1 — LOAD & VALIDATE
def load_data(df: pd.DataFrame = None) -> pd.DataFrame:
    """Load dataset from memory (API/Streamlit) or from file."""
    if df is None:
        if not os.path.exists(BASE_PATH):
            raise FileNotFoundError(f"Dataset not found at: {BASE_PATH}")
        df = pd.read_csv(BASE_PATH)

    required = {
        "risk_category", "burnout_category",
        "anomaly_status", "attrition_score", "burnout_score"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Encode OverTime if string
    if "OverTime" in df.columns and df["OverTime"].dtype == object:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    return df.reset_index(drop=True)


# STEP 2 — WEIGHTED PRIORITY SCORE
def compute_priority_score(row: pd.Series) -> float:
    """
    Compute a numeric urgency score for each employee.
    Higher = more urgent HR attention needed.
    This drives the priority tier and sorting.
    """
    score = 0.0

    # Risk tier base score
    score += {"High Risk": 60, "Medium Risk": 35, "Low Risk": 10}.get(
        row.get("risk_category", "Low Risk"), 10
    )

    # Burnout tier
    score += {"Severe Burnout": 45, "Moderate Burnout": 25, "Low Burnout": 10}.get(
        row.get("burnout_category", "Low Burnout"), 10
    )

    # Anomaly flag
    if row.get("anomaly_status") == "Anomalous Employee":
        score += 20

    # Continuous scores
    score += row.get("attrition_score", 0) * 0.25
    score += row.get("burnout_score", 0)   * 0.25

    # Income penalty
    income = row.get("MonthlyIncome", 0)
    if income < 3000:
        score += 10
    elif income > 10000:
        score -= 5

    # Tenure signals
    tenure = row.get("YearsAtCompany", 0)
    if tenure < 2:
        score += 10   # new employees are flight risks
    elif tenure > 10:
        score += 5    # long tenured but still at risk = serious signal

    # Overtime flag
    if row.get("OverTime") == 1:
        score += 8

    # Promotion stagnation
    years_no_promo = row.get("YearsSinceLastPromotion", 0)
    if years_no_promo >= 4:
        score += 10
    elif years_no_promo >= 2:
        score += 5

    return round(score, 2)


def priority_label(score: float) -> str:
    if score >= 120:
        return "Critical"
    elif score >= 80:
        return "High"
    elif score >= 50:
        return "Medium"
    return "Low"



# STEP 3 — CONTENT-BASED PROFILE MATRIX
def build_profile_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Normalise PROFILE_FEATURES into a [0,1] matrix.
    Each row = one employee's content vector.
    This is what makes it a Content-Based system.
    """
    available = [f for f in PROFILE_FEATURES if f in df.columns]
    feature_df = df[available].copy()
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    scaler = MinMaxScaler()
    return scaler.fit_transform(feature_df)


# STEP 4 — FIND SIMILAR HEALTHY PEERS
def find_healthy_peers(
    profile_matrix: np.ndarray,
    df: pd.DataFrame,
    employee_idx: int,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Find the top_n employees most SIMILAR to the target employee
    but who are in a HEALTHY state (Low Risk + Low/No Burnout).

    These are the "reference points" — employees like this one
    who are doing well. The gap between them and the at-risk
    employee surfaces the root causes.
    """
    emp_vec = profile_matrix[employee_idx].reshape(1, -1)
    sim_scores = cosine_similarity(emp_vec, profile_matrix)[0]

    temp = df.copy()
    temp["_sim"] = sim_scores

    # Exclude the employee themselves
    temp = temp[temp.index != employee_idx]

    # Prefer healthy peers as reference
    healthy = temp[
        (temp["risk_category"] == "Low Risk") &
        (temp["burnout_category"].isin(["Low Burnout", "No Burnout"]))
    ]

    if len(healthy) >= top_n:
        return healthy.nlargest(top_n, "_sim")

    # Fallback: use all employees if not enough healthy peers found
    return temp.nlargest(top_n, "_sim")


# STEP 5 — PEER INSIGHTS & GAP WARNINGS
def build_peer_insights(peers: pd.DataFrame) -> dict:
    """Summarise what the healthy reference peers look like."""
    if peers.empty:
        return {}
    return {
        "peer_avg_income":        round(peers["MonthlyIncome"].mean(), 2),
        "peer_avg_satisfaction":  round(peers["JobSatisfaction"].mean(), 2),
        "peer_avg_wlb":           round(peers["WorkLifeBalance"].mean(), 2),
        "peer_avg_training":      round(peers["TrainingTimesLastYear"].mean(), 2),
        "peer_avg_years_promo":   round(peers["YearsSinceLastPromotion"].mean(), 2),
        "peer_avg_attrition":     round(peers["attrition_score"].mean(), 2),
        "peer_count":             len(peers),
    }


def generate_gap_warnings(row: pd.Series, peers: dict) -> list:
    """
    Compare the employee's actual stats to their healthy peers.
    This is the core of the content-based insight:
    'Employees similar to you, but stable, have X. You have Y.'
    """
    if not peers:
        return []

    warnings = []

    income_gap = peers["peer_avg_income"] - row.get("MonthlyIncome", 0)
    if income_gap > 1500:
        warnings.append(
            f"Income is ${income_gap:,.0f} below similar healthy peers — "
            "compensation gap is a likely attrition driver"
        )

    sat_gap = peers["peer_avg_satisfaction"] - row.get("JobSatisfaction", 3)
    if sat_gap > 0.8:
        warnings.append(
            f"Job satisfaction is {sat_gap:.1f} pts below healthy peers — "
            "role engagement intervention needed"
        )

    wlb_gap = peers["peer_avg_wlb"] - row.get("WorkLifeBalance", 3)
    if wlb_gap > 0.8:
        warnings.append(
            f"Work-life balance is {wlb_gap:.1f} pts below healthy peers — "
            "schedule flexibility should be explored"
        )

    promo_lag = (
        row.get("YearsSinceLastPromotion", 0) - peers["peer_avg_years_promo"]
    )
    if promo_lag > 2:
        warnings.append(
            f"{promo_lag:.0f} more years without promotion vs healthy peers — "
            "career stagnation risk detected"
        )

    train_gap = peers["peer_avg_training"] - row.get("TrainingTimesLastYear", 0)
    if train_gap > 1.5:
        warnings.append(
            f"Training is {train_gap:.1f} sessions/year below healthy peers — "
            "L&D investment gap identified"
        )

    return warnings


# STEP 6 — HR ACTION ENGINE
def generate_hr_actions(row: pd.Series) -> list:
    """
    Generate specific, actionable HR recommendations.
    Targets root causes using:
    - Priority score tier
    - Burnout category
    - Anomaly flag
    - danger_explanation column (Tala's output)
    - Job role
    - Quantitative signals (promotion lag, overtime, training)
    """
    actions = []
    score  = row.get("priority_score", 0)
    danger = str(row.get("danger_explanation", "")).lower()
    role   = str(row.get("JobRole", ""))

    # Priority tier
    if score >= 120:
        actions.append(
            "Schedule URGENT retention meeting within 48 hours — "
            "deep discussion on satisfaction, workload, and career path"
        )
        actions.append(
            "Assign dedicated HR case manager for personalised intervention plan"
        )
        actions.append(
            "Escalate to HR leadership for immediate strategic review"
        )
    elif score >= 80:
        actions.append(
            "Initiate structured weekly engagement check-ins to detect "
            "dissatisfaction signals early"
        )
        actions.append(
            "Build personalised career development plan aligned with employee goals"
        )
    elif score >= 50:
        actions.append(
            "Schedule monthly one-on-one well-being review — "
            "add to 90-day HR watchlist"
        )
    else:
        actions.append(
            "Maintain standard engagement monitoring with quarterly feedback sessions"
        )

    # Burnout 
    burnout = row.get("burnout_category", "")
    if burnout == "Severe Burnout":
        actions.append(
            "Implement IMMEDIATE workload reduction (up to 50%) "
            "and enrol in mandatory wellness recovery programme"
        )
        actions.append(
            "Refer employee to professional mental health support (EAP) without delay"
        )
    elif burnout == "Moderate Burnout":
        actions.append(
            "Introduce flexible scheduling and structured rest periods "
            "to restore work-life balance"
        )
        actions.append(
            "Provide access to stress management workshops and mindfulness resources"
        )

    # Anomaly
    if row.get("anomaly_status") == "Anomalous Employee":
        actions.append(
            "Conduct behavioural analysis — review attendance, output logs, "
            "and cross-check peer performance data to identify root cause"
        )

    # Root-cause targeting
    if "low job satisfaction" in danger:
        actions.append(
            "Conduct confidential job satisfaction survey — "
            "explore role enrichment or lateral move opportunities"
        )
    if "poor work-life balance" in danger:
        actions.append(
            "Review overtime patterns — enforce WLB policy if chronically exceeded; "
            "offer remote/hybrid option where feasible"
        )
    if "low monthly income" in danger:
        actions.append(
            "Trigger compensation benchmarking review — "
            "evaluate eligibility for next salary band promotion"
        )
    if "performance needs attention" in danger:
        actions.append(
            "Enrol in targeted skills development programme; "
            "initiate Performance Improvement Plan (PIP) if below threshold"
        )

    # Quantitative signals
    if row.get("OverTime") == 1:
        actions.append(
            "Flag overtime overload — discuss workload redistribution with line manager"
        )

    years_no_promo = row.get("YearsSinceLastPromotion", 0)
    if years_no_promo >= 4:
        actions.append(
            f"No promotion in {int(years_no_promo)} years — "
            "urgently review career progression roadmap and set clear milestones"
        )
    elif years_no_promo >= 2:
        actions.append(
            f"No promotion in {int(years_no_promo)} years — "
            "discuss promotion eligibility at next review"
        )

    attrition_score = row.get("attrition_score", 0)
    if attrition_score > 60:
        actions.append(
            f"Attrition score {attrition_score:.1f} — "
            "consider targeted retention bonus or fast-track development offer"
        )

    training = row.get("TrainingTimesLastYear", 3)
    if training == 0:
        actions.append(
            "Zero training sessions last year — "
            "enrol in mandatory L&D programme immediately"
        )
    elif training == 1:
        actions.append(
            "Only 1 training session last year — "
            "increase L&D participation to at least 3 sessions/year"
        )

    # Role-specific actions
    if "Manager" in role:
        actions.append(
            "Enrol in executive leadership coaching to strengthen "
            "team management and delegation skills"
        )
    elif "Sales" in role:
        actions.append(
            "Review incentive structure and quota targets — "
            "ensure commission model is motivating, not demotivating"
        )
    elif "Scientist" in role or "Research" in role:
        actions.append(
            "Audit research project load to reduce cognitive overload "
            "and protect deep-work time"
        )
    elif "Technician" in role or "Laboratory" in role:
        actions.append(
            "Review technical role growth path — "
            "consider specialist track or team lead opportunities"
        )

    # Deduplicate while preserving order
    seen, unique = set(), []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique.append(a)

    return unique if unique else [
        "No immediate action required — continue standard monitoring"
    ]


# STEP 7 — RETENTION STRATEGY LABELS

RETENTION_STRATEGIES = {
    ("High Risk",   "Severe Burnout"):   "Emergency Retention Protocol",
    ("High Risk",   "Moderate Burnout"): "Urgent Retention Intervention",
    ("High Risk",   "Low Burnout"):      "Proactive Retention Plan",
    ("Medium Risk", "Severe Burnout"):   "Burnout Recovery & Stabilisation",
    ("Medium Risk", "Moderate Burnout"): "Engagement & Recovery Programme",
    ("Medium Risk", "Low Burnout"):      "Career Development Focus",
    ("Low Risk",    "Severe Burnout"):   "Workload Relief Initiative",
    ("Low Risk",    "Moderate Burnout"): "Preventive Well-being Programme",
    ("Low Risk",    "Low Burnout"):      "Standard Engagement Maintenance",
}

def get_retention_strategy(risk: str, burnout: str) -> str:
    return RETENTION_STRATEGIES.get(
        (risk, burnout), "General HR Monitoring"
    )


# STEP 8 — MAIN PIPELINE


def generate_recommendations(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full content-based HR recommendation pipeline.

    Steps:
      1. Load and validate dataset
      2. Compute weighted priority score per employee
      3. Build normalised content profile matrix
      4. For each employee → find similar healthy peers
      5. Compute peer-gap warnings (content-based insight)
      6. Generate targeted HR actions
      7. Assign retention strategy label
      8. Sort by priority score descending
      9. Return structured DataFrame

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pass directly from Streamlit/API.
        If None, loads from BASE_PATH.

    Returns
    -------
    pd.DataFrame — one recommendation record per employee,
    sorted by urgency (most critical first).
    """

    df = load_data(df)

    # ── Priority scores ────────────────────────────────────
    df["priority_score"] = df.apply(compute_priority_score, axis=1)
    df["priority"]       = df["priority_score"].apply(priority_label)

    # ── Content-based profile matrix ───────────────────────
    profile_matrix = build_profile_matrix(df)

    records = []

    for idx, row in df.iterrows():

        # Find similar healthy peers
        peers        = find_healthy_peers(profile_matrix, df, idx, top_n=5)
        peer_info    = build_peer_insights(peers)
        gap_warnings = generate_gap_warnings(row, peer_info)

        # HR actions (uses priority_score already in row) 
        hr_actions = generate_hr_actions(row)

        # Retention strategy
        strategy = get_retention_strategy(
            row.get("risk_category", "Low Risk"),
            row.get("burnout_category", "Low Burnout")
        )

        record = {
            "EmployeeNumber":    row.get("EmployeeNumber", idx),
            "Age":               row.get("Age"),
            "JobRole":           row.get("JobRole"),
            "Department":        row.get("Department"),
            "MonthlyIncome":     row.get("MonthlyIncome"),

            "risk_category":     row.get("risk_category"),
            "burnout_category":  row.get("burnout_category"),
            "anomaly_status":    row.get("anomaly_status"),
            "attrition_score":   round(row.get("attrition_score", 0), 2),
            "burnout_score":     round(row.get("burnout_score", 0), 2),

            "priority_score":    row.get("priority_score"),
            "priority":          row.get("priority"),
            "retention_strategy":strategy,

            "hr_actions":        hr_actions,
            "action_count":      len(hr_actions),

            "peer_gap_warnings": gap_warnings,
            "gap_count":         len(gap_warnings),
            "peer_avg_income":   peer_info.get("peer_avg_income"),
            "peer_avg_satisfaction": peer_info.get("peer_avg_satisfaction"),
            "peer_avg_wlb":      peer_info.get("peer_avg_wlb"),
        }

        records.append(record)

    result = pd.DataFrame(records)

    # Sort: Critical first, then by attrition score
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    result["_rank"] = result["priority"].map(priority_order)
    result = (
        result
        .sort_values(["_rank", "attrition_score"], ascending=[True, False])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    # Return only defined output columns that exist
    final_cols = [c for c in OUTPUT_COLS if c in result.columns]
    return result[final_cols]



# STEP 9 — SUMMARY REPORT (for dashboard)

def generate_summary_report(recs: pd.DataFrame) -> dict:
    """Aggregate stats for the Streamlit dashboard."""
    return {
        "total_employees":           len(recs),
        "critical_count":            int((recs["priority"] == "Critical").sum()),
        "high_count":                int((recs["priority"] == "High").sum()),
        "medium_count":              int((recs["priority"] == "Medium").sum()),
        "low_count":                 int((recs["priority"] == "Low").sum()),
        "anomalous_employees":       int((recs["anomaly_status"] == "Anomalous Employee").sum()),
        "avg_priority_score":        round(recs["priority_score"].mean(), 2),
        "avg_attrition_score":       round(recs["attrition_score"].mean(), 2),
        "avg_burnout_score":         round(recs["burnout_score"].mean(), 2),
        "employees_with_peer_gaps":  int((recs["gap_count"] > 0).sum()),
        "avg_actions_per_employee":  round(recs["action_count"].mean(), 2),
        "top_retention_strategy":    recs["retention_strategy"].value_counts().idxmax(),
        "risk_distribution":         recs["risk_category"].value_counts().to_dict(),
        "burnout_distribution":      recs["burnout_category"].value_counts().to_dict(),
        "priority_distribution":     recs["priority"].value_counts().to_dict(),
    }


# QUICK TEST

if __name__ == "__main__":
    print("=" * 55)
    print("  TalentGuard — HR Recommendation Engine Test")
    print("=" * 55)

    try:
        recs = generate_recommendations()
        print(f"\n Recommendations generated for {len(recs)} employees\n")

        # Show top Critical employee
        critical = recs[recs["priority"] == "Critical"]
        if not critical.empty:
            s = critical.iloc[0]
            print("── Top Critical Employee ──────────────────────────")
            print(f"  ID:               {s['EmployeeNumber']}")
            print(f"  Role:             {s.get('JobRole', 'N/A')}")
            print(f"  Risk:             {s['risk_category']}")
            print(f"  Burnout:          {s['burnout_category']}")
            print(f"  Priority Score:   {s['priority_score']}")
            print(f"  Attrition Score:  {s['attrition_score']}")
            print(f"  Strategy:         {s['retention_strategy']}")
            print(f"\n  HR Actions ({s['action_count']}):")
            for a in s["hr_actions"]:
                print(f"    → {a}")
            print(f"\n  Peer Gap Warnings ({s['gap_count']}):")
            for w in s["peer_gap_warnings"]:
                print(f"     {w}")

        print("\n── Summary Report ─────────────────────────────────")
        summary = generate_summary_report(recs)
        for k, v in summary.items():
            print(f"  {k}: {v}")

    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("   Run after Toqa & Tala's pipeline generates V5.")