import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


PROCESSED_DIR = os.path.join("backend", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

BASE_PATH = os.path.join(
    PROCESSED_DIR,
    "V5_risk_scored_dataset.csv"
)

OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "V6_recommendations.csv"
)


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


def load_data(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        if not os.path.exists(BASE_PATH):
            raise FileNotFoundError(
                f"Risk scored dataset not found at: {BASE_PATH}. "
                "Please upload a company dataset and run the full pipeline first."
            )
        df = pd.read_csv(BASE_PATH)

    required = {
        "risk_category",
        "burnout_category",
        "anomaly_status",
        "attrition_score",
        "burnout_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing required recommendation columns: {missing}. "
            "Please run risk detection before recommendations."
        )

    df = df.copy()

    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].apply(
            lambda x: 1 if str(x).strip().lower() in ["yes", "true", "1"] else 0
        )
    else:
        df["OverTime"] = 0

    return df.reset_index(drop=True)


def safe_get_number(row: pd.Series, col: str, default: float = 0):
    value = row.get(col, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def compute_priority_score(row: pd.Series) -> float:
    score = 0.0

    score += {
        "High Risk": 60,
        "Medium Risk": 35,
        "Low Risk": 10
    }.get(row.get("risk_category", "Low Risk"), 10)

    score += {
        "Severe Burnout": 45,
        "Moderate Burnout": 25,
        "Low Burnout": 10,
        "No Burnout": 5
    }.get(row.get("burnout_category", "Low Burnout"), 10)

    if row.get("anomaly_status") == "Anomalous Employee":
        score += 20

    score += safe_get_number(row, "attrition_score") * 0.25
    score += safe_get_number(row, "burnout_score") * 0.25

    income = safe_get_number(row, "MonthlyIncome", 5000)

    if income < 3000:
        score += 10
    elif income > 10000:
        score -= 5

    tenure = safe_get_number(row, "YearsAtCompany", 1)

    if tenure < 2:
        score += 10
    elif tenure > 10:
        score += 5

    if row.get("OverTime") == 1:
        score += 8

    years_no_promo = safe_get_number(row, "YearsSinceLastPromotion", 0)

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


def build_profile_matrix(df: pd.DataFrame) -> np.ndarray:
    available = [
        col for col in PROFILE_FEATURES
        if col in df.columns
    ]

    if not available:
        available = [
            "attrition_score",
            "burnout_score",
            "OverTime"
        ]

    feature_df = df[available].copy()

    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        median_value = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(
            median_value if pd.notna(median_value) else 0
        )

    nunique = feature_df.nunique()
    feature_df = feature_df[nunique[nunique > 1].index]

    if feature_df.empty:
        feature_df = pd.DataFrame({
            "attrition_score": df["attrition_score"],
            "burnout_score": df["burnout_score"],
        })

    scaler = MinMaxScaler()
    return scaler.fit_transform(feature_df)


def find_healthy_peers(
    profile_matrix: np.ndarray,
    df: pd.DataFrame,
    employee_idx: int,
    top_n: int = 5
) -> pd.DataFrame:
    if len(df) <= 1:
        return pd.DataFrame()

    emp_vec = profile_matrix[employee_idx].reshape(1, -1)
    sim_scores = cosine_similarity(emp_vec, profile_matrix)[0]

    temp = df.copy()
    temp["_sim"] = sim_scores
    temp = temp[temp.index != employee_idx]

    healthy = temp[
        (temp["risk_category"] == "Low Risk") &
        (temp["burnout_category"].isin(["Low Burnout", "No Burnout"]))
    ]

    if len(healthy) >= top_n:
        return healthy.nlargest(top_n, "_sim")

    if not healthy.empty:
        return healthy.nlargest(min(top_n, len(healthy)), "_sim")

    return temp.nlargest(min(top_n, len(temp)), "_sim")


def build_peer_insights(peers: pd.DataFrame) -> dict:
    if peers.empty:
        return {}

    def avg(col, default=None):
        if col not in peers.columns:
            return default
        return round(pd.to_numeric(peers[col], errors="coerce").mean(), 2)

    return {
        "peer_avg_income": avg("MonthlyIncome"),
        "peer_avg_satisfaction": avg("JobSatisfaction"),
        "peer_avg_wlb": avg("WorkLifeBalance"),
        "peer_avg_training": avg("TrainingTimesLastYear"),
        "peer_avg_years_promo": avg("YearsSinceLastPromotion"),
        "peer_avg_attrition": avg("attrition_score"),
        "peer_count": int(len(peers)),
    }


def generate_gap_warnings(row: pd.Series, peers: dict) -> list:
    if not peers:
        return []

    warnings = []

    if peers.get("peer_avg_income") is not None:
        income_gap = peers["peer_avg_income"] - safe_get_number(row, "MonthlyIncome", 0)
        if income_gap > 1500:
            warnings.append(
                f"Income is ${income_gap:,.0f} below similar healthy peers — compensation gap may increase attrition risk"
            )

    if peers.get("peer_avg_satisfaction") is not None:
        sat_gap = peers["peer_avg_satisfaction"] - safe_get_number(row, "JobSatisfaction", 3)
        if sat_gap > 0.8:
            warnings.append(
                f"Job satisfaction is {sat_gap:.1f} points below healthy peers — engagement intervention is recommended"
            )

    if peers.get("peer_avg_wlb") is not None:
        wlb_gap = peers["peer_avg_wlb"] - safe_get_number(row, "WorkLifeBalance", 3)
        if wlb_gap > 0.8:
            warnings.append(
                f"Work-life balance is {wlb_gap:.1f} points below healthy peers — schedule flexibility should be reviewed"
            )

    if peers.get("peer_avg_years_promo") is not None:
        promo_lag = safe_get_number(row, "YearsSinceLastPromotion", 0) - peers["peer_avg_years_promo"]
        if promo_lag > 2:
            warnings.append(
                f"{promo_lag:.0f} more years without promotion compared with healthy peers — career stagnation risk detected"
            )

    if peers.get("peer_avg_training") is not None:
        train_gap = peers["peer_avg_training"] - safe_get_number(row, "TrainingTimesLastYear", 0)
        if train_gap > 1.5:
            warnings.append(
                f"Training is {train_gap:.1f} sessions/year below healthy peers — learning and development gap identified"
            )

    return warnings


def generate_hr_actions(row: pd.Series) -> list:
    actions = []

    score = safe_get_number(row, "priority_score", 0)
    danger = str(row.get("danger_explanation", "")).lower()
    role = str(row.get("JobRole", ""))

    if score >= 120:
        actions.append("Schedule urgent retention meeting within 48 hours.")
        actions.append("Assign dedicated HR case manager for personalized intervention.")
        actions.append("Escalate case to HR leadership for immediate review.")
    elif score >= 80:
        actions.append("Start weekly engagement check-ins.")
        actions.append("Build personalized career development plan.")
    elif score >= 50:
        actions.append("Schedule monthly well-being review and add employee to 90-day HR watchlist.")
    else:
        actions.append("Maintain standard engagement monitoring with quarterly feedback.")

    burnout = row.get("burnout_category", "")

    if burnout == "Severe Burnout":
        actions.append("Apply immediate workload reduction and wellness recovery plan.")
        actions.append("Refer employee to professional support or EAP resources.")
    elif burnout == "Moderate Burnout":
        actions.append("Introduce flexible scheduling and structured rest periods.")
        actions.append("Provide stress management and wellness resources.")

    if row.get("anomaly_status") == "Anomalous Employee":
        actions.append("Review attendance, workload, and performance logs to identify unusual patterns.")

    if "low job satisfaction" in danger:
        actions.append("Conduct confidential satisfaction review and explore role enrichment.")
    if "poor work-life balance" in danger:
        actions.append("Review overtime patterns and workload distribution.")
    if "low monthly income" in danger:
        actions.append("Trigger compensation benchmarking review.")
    if "performance needs attention" in danger:
        actions.append("Recommend targeted skills development or performance support plan.")

    if row.get("OverTime") == 1:
        actions.append("Flag overtime overload and discuss redistribution with line manager.")

    years_no_promo = safe_get_number(row, "YearsSinceLastPromotion", 0)

    if years_no_promo >= 4:
        actions.append(f"No promotion in {int(years_no_promo)} years — urgently review career progression roadmap.")
    elif years_no_promo >= 2:
        actions.append(f"No promotion in {int(years_no_promo)} years — discuss promotion eligibility.")

    attrition_score = safe_get_number(row, "attrition_score", 0)

    if attrition_score > 60:
        actions.append(f"Attrition score {attrition_score:.1f} — consider targeted retention incentive or development offer.")

    training = safe_get_number(row, "TrainingTimesLastYear", 3)

    if training == 0:
        actions.append("No training last year — enroll employee in mandatory learning plan.")
    elif training == 1:
        actions.append("Only one training session last year — increase learning participation.")

    if "Manager" in role:
        actions.append("Offer leadership coaching and delegation support.")
    elif "Sales" in role:
        actions.append("Review incentive structure and quota pressure.")
    elif "Scientist" in role or "Research" in role:
        actions.append("Review research workload and protect deep-work time.")
    elif "Technician" in role or "Laboratory" in role:
        actions.append("Review technical career path and growth opportunities.")

    unique_actions = []
    seen = set()

    for action in actions:
        if action not in seen:
            seen.add(action)
            unique_actions.append(action)

    return unique_actions


RETENTION_STRATEGIES = {
    ("High Risk", "Severe Burnout"): "Emergency Retention Protocol",
    ("High Risk", "Moderate Burnout"): "Urgent Retention Intervention",
    ("High Risk", "Low Burnout"): "Proactive Retention Plan",
    ("Medium Risk", "Severe Burnout"): "Burnout Recovery & Stabilisation",
    ("Medium Risk", "Moderate Burnout"): "Engagement & Recovery Programme",
    ("Medium Risk", "Low Burnout"): "Career Development Focus",
    ("Low Risk", "Severe Burnout"): "Workload Relief Initiative",
    ("Low Risk", "Moderate Burnout"): "Preventive Well-being Programme",
    ("Low Risk", "Low Burnout"): "Standard Engagement Maintenance",
}


def get_retention_strategy(risk: str, burnout: str) -> str:
    return RETENTION_STRATEGIES.get(
        (risk, burnout),
        "General HR Monitoring"
    )


def generate_recommendations(df: pd.DataFrame = None) -> pd.DataFrame:
    df = load_data(df)

    df["priority_score"] = df.apply(compute_priority_score, axis=1)
    df["priority"] = df["priority_score"].apply(priority_label)

    profile_matrix = build_profile_matrix(df)

    records = []

    for idx, row in df.iterrows():
        peers = find_healthy_peers(profile_matrix, df, idx, top_n=5)
        peer_info = build_peer_insights(peers)
        gap_warnings = generate_gap_warnings(row, peer_info)
        hr_actions = generate_hr_actions(row)

        strategy = get_retention_strategy(
            row.get("risk_category", "Low Risk"),
            row.get("burnout_category", "Low Burnout")
        )

        record = {
            "EmployeeNumber": row.get("EmployeeNumber", idx),
            "Age": row.get("Age"),
            "JobRole": row.get("JobRole"),
            "Department": row.get("Department"),
            "MonthlyIncome": row.get("MonthlyIncome"),
            "risk_category": row.get("risk_category"),
            "burnout_category": row.get("burnout_category"),
            "anomaly_status": row.get("anomaly_status"),
            "attrition_score": round(safe_get_number(row, "attrition_score"), 2),
            "burnout_score": round(safe_get_number(row, "burnout_score"), 2),
            "priority_score": row.get("priority_score"),
            "priority": row.get("priority"),
            "retention_strategy": strategy,
            "hr_actions": hr_actions,
            "action_count": len(hr_actions),
            "peer_gap_warnings": gap_warnings,
            "gap_count": len(gap_warnings),
            "peer_avg_income": peer_info.get("peer_avg_income"),
            "peer_avg_satisfaction": peer_info.get("peer_avg_satisfaction"),
            "peer_avg_wlb": peer_info.get("peer_avg_wlb"),
        }

        records.append(record)

    result = pd.DataFrame(records)

    priority_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    result["_rank"] = result["priority"].map(priority_order)

    result = (
        result
        .sort_values(["_rank", "attrition_score"], ascending=[True, False])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    final_cols = [
        col for col in OUTPUT_COLS
        if col in result.columns
    ]

    return result[final_cols]


def save_recommendations(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    save_df = df.copy()

    if "hr_actions" in save_df.columns:
        save_df["hr_actions"] = save_df["hr_actions"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else x
        )

    if "peer_gap_warnings" in save_df.columns:
        save_df["peer_gap_warnings"] = save_df["peer_gap_warnings"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else x
        )

    save_df.to_csv(output_path, index=False)
    return output_path


def generate_summary_report(recs: pd.DataFrame) -> dict:
    return {
        "total_employees": int(len(recs)),
        "critical_count": int((recs["priority"] == "Critical").sum()) if "priority" in recs.columns else 0,
        "high_count": int((recs["priority"] == "High").sum()) if "priority" in recs.columns else 0,
        "medium_count": int((recs["priority"] == "Medium").sum()) if "priority" in recs.columns else 0,
        "low_count": int((recs["priority"] == "Low").sum()) if "priority" in recs.columns else 0,
        "anomalous_employees": int((recs["anomaly_status"] == "Anomalous Employee").sum()) if "anomaly_status" in recs.columns else 0,
        "avg_priority_score": round(recs["priority_score"].mean(), 2) if "priority_score" in recs.columns else 0,
        "avg_attrition_score": round(recs["attrition_score"].mean(), 2) if "attrition_score" in recs.columns else 0,
        "avg_burnout_score": round(recs["burnout_score"].mean(), 2) if "burnout_score" in recs.columns else 0,
        "employees_with_peer_gaps": int((recs["gap_count"] > 0).sum()) if "gap_count" in recs.columns else 0,
        "avg_actions_per_employee": round(recs["action_count"].mean(), 2) if "action_count" in recs.columns else 0,
        "top_retention_strategy": recs["retention_strategy"].value_counts().idxmax() if "retention_strategy" in recs.columns and not recs.empty else None,
        "risk_distribution": recs["risk_category"].value_counts().to_dict() if "risk_category" in recs.columns else {},
        "burnout_distribution": recs["burnout_category"].value_counts().to_dict() if "burnout_category" in recs.columns else {},
        "priority_distribution": recs["priority"].value_counts().to_dict() if "priority" in recs.columns else {},
    }


if __name__ == "__main__":
    recs = generate_recommendations()
    path = save_recommendations(recs)
    print(f"Recommendations saved to: {path}")
    print(generate_summary_report(recs))