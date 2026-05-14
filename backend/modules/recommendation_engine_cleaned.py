import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PROFILE FEATURES (FOR NEW DATASET)
# =========================================================

PROFILE_FEATURES = [
    "Age",
    "DistanceFromHome",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "OverTime"
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data(df: pd.DataFrame):
    df = df.copy()

    # encode OverTime
    if df["OverTime"].dtype == object:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    # fill missing numeric
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df.reset_index(drop=True)


# =========================================================
# PRIORITY SCORE
# =========================================================

def compute_priority_score(row):
    score = 0

    if row.get("Attrition", 0) == 1:
        score += 50

    if row.get("OverTime", 0) == 1:
        score += 15

    if row.get("JobSatisfaction", 3) <= 2:
        score += 15

    if row.get("WorkLifeBalance", 3) <= 2:
        score += 15

    if row.get("EnvironmentSatisfaction", 3) <= 2:
        score += 10

    if row.get("MonthlyIncome", 0) < 3000:
        score += 15

    if row.get("YearsSinceLastPromotion", 0) >= 5:
        score += 15

    elif row.get("YearsSinceLastPromotion", 0) >= 3:
        score += 8

    if row.get("TotalWorkingYears", 0) <= 2:
        score += 10

    if row.get("TrainingTimesLastYear", 0) <= 1:
        score += 10

    return round(score, 2)


def priority_label(score):
    if score >= 100:
        return "Critical"
    elif score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"


# =========================================================
# PROFILE MATRIX
# =========================================================

def build_profile_matrix(df):
    features = [f for f in PROFILE_FEATURES if f in df.columns]

    scaler = MinMaxScaler()
    return scaler.fit_transform(df[features])


# =========================================================
# PEERS
# =========================================================

def find_healthy_peers(matrix, df, idx, top_n=5):
    vec = matrix[idx].reshape(1, -1)
    sim = cosine_similarity(vec, matrix)[0]

    temp = df.copy()
    temp["sim"] = sim
    temp = temp[temp.index != idx]

    healthy = temp[
        (temp["Attrition"] == 0) &
        (temp["JobSatisfaction"] >= 3) &
        (temp["WorkLifeBalance"] >= 3)
    ]

    if len(healthy) >= top_n:
        return healthy.nlargest(top_n, "sim")

    return temp.nlargest(top_n, "sim")


# =========================================================
# GAPS
# =========================================================

def build_peer_insights(peers):
    if peers.empty:
        return {}

    return {
        "peer_avg_income": peers["MonthlyIncome"].mean(),
        "peer_avg_satisfaction": peers["JobSatisfaction"].mean(),
        "peer_avg_wlb": peers["WorkLifeBalance"].mean(),
    }


def generate_gap_warnings(row, peer):
    warnings = []

    if not peer:
        return warnings

    if peer["peer_avg_income"] - row["MonthlyIncome"] > 1500:
        warnings.append("Income gap vs healthy peers")

    if peer["peer_avg_satisfaction"] - row["JobSatisfaction"] > 1:
        warnings.append("Low job satisfaction vs peers")

    if peer["peer_avg_wlb"] - row["WorkLifeBalance"] > 1:
        warnings.append("Poor work-life balance vs peers")

    return warnings


# =========================================================
# ACTIONS
# =========================================================

def generate_hr_actions(row):
    actions = []

    if row["Attrition"] == 1:
        actions.append("Immediate retention interview")

    if row["OverTime"] == 1:
        actions.append("Reduce overtime workload")

    if row["JobSatisfaction"] <= 2:
        actions.append("Improve role satisfaction")

    if row["WorkLifeBalance"] <= 2:
        actions.append("Improve work-life balance")

    if row["MonthlyIncome"] < 3000:
        actions.append("Review salary")

    if row["YearsSinceLastPromotion"] >= 5:
        actions.append("Promotion review needed")

    if not actions:
        actions.append("No urgent action")

    return actions


# =========================================================
# MAIN FUNCTION
# =========================================================

def generate_recommendations(df: pd.DataFrame):

    df = load_data(df)

    df["priority_score"] = df.apply(compute_priority_score, axis=1)
    df["priority"] = df["priority_score"].apply(priority_label)

    matrix = build_profile_matrix(df)

    records = []

    for i, row in df.iterrows():

        peers = find_healthy_peers(matrix, df, i)
        peer_info = build_peer_insights(peers)

        gaps = generate_gap_warnings(row, peer_info)
        actions = generate_hr_actions(row)

        records.append({
            "EmployeeNumber": row["EmployeeNumber"],
            "JobRole": row["JobRole"],
            "MonthlyIncome": row["MonthlyIncome"],
            "priority_score": row["priority_score"],
            "priority": row["priority"],
            "peer_gap_warnings": gaps,
            "action_count": len(actions),
            "hr_actions": actions,
        })

    return pd.DataFrame(records)