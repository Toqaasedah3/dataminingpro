import os
import glob
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

PROCESSED_DIR = "data/processed"
CLUSTERED_PATH = "data/processed/clustered_data.csv"


def get_latest_cleaned_file():
    files = glob.glob("data/cleaned/cleaned_dataset_*.csv")

    if not files:
        raise FileNotFoundError("No cleaned dataset found in data/cleaned")

    return max(files, key=os.path.getctime)


def classify_risk(score):
    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    return "Low Risk"


def burnout_category(score):
    if score >= 70:
        return "Severe Burnout"
    elif score >= 40:
        return "Moderate Burnout"
    return "Low Burnout"


def explain_risk(row):
    reasons = []

    if str(row["OverTime"]).lower() == "yes":
        reasons.append("works overtime")

    if row["JobSatisfaction"] <= 2:
        reasons.append("low job satisfaction")

    if row["WorkLifeBalance"] <= 2:
        reasons.append("poor work-life balance")

    if row["MonthlyIncome"] < 4000:
        reasons.append("low monthly income")

    if row["PerformanceRating"] <= 3:
        reasons.append("performance needs attention")

    if "cluster_label" in row.index:
        cluster_label = str(row["cluster_label"])

        if "Low Income" in cluster_label:
            reasons.append("belongs to a low-income employee cluster")

        if "Low Satisfaction" in cluster_label:
            reasons.append("belongs to a low-satisfaction employee cluster")

        if "Poor Work-Life Balance" in cluster_label:
            reasons.append("belongs to a poor work-life balance cluster")

    if not reasons:
        return "No major risk indicators detected."

    return "Employee risk is related to: " + ", ".join(reasons) + "."


def run_risk_detection(input_path=None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if input_path is not None:
        df = pd.read_csv(input_path)
        used_input_file = input_path
        used_clustered_data = "cluster" in df.columns or "cluster_label" in df.columns

    elif os.path.exists(CLUSTERED_PATH):
        df = pd.read_csv(CLUSTERED_PATH)
        used_input_file = CLUSTERED_PATH
        used_clustered_data = True

    else:
        cleaned_path = get_latest_cleaned_file()
        df = pd.read_csv(cleaned_path)
        used_input_file = cleaned_path
        used_clustered_data = False

    result_df = df.copy()

    required_cols = [
        "MonthlyIncome",
        "OverTime",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
        "YearsAtCompany",
        "Attrition"
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    temp = pd.DataFrame()

    temp["salary_risk"] = 1 - MinMaxScaler().fit_transform(
        df[["MonthlyIncome"]]
    ).flatten()

    temp["overtime_risk"] = df["OverTime"].apply(
        lambda x: 1 if str(x).lower() == "yes" else 0
    )

    temp["satisfaction_risk"] = 1 - MinMaxScaler().fit_transform(
        df[["JobSatisfaction"]]
    ).flatten()

    temp["work_life_risk"] = 1 - MinMaxScaler().fit_transform(
        df[["WorkLifeBalance"]]
    ).flatten()

    temp["performance_risk"] = 1 - MinMaxScaler().fit_transform(
        df[["PerformanceRating"]]
    ).flatten()

    temp["years_risk"] = MinMaxScaler().fit_transform(
        df[["YearsAtCompany"]]
    ).flatten()

    result_df["attrition_score"] = (
        temp["salary_risk"] * 20 +
        temp["overtime_risk"] * 25 +
        temp["satisfaction_risk"] * 25 +
        temp["work_life_risk"] * 15 +
        temp["performance_risk"] * 10 +
        temp["years_risk"] * 5
    ).round(2)

    result_df["burnout_score"] = (
        temp["overtime_risk"] * 40 +
        temp["work_life_risk"] * 30 +
        temp["satisfaction_risk"] * 20 +
        temp["performance_risk"] * 10
    ).round(2)

    features = temp[
        [
            "salary_risk",
            "overtime_risk",
            "satisfaction_risk",
            "work_life_risk",
            "performance_risk",
            "years_risk"
        ]
    ]

    model = IsolationForest(contamination=0.10, random_state=42)

    result_df["anomaly_flag"] = model.fit_predict(features)

    result_df["anomaly_status"] = result_df["anomaly_flag"].apply(
        lambda x: "Anomalous Employee" if x == -1 else "Normal Employee"
    )

    result_df["risk_category"] = result_df["attrition_score"].apply(classify_risk)

    result_df["burnout_category"] = result_df["burnout_score"].apply(
        burnout_category
    )

    result_df["danger_explanation"] = result_df.apply(
        lambda row: explain_risk(row),
        axis=1
    )

    output_path = os.path.join(PROCESSED_DIR, "V5_risk_scored_dataset.csv")
    result_df.to_csv(output_path, index=False)

    summary = {
        "total_employees": int(len(result_df)),
        "high_risk": int((result_df["risk_category"] == "High Risk").sum()),
        "medium_risk": int((result_df["risk_category"] == "Medium Risk").sum()),
        "low_risk": int((result_df["risk_category"] == "Low Risk").sum()),
        "severe_burnout": int(
            (result_df["burnout_category"] == "Severe Burnout").sum()
        ),
        "anomalies": int(
            (result_df["anomaly_status"] == "Anomalous Employee").sum()
        ),
        "used_input_file": used_input_file,
        "used_clustered_data": used_clustered_data,
        "has_cluster_column": "cluster" in result_df.columns,
        "has_cluster_label_column": "cluster_label" in result_df.columns,
        "output_path": output_path
    }

    return result_df, summary