import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

PROCESSED_DIR = "data/processed"


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


def run_risk_detection(input_path=None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if input_path is None:
        input_path = get_latest_cleaned_file()

    df = pd.read_csv(input_path)
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

    temp["salary_risk"] = 1 - MinMaxScaler().fit_transform(df[["MonthlyIncome"]]).flatten()
    temp["overtime_risk"] = df["OverTime"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)
    temp["satisfaction_risk"] = 1 - MinMaxScaler().fit_transform(df[["JobSatisfaction"]]).flatten()
    temp["work_life_risk"] = 1 - MinMaxScaler().fit_transform(df[["WorkLifeBalance"]]).flatten()
    temp["performance_risk"] = 1 - MinMaxScaler().fit_transform(df[["PerformanceRating"]]).flatten()
    temp["years_risk"] = MinMaxScaler().fit_transform(df[["YearsAtCompany"]]).flatten()

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
    result_df["burnout_category"] = result_df["burnout_score"].apply(burnout_category)

    result_df["danger_explanation"] = result_df.apply(
        lambda row: explain_risk(row),
        axis=1
    )

    output_path = os.path.join(PROCESSED_DIR, "V3_risk_scored_dataset.csv")
    result_df.to_csv(output_path, index=False)

    summary = {
        "total_employees": int(len(result_df)),
        "high_risk": int((result_df["risk_category"] == "High Risk").sum()),
        "medium_risk": int((result_df["risk_category"] == "Medium Risk").sum()),
        "low_risk": int((result_df["risk_category"] == "Low Risk").sum()),
        "severe_burnout": int((result_df["burnout_category"] == "Severe Burnout").sum()),
        "anomalies": int((result_df["anomaly_status"] == "Anomalous Employee").sum()),
        "output_path": output_path
    }

    return result_df, summary


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

    if not reasons:
        return "No major risk indicators detected."

    return "Employee risk is related to: " + ", ".join(reasons) + "."