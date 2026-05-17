import os
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


PROCESSED_DIR = os.path.join("backend", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SEGMENTED_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "clustered_data.csv"
)

OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "V5_risk_scored_dataset.csv"
)


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


def is_overtime(value):
    value = str(value).strip().lower()
    return value in ["yes", "true", "1"]


def explain_risk(row):
    reasons = []

    if is_overtime(row.get("OverTime", "No")):
        reasons.append("works overtime")

    if row.get("JobSatisfaction", 3) <= 2:
        reasons.append("low job satisfaction")

    if row.get("WorkLifeBalance", 3) <= 2:
        reasons.append("poor work-life balance")

    if row.get("MonthlyIncome", 0) < 4000:
        reasons.append("low monthly income")

    if row.get("PerformanceRating", 3) <= 3:
        reasons.append("performance needs attention")

    if "cluster_label" in row.index:
        cluster_label = str(row.get("cluster_label", ""))

        if "Low Income" in cluster_label:
            reasons.append("belongs to a low-income employee cluster")

        if "Low Satisfaction" in cluster_label:
            reasons.append("belongs to a low-satisfaction employee cluster")

        if "Poor Work-Life Balance" in cluster_label:
            reasons.append("belongs to a poor work-life balance cluster")

    if not reasons:
        return "No major risk indicators detected."

    return "Employee risk is related to: " + ", ".join(reasons) + "."


def load_segmented_dataset():
    if not os.path.exists(SEGMENTED_DATA_PATH):
        raise FileNotFoundError(
            f"Segmented dataset not found at: {SEGMENTED_DATA_PATH}. "
            "Please upload a company dataset and run segmentation first."
        )

    df = pd.read_csv(SEGMENTED_DATA_PATH)

    if "cluster" not in df.columns:
        raise ValueError("The segmented dataset must contain a 'cluster' column.")

    if "cluster_label" not in df.columns:
        raise ValueError("The segmented dataset must contain a 'cluster_label' column.")

    return df


def prepare_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "MonthlyIncome": df["MonthlyIncome"].median() if "MonthlyIncome" in df.columns else 5000,
        "OverTime": "No",
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
        "PerformanceRating": 3,
        "YearsAtCompany": 1,
        "Attrition": 0,
        "cluster": 0,
        "cluster_label": "General Employee Segment",
    }

    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value

    numeric_cols = [
        "MonthlyIncome",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
        "YearsAtCompany",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value if pd.notna(median_value) else defaults[col])

    if df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({
            "Yes": 1,
            "No": 0,
            "yes": 1,
            "no": 0,
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
        }).fillna(0)

    return df


def scale_inverse_risk(df: pd.DataFrame, column: str):
    return 1 - MinMaxScaler().fit_transform(df[[column]]).flatten()


def scale_direct_risk(df: pd.DataFrame, column: str):
    return MinMaxScaler().fit_transform(df[[column]]).flatten()


def run_risk_detection(input_path: str = None, df: pd.DataFrame = None, contamination: float = 0.10):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if df is not None:
        result_df = df.copy()
        used_input_file = "uploaded company dataset from in-memory segmentation result"

    elif input_path is not None:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Provided input_path not found: {input_path}")
        result_df = pd.read_csv(input_path)
        used_input_file = input_path

    else:
        result_df = load_segmented_dataset()
        used_input_file = SEGMENTED_DATA_PATH

    result_df = prepare_required_columns(result_df)

    temp = pd.DataFrame(index=result_df.index)

    temp["salary_risk"] = scale_inverse_risk(result_df, "MonthlyIncome")

    temp["overtime_risk"] = result_df["OverTime"].apply(
        lambda x: 1 if is_overtime(x) else 0
    )

    temp["satisfaction_risk"] = scale_inverse_risk(result_df, "JobSatisfaction")
    temp["work_life_risk"] = scale_inverse_risk(result_df, "WorkLifeBalance")
    temp["performance_risk"] = scale_inverse_risk(result_df, "PerformanceRating")
    temp["years_risk"] = scale_direct_risk(result_df, "YearsAtCompany")

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
            "years_risk",
        ]
    ]

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    result_df["anomaly_flag"] = model.fit_predict(features)

    result_df["anomaly_status"] = result_df["anomaly_flag"].apply(
        lambda x: "Anomalous Employee" if x == -1 else "Normal Employee"
    )

    result_df["risk_category"] = result_df["attrition_score"].apply(classify_risk)
    result_df["burnout_category"] = result_df["burnout_score"].apply(burnout_category)
    result_df["danger_explanation"] = result_df.apply(explain_risk, axis=1)

    result_df.drop(columns=["anomaly_flag"], inplace=True, errors="ignore")

    result_df.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "total_employees": int(len(result_df)),
        "high_risk": int((result_df["risk_category"] == "High Risk").sum()),
        "medium_risk": int((result_df["risk_category"] == "Medium Risk").sum()),
        "low_risk": int((result_df["risk_category"] == "Low Risk").sum()),
        "severe_burnout": int((result_df["burnout_category"] == "Severe Burnout").sum()),
        "moderate_burnout": int((result_df["burnout_category"] == "Moderate Burnout").sum()),
        "low_burnout": int((result_df["burnout_category"] == "Low Burnout").sum()),
        "anomalies": int((result_df["anomaly_status"] == "Anomalous Employee").sum()),
        "used_input_file": used_input_file,
        "used_uploaded_company_data": True,
        "has_cluster_column": "cluster" in result_df.columns,
        "has_cluster_label_column": "cluster_label" in result_df.columns,
        "output_path": OUTPUT_PATH,
    }

    return result_df, summary