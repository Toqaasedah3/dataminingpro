import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Risk Dashboard", layout="wide")

DATA_PATH = "data/processed/V3_risk_scored_dataset.csv"

st.title("Employee Risk Intelligence Dashboard")
st.write("Risk Detection, Burnout Intelligence, Attrition Score, Alerts, and Employee Danger Categories")

if not os.path.exists(DATA_PATH):
    st.error("V3 risk scored dataset not found. Please run /run-risk-detection from Swagger first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

total = len(df)
low = (df["risk_category"] == "Low Risk").sum()
medium = (df["risk_category"] == "Medium Risk").sum()
high = (df["risk_category"] == "High Risk").sum()
anomalies = (df["anomaly_status"] == "Anomalous Employee").sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Employees", total)
c2.metric("Low Risk", low)
c3.metric("Medium Risk", medium)
c4.metric("High Risk", high)
c5.metric("Anomalies", anomalies)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Risk Classification")
    risk_counts = df["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Count"]
    fig = px.bar(risk_counts, x="Risk Category", y="Count", text="Count")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Burnout Categories")
    burnout_counts = df["burnout_category"].value_counts().reset_index()
    burnout_counts.columns = ["Burnout Category", "Count"]
    fig2 = px.pie(burnout_counts, names="Burnout Category", values="Count")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Risk Alerts")

alerts = df[
    (df["risk_category"] == "High Risk") |
    (df["burnout_category"] == "Severe Burnout") |
    (df["anomaly_status"] == "Anomalous Employee")
]

if alerts.empty:
    st.success("No critical alerts found.")
else:
    st.warning(f"{len(alerts)} employees need HR attention.")
    st.dataframe(
        alerts[
            [
                "EmployeeNumber",
                "Department",
                "JobRole",
                "OverTime",
                "JobSatisfaction",
                "WorkLifeBalance",
                "MonthlyIncome",
                "attrition_score",
                "burnout_score",
                "risk_category",
                "burnout_category",
                "anomaly_status",
                "danger_explanation"
            ]
        ],
        use_container_width=True
    )

st.divider()

st.subheader("All Employees Risk Table")

risk_filter = st.selectbox(
    "Filter by risk category",
    ["All", "Low Risk", "Medium Risk", "High Risk"]
)

filtered = df.copy()

if risk_filter != "All":
    filtered = filtered[filtered["risk_category"] == risk_filter]

st.dataframe(
    filtered[
        [
            "EmployeeNumber",
            "Department",
            "JobRole",
            "Attrition",
            "OverTime",
            "JobSatisfaction",
            "WorkLifeBalance",
            "MonthlyIncome",
            "attrition_score",
            "burnout_score",
            "risk_category",
            "burnout_category",
            "danger_explanation"
        ]
    ],
    use_container_width=True
)

st.divider()

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download V3 Risk Scored Dataset",
    data=csv,
    file_name="V3_risk_scored_dataset.csv",
    mime="text/csv"
)