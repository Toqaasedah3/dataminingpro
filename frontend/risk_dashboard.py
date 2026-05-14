import os
import pandas as pd
import streamlit as st
import plotly.express as px

RISK_PATH = "data/processed/V5_risk_scored_dataset.csv"

st.set_page_config(page_title="Risk Detection Results", layout="wide")

st.title("Risk Detection Results")
st.write("Attrition risk, burnout intelligence, anomaly detection, and risk explanations.")

if not os.path.exists(RISK_PATH):
    st.error("V5 risk scored dataset not found. Run /run-risk-detection first.")
    st.stop()

df = pd.read_csv(RISK_PATH)

total = len(df)
low = int((df["risk_category"] == "Low Risk").sum())
medium = int((df["risk_category"] == "Medium Risk").sum())
high = int((df["risk_category"] == "High Risk").sum())
anomalies = int((df["anomaly_status"] == "Anomalous Employee").sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Employees", total)
c2.metric("Low Risk", low)
c3.metric("Medium Risk", medium)
c4.metric("High Risk", high)
c5.metric("Anomalies", anomalies)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Category Distribution")
    risk_counts = df["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Count"]
    fig = px.bar(
        risk_counts,
        x="Risk Category",
        y="Count",
        text="Count",
        title="Employees by Risk Level"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Burnout Category Distribution")
    burnout_counts = df["burnout_category"].value_counts().reset_index()
    burnout_counts.columns = ["Burnout Category", "Count"]
    fig = px.pie(
        burnout_counts,
        names="Burnout Category",
        values="Count",
        title="Burnout Categories"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Anomaly Detection")
    anomaly_counts = df["anomaly_status"].value_counts().reset_index()
    anomaly_counts.columns = ["Anomaly Status", "Count"]
    fig = px.bar(
        anomaly_counts,
        x="Anomaly Status",
        y="Count",
        text="Count",
        title="Normal vs Anomalous Employees"
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Risk by Cluster")
    if "cluster_label" in df.columns:
        cluster_risk = (
            df.groupby(["cluster_label", "risk_category"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            cluster_risk,
            x="cluster_label",
            y="Count",
            color="risk_category",
            barmode="group",
            title="Risk Distribution by Employee Segment"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cluster labels are not available.")

st.divider()

st.subheader("Risk Alerts")

alerts = df[
    (df["risk_category"] == "High Risk") |
    (df["burnout_category"] == "Severe Burnout") |
    (df["anomaly_status"] == "Anomalous Employee")
]

alert_columns = [
    "EmployeeNumber",
    "Department",
    "JobRole",
    "MonthlyIncome",
    "OverTime",
    "JobSatisfaction",
    "WorkLifeBalance",
    "cluster_label",
    "attrition_score",
    "burnout_score",
    "risk_category",
    "burnout_category",
    "anomaly_status",
    "danger_explanation"
]

alert_columns = [col for col in alert_columns if col in df.columns]

if alerts.empty:
    st.success("No critical risk alerts found.")
else:
    st.warning(f"{len(alerts)} employees need HR attention.")
    st.dataframe(alerts[alert_columns], use_container_width=True)

st.divider()

st.subheader("Employee Risk Explanation Table")

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    risk_filter = st.selectbox(
        "Filter by risk category",
        ["All", "Low Risk", "Medium Risk", "High Risk"]
    )

with filter_col2:
    if "cluster_label" in df.columns:
        cluster_filter = st.selectbox(
            "Filter by cluster",
            ["All"] + sorted(df["cluster_label"].dropna().unique().tolist())
        )
    else:
        cluster_filter = "All"

filtered_df = df.copy()

if risk_filter != "All":
    filtered_df = filtered_df[filtered_df["risk_category"] == risk_filter]

if cluster_filter != "All" and "cluster_label" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["cluster_label"] == cluster_filter]

table_columns = [
    "EmployeeNumber",
    "Department",
    "JobRole",
    "cluster_label",
    "attrition_score",
    "burnout_score",
    "risk_category",
    "burnout_category",
    "anomaly_status",
    "danger_explanation"
]

table_columns = [col for col in table_columns if col in filtered_df.columns]

st.dataframe(filtered_df[table_columns], use_container_width=True)

st.divider()

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download V5 Risk Scored Dataset",
    data=csv,
    file_name="V5_risk_scored_dataset.csv",
    mime="text/csv"
)