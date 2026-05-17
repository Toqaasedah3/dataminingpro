import os
import pandas as pd
import streamlit as st
import plotly.express as px


RISK_PATH = "data/processed/V5_risk_scored_dataset.csv"


def render_risk_dashboard():
    st.subheader("Employee Risk Detection Results")
    st.write("Attrition risk, burnout intelligence, anomaly detection, cluster context, and employee explanations.")

    if not os.path.exists(RISK_PATH):
        st.warning("V5 risk scored dataset not found. Please run /run-risk-detection first.")
        return

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
            title="Employees by Risk Category"
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

    st.subheader("Anomaly Detection Results")

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

    st.divider()

    st.subheader("Cluster + Risk Insights")

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
            title="Risk Distribution by Employee Cluster"
        )

        fig.update_layout(
            xaxis_title="Cluster Label",
            yaxis_title="Employee Count"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cluster labels are not available in the risk dataset.")

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
        "cluster_label",
        "MonthlyIncome",
        "OverTime",
        "JobSatisfaction",
        "WorkLifeBalance",
        "attrition_score",
        "burnout_score",
        "risk_category",
        "burnout_category",
        "anomaly_status",
        "danger_explanation"
    ]

    alert_columns = [
        col for col in alert_columns
        if col in df.columns
    ]

    if alerts.empty:
        st.success("No critical risk alerts found.")
    else:
        st.warning(f"{len(alerts)} employees need HR attention.")
        st.dataframe(alerts[alert_columns], use_container_width=True)

    st.divider()

    st.subheader("Employee Risk Explanation Table")

    col1, col2 = st.columns(2)

    with col1:
        risk_filter = st.selectbox(
            "Filter by risk category",
            ["All", "Low Risk", "Medium Risk", "High Risk"]
        )

    with col2:
        if "Department" in df.columns:
            department_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())
            department_filter = st.selectbox(
                "Filter by department",
                department_options
            )
        else:
            department_filter = "All"

    filtered_df = df.copy()

    if risk_filter != "All":
        filtered_df = filtered_df[
            filtered_df["risk_category"] == risk_filter
        ]

    if department_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Department"] == department_filter
        ]

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

    table_columns = [
        col for col in table_columns
        if col in filtered_df.columns
    ]

    st.dataframe(filtered_df[table_columns], use_container_width=True)

    st.divider()

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download V5 Risk Scored Dataset",
        data=csv,
        file_name="V5_risk_scored_dataset.csv",
        mime="text/csv"
    )