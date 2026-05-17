import os
import pandas as pd
import streamlit as st
import plotly.express as px


CLUSTERED_PATH = "backend/data/clustered/clustered_data.csv"
RISK_PATH = "data/processed/V5_risk_scored_dataset.csv"

RECOMMENDATION_PATHS = [
    "data/processed/V6_final_recommendation_dataset.csv",
    "data/processed/final_recommendation_dataset.csv",
    "data/processed/recommendation_dataset.csv"
]


def _load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)

    return None


def _load_recommendation_df():
    for path in RECOMMENDATION_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)

    return None


def render_full_hr_intelligence_dashboard():
    st.title("Full HR Intelligence Dashboard")
    st.write("Executive summary combining segmentation, risk detection, burnout intelligence, and HR recommendations.")

    clustered_df = _load_csv(CLUSTERED_PATH)
    risk_df = _load_csv(RISK_PATH)
    recommendation_df = _load_recommendation_df()

    if clustered_df is None and risk_df is None and recommendation_df is None:
        st.warning("No prepared outputs found yet. Please run segmentation, risk detection, and recommendations first.")
        return

    st.subheader("Executive Overview")

    total_employees = 0

    if risk_df is not None:
        total_employees = len(risk_df)
    elif clustered_df is not None:
        total_employees = len(clustered_df)
    elif recommendation_df is not None:
        total_employees = len(recommendation_df)

    total_clusters = (
        clustered_df["cluster"].nunique()
        if clustered_df is not None and "cluster" in clustered_df.columns
        else 0
    )

    anomalies = (
        int((risk_df["anomaly_status"] == "Anomalous Employee").sum())
        if risk_df is not None and "anomaly_status" in risk_df.columns
        else 0
    )

    high_risk = (
        int((risk_df["risk_category"] == "High Risk").sum())
        if risk_df is not None and "risk_category" in risk_df.columns
        else 0
    )

    medium_risk = (
        int((risk_df["risk_category"] == "Medium Risk").sum())
        if risk_df is not None and "risk_category" in risk_df.columns
        else 0
    )

    critical_recommendations = (
        int((recommendation_df["priority"] == "Critical").sum())
        if recommendation_df is not None and "priority" in recommendation_df.columns
        else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Employees", total_employees)
    c2.metric("Clusters", total_clusters)
    c3.metric("Medium Risk", medium_risk)
    c4.metric("High Risk", high_risk)
    c5.metric("Critical Recommendations", critical_recommendations)

    st.divider()

    st.subheader("Integrated Visual Insights")

    col1, col2 = st.columns(2)

    with col1:
        if clustered_df is not None and "cluster_label" in clustered_df.columns:
            cluster_counts = (
                clustered_df["cluster_label"]
                .value_counts()
                .reset_index()
            )

            cluster_counts.columns = ["Cluster Label", "Count"]

            fig_cluster = px.bar(
                cluster_counts,
                x="Cluster Label",
                y="Count",
                text="Count",
                title="Employee Segments"
            )

            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.info("Cluster data is not available.")

    with col2:
        if risk_df is not None and "risk_category" in risk_df.columns:
            risk_counts = (
                risk_df["risk_category"]
                .value_counts()
                .reset_index()
            )

            risk_counts.columns = ["Risk Category", "Count"]

            fig_risk = px.pie(
                risk_counts,
                names="Risk Category",
                values="Count",
                title="Risk Distribution"
            )

            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("Risk data is not available.")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        if risk_df is not None and "burnout_category" in risk_df.columns:
            burnout_counts = (
                risk_df["burnout_category"]
                .value_counts()
                .reset_index()
            )

            burnout_counts.columns = ["Burnout Category", "Count"]

            fig_burnout = px.bar(
                burnout_counts,
                x="Burnout Category",
                y="Count",
                text="Count",
                title="Burnout Distribution"
            )

            st.plotly_chart(fig_burnout, use_container_width=True)
        else:
            st.info("Burnout data is not available.")

    with col4:
        if recommendation_df is not None and "priority" in recommendation_df.columns:
            priority_counts = (
                recommendation_df["priority"]
                .value_counts()
                .reindex(["Critical", "High", "Medium", "Low"])
                .fillna(0)
                .reset_index()
            )

            priority_counts.columns = ["Priority", "Count"]

            fig_priority = px.bar(
                priority_counts,
                x="Priority",
                y="Count",
                text="Count",
                title="Recommendation Priority Levels"
            )

            st.plotly_chart(fig_priority, use_container_width=True)
        else:
            st.info("Recommendation data is not available.")

    st.divider()

    st.subheader("Risk by Employee Segment")

    if risk_df is not None and {"cluster_label", "risk_category"}.issubset(risk_df.columns):
        cluster_risk = (
            risk_df.groupby(["cluster_label", "risk_category"])
            .size()
            .reset_index(name="Count")
        )

        fig_cluster_risk = px.bar(
            cluster_risk,
            x="cluster_label",
            y="Count",
            color="risk_category",
            barmode="group",
            title="Risk Distribution Across Employee Segments"
        )

        st.plotly_chart(fig_cluster_risk, use_container_width=True)
    else:
        st.info("Cluster-risk integration is not available yet.")

    st.divider()

    st.subheader("Department-Level Intelligence")

    if risk_df is not None and {"Department", "risk_category"}.issubset(risk_df.columns):
        dept_risk = (
            risk_df.groupby(["Department", "risk_category"])
            .size()
            .reset_index(name="Count")
        )

        fig_dept_risk = px.bar(
            dept_risk,
            x="Department",
            y="Count",
            color="risk_category",
            barmode="group",
            title="Risk Distribution by Department"
        )

        st.plotly_chart(fig_dept_risk, use_container_width=True)
    else:
        st.info("Department risk data is not available.")

    st.divider()

    st.subheader("Executive Summary")

    summary_lines = []

    summary_lines.append(
        f"The system analyzed {total_employees} employees across the HR dataset."
    )

    if total_clusters > 0:
        summary_lines.append(
            f"Employee segmentation identified {total_clusters} main workforce groups using K-Means and PCA."
        )

    if risk_df is not None:
        summary_lines.append(
            f"The risk engine detected {medium_risk} medium-risk employees, {high_risk} high-risk employees, and {anomalies} anomalous employee profiles."
        )

    if recommendation_df is not None and "priority" in recommendation_df.columns:
        high_priority = int((recommendation_df["priority"] == "High").sum())

        summary_lines.append(
            f"The recommendation engine generated HR action priorities, including {critical_recommendations} critical and {high_priority} high-priority cases."
        )

    for line in summary_lines:
        st.write(f"- {line}")

    st.divider()

    st.subheader("Final Data Availability")

    availability = {
        "Clustered Dataset": CLUSTERED_PATH,
        "Risk Scored Dataset": RISK_PATH,
        "Recommendation Dataset": " / ".join(RECOMMENDATION_PATHS)
    }

    for name, path in availability.items():
        if name == "Recommendation Dataset":
            exists = recommendation_df is not None
        else:
            exists = os.path.exists(path)

        if exists:
            st.success(f"{name} is available.")
        else:
            st.warning(f"{name} is not available yet.")