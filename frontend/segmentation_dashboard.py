import os
import pandas as pd
import streamlit as st
import plotly.express as px


CLUSTERED_PATH = "backend/data/clustered/clustered_data.csv"


def render_segmentation_dashboard(segmentation_result=None):
    st.subheader("HR Employee Segmentation Dashboard")
    st.write("Visual analysis of employee clusters using PCA and K-Means.")

    if not os.path.exists(CLUSTERED_PATH):
        st.warning("No clustered data found. Please run the segmentation pipeline first.")
        return

    df = pd.read_csv(CLUSTERED_PATH)

    if df.empty:
        st.warning("Clustered dataset is empty.")
        return

    total_employees = len(df)
    total_clusters = df["cluster"].nunique() if "cluster" in df.columns else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Employees", total_employees)
    col2.metric("Number of Clusters", total_clusters)

    if segmentation_result and "best_k" in segmentation_result:
        col3.metric("Best K", segmentation_result["best_k"])
    else:
        col3.metric("Best K", total_clusters)

    st.divider()

    st.subheader("Cluster Distribution")

    if "cluster_label" in df.columns:
        cluster_counts = (
            df.groupby(["cluster", "cluster_label"])
            .size()
            .reset_index(name="employee_count")
        )
        x_col = "cluster_label"
    else:
        cluster_counts = (
            df.groupby("cluster")
            .size()
            .reset_index(name="employee_count")
        )
        x_col = "cluster"

    fig_cluster_count = px.bar(
        cluster_counts,
        x=x_col,
        y="employee_count",
        text="employee_count",
        title="Number of Employees in Each Cluster"
    )

    fig_cluster_count.update_traces(textposition="outside")
    fig_cluster_count.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Employee Count"
    )

    st.plotly_chart(fig_cluster_count, use_container_width=True)

    st.divider()

    st.subheader("PCA Cluster Visualization")

    if {"pca_1", "pca_2", "cluster"}.issubset(df.columns):
        hover_cols = [
            col for col in [
                "Age",
                "MonthlyIncome",
                "JobSatisfaction",
                "WorkLifeBalance",
                "YearsAtCompany"
            ]
            if col in df.columns
        ]

        color_col = "cluster_label" if "cluster_label" in df.columns else "cluster"

        fig_pca = px.scatter(
            df,
            x="pca_1",
            y="pca_2",
            color=color_col,
            hover_data=hover_cols,
            title="Employee Clusters Based on PCA Components"
        )

        fig_pca.update_layout(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2"
        )

        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("PCA columns are not available in the clustered data.")

    st.divider()

    st.subheader("Cluster Profiles")

    profile_columns = [
        "Age",
        "MonthlyIncome",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "YearsAtCompany",
        "TotalWorkingYears",
        "DistanceFromHome",
        "PerformanceRating"
    ]

    available_profile_cols = [
        col for col in profile_columns
        if col in df.columns
    ]

    if available_profile_cols:
        profile_df = (
            df.groupby("cluster")[available_profile_cols]
            .mean()
            .round(2)
            .reset_index()
        )

        st.dataframe(profile_df, use_container_width=True)

        selected_metric = st.selectbox(
            "Choose a metric to compare across clusters",
            available_profile_cols
        )

        fig_metric = px.bar(
            profile_df,
            x="cluster",
            y=selected_metric,
            text=selected_metric,
            title=f"Average {selected_metric} by Cluster"
        )

        fig_metric.update_traces(textposition="outside")
        st.plotly_chart(fig_metric, use_container_width=True)
    else:
        st.info("No numeric profile columns found.")

    st.divider()

    st.subheader("Cluster Comparison Heatmap")

    if available_profile_cols:
        heatmap_df = profile_df.set_index("cluster")[available_profile_cols]

        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=True,
            aspect="auto",
            title="Average Feature Values per Cluster"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    st.subheader("Model Evaluation")

    if segmentation_result and "evaluation_table" in segmentation_result:
        eval_df = pd.DataFrame(segmentation_result["evaluation_table"])

        st.dataframe(eval_df, use_container_width=True)

        fig_silhouette = px.line(
            eval_df,
            x="k",
            y="silhouette",
            markers=True,
            title="Silhouette Score by Number of Clusters"
        )

        st.plotly_chart(fig_silhouette, use_container_width=True)

        fig_elbow = px.line(
            eval_df,
            x="k",
            y="inertia",
            markers=True,
            title="Elbow Method - Inertia by K"
        )

        st.plotly_chart(fig_elbow, use_container_width=True)
    else:
        st.info("Evaluation results are available after running segmentation from this session.")

    st.divider()

    st.subheader("Clustered Data Preview")

    st.dataframe(df.head(50), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clustered Data CSV",
        data=csv,
        file_name="clustered_data.csv",
        mime="text/csv"
    )