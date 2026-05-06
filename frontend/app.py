import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="HR Talent Mining Dashboard",
    layout="wide"
)

st.sidebar.title("HR Talent Mining")
page = st.sidebar.radio(
    "Navigation",
    ["Upload Dataset", "Employee Segmentation", "Dataset Versions"]
)

st.title("HR Talent Mining & Recruitment Intelligence Platform")
st.caption("A practical Python Data Mining system for HR analytics.")

if "upload_response" not in st.session_state:
    st.session_state.upload_response = None

if "segmentation_response" not in st.session_state:
    st.session_state.segmentation_response = None

if page == "Upload Dataset":
    st.header("Upload Company HR Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded in frontend preview.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Column Names")
        st.write(list(df.columns))

        if st.button("Send to Backend and Clean Dataset"):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{API_URL}/upload-dataset", files=files)

            if response.status_code == 200:
                st.session_state.upload_response = response.json()
                st.success("Dataset uploaded, cleaned, and versioned successfully.")
            else:
                st.error(response.text)

    if st.session_state.upload_response:
        result = st.session_state.upload_response

        st.subheader("Backend Processing Results")
        st.json({
            "detected_columns": result["detected_columns"],
            "missing_important_columns": result["missing_important_columns"],
            "raw_version_path": result["raw_version_path"],
            "cleaned_version_path": result["cleaned_version_path"],
        })

elif page == "Employee Segmentation":
    st.header("Employee Segmentation: K-Means + PCA")

    if st.button("Run Segmentation"):
        response = requests.post(f"{API_URL}/run-segmentation")

        if response.status_code == 200:
            st.session_state.segmentation_response = response.json()
            st.success("Segmentation completed successfully.")
        else:
            st.error(response.text)

    if st.session_state.segmentation_response:
        result = st.session_state.segmentation_response

        col1, col2 = st.columns(2)
        col1.metric("Best K", result["best_k"])
        col2.metric("PCA Variance", f"{sum(result['pca_variance']):.2f}")

        st.subheader("Cluster Summary")
        st.dataframe(pd.DataFrame(result["summary"]), use_container_width=True)

        st.subheader("Clustered Data Preview")
        preview_df = pd.DataFrame(result["preview"])
        st.dataframe(preview_df, use_container_width=True)

        if "pca_1" in preview_df.columns and "pca_2" in preview_df.columns:
            fig = px.scatter(
                preview_df,
                x="pca_1",
                y="pca_2",
                color="cluster_name",
                title="PCA Employee Cluster Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Dataset Versions":
    st.header("Dataset Versions")

    response = requests.get(f"{API_URL}/dataset-info")
    if response.status_code == 200:
        info = response.json()
        st.json(info)
    else:
        st.error(response.text)
