import os
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

from recommendation_dashboard import show_recommendation_dashboard
from cleaned_recommendation_dashboard import (
    show_cleaned_recommendation_dashboard
)

API_URL = "http://127.0.0.1:8000"
RISK_DATA_PATH = "data/processed/V3_risk_scored_dataset.csv"

st.set_page_config(
    page_title="HR Talent Mining Dashboard",
    layout="wide"
)

# SIDEBAR

st.sidebar.title("HR Talent Mining")

page = st.sidebar.radio(
    "Navigation",
    [
        "Upload Dataset",
        "Employee Segmentation",
        "Risk Dashboard",
        "Recommendation Engine",
        "Recommendation Engine (Cleaned)",
        "Dataset Versions"
    ]
)

# TITLE

st.title("HR Talent Mining & Recruitment Intelligence Platform")
st.caption("A practical Python Data Mining system for HR analytics.")

# SESSION STATE

if "upload_response" not in st.session_state:
    st.session_state.upload_response = None

if "segmentation_response" not in st.session_state:
    st.session_state.segmentation_response = None

if "risk_response" not in st.session_state:
    st.session_state.risk_response = None

if "recommendation_df" not in st.session_state:
    st.session_state.recommendation_df = None

if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None

if "cleaned_recommendation_df" not in st.session_state:
    st.session_state.cleaned_recommendation_df = None

if "cleaned_recommendation_result" not in st.session_state:
    st.session_state.cleaned_recommendation_result = None


# UPLOAD DATASET

if page == "Upload Dataset":

    st.header("Upload Company HR Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

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

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            response = requests.post(
                f"{API_URL}/upload-dataset",
                files=files
            )

            if response.status_code == 200:

                st.session_state.upload_response = response.json()

                st.success(
                    "Dataset uploaded and cleaned successfully."
                )

            else:
                st.error(response.text)

    if st.session_state.upload_response:

        result = st.session_state.upload_response

        st.subheader("Backend Processing Results")

        st.json(result)


# RECOMMENDATION ENGINE

elif page == "Recommendation Engine":

    st.header("AI Recommendation Engine")

    if st.button("Run Recommendations"):

        response = requests.post(
            f"{API_URL}/run-recommendations"
        )

        if response.status_code == 200:

            result = response.json()

            st.session_state.recommendation_df = pd.DataFrame(
                result["recommendations"]
            )

            st.session_state.recommendation_result = result

            st.success(
                "Recommendations generated successfully"
            )

        else:
            st.error(response.text)

    if st.session_state.recommendation_df is not None:

        show_recommendation_dashboard(
            st.session_state.recommendation_df,
            st.session_state.recommendation_result
        )

    else:
        st.info(
            "Click 'Run Recommendations' to generate results."
        )


# CLEANED RECOMMENDATION ENGINE


elif page == "Recommendation Engine (Cleaned)":

    st.header("Cleaned AI Recommendation Engine")

    st.write(
        "Run recommendation analysis using the cleaned HR dataset."
    )

    st.warning(
        "You must upload a dataset first from the Upload Dataset page."
    )

    if st.button("Run Cleaned Recommendations"):

        response = requests.post(
            f"{API_URL}/run-cleaned-recommendations"
        )

        if response.status_code == 200:

            result = response.json()

            st.session_state.cleaned_recommendation_df = pd.DataFrame(
                result["recommendations"]
            )

            st.session_state.cleaned_recommendation_result = result

            st.success(
                "Cleaned recommendations generated successfully."
            )

        else:
            st.error(response.text)

    if st.session_state.cleaned_recommendation_df is not None:

        show_cleaned_recommendation_dashboard(
            st.session_state.cleaned_recommendation_df,
            st.session_state.cleaned_recommendation_result
        )

    else:
        st.info(
            "Upload dataset first, then run cleaned recommendations."
        )


# DATASET VERSIONS

elif page == "Dataset Versions":

    st.header("Dataset Versions")

    response = requests.get(
        f"{API_URL}/dataset-info"
    )

    if response.status_code == 200:

        info = response.json()

        st.json(info)

    else:
        st.error(response.text)