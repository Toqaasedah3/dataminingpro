import os
import sys
import pandas as pd
import requests
import streamlit as st

# Make frontend folder imports stable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from segmentation_dashboard import render_segmentation_dashboard
from risk_dashboard import render_risk_dashboard
from recommendation_dashboard import render_recommendation_dashboard
from full_hr_intelligence_dashboard import render_full_hr_intelligence_dashboard


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="HR Talent Mining Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #111827;
    }

    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    [data-testid="stMetricValue"] {
        color: #111827;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #111827;
    }

    .section-card {
        padding: 1.2rem;
        border-radius: 1rem;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# SESSION STATE
# =========================

if "upload_response" not in st.session_state:
    st.session_state.upload_response = None

if "segmentation_response" not in st.session_state:
    st.session_state.segmentation_response = None

if "risk_response" not in st.session_state:
    st.session_state.risk_response = None

if "recommendation_response" not in st.session_state:
    st.session_state.recommendation_response = None


# =========================
# SIDEBAR
# =========================

st.sidebar.title("HR Talent Mining")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Upload / Load Dataset",
        "Segmentation Dashboard",
        "Risk Dashboard",
        "Recommendations Dashboard",
        "Full HR Intelligence Dashboard",
        "Dataset Versions"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Pipeline: Data Loading → Segmentation → Risk Detection → Recommendations → Final HR Intelligence"
)


# =========================
# HOME
# =========================

if page == "Home":
    st.title("HR Talent Mining & Recruitment Intelligence Platform")
    st.caption("A practical Data Mining system for HR analytics, workforce segmentation, risk intelligence, and HR recommendations.")

    st.markdown("### Project Pipeline")

    st.info(
        """
        This platform transforms HR datasets into useful intelligence through an integrated pipeline:

        1. Data loading and preprocessing  
        2. Employee segmentation using K-Means and PCA  
        3. Risk detection and burnout intelligence  
        4. HR recommendations and retention strategies  
        5. Final integrated dashboard for decision support
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Toqa")
        st.write("Data engineering, preprocessing, K-Means clustering, PCA, and segmentation dashboard.")

    with col2:
        st.markdown("### Tala")
        st.write("Risk detection, burnout intelligence, anomaly detection, explanations, and risk dashboard.")

    with col3:
        st.markdown("### Alaa")
        st.write("Recommendation engine, retention strategies, final dashboard integration, and final product.")

    st.divider()

    st.markdown("### Recommended Running Order")

    st.code(
        """
1. POST /load-from-huggingface
2. POST /run-segmentation
3. POST /run-risk-detection
4. POST /run-recommendations
5. Open Streamlit dashboards
        """,
        language="text"
    )


# =========================
# UPLOAD / LOAD DATASET
# =========================

elif page == "Upload / Load Dataset":
    st.title("Upload / Load HR Dataset")
    st.write("Load the HR dataset either from Hugging Face or by uploading a CSV file.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Load from Hugging Face")

        if st.button("Load Dataset from Hugging Face", use_container_width=True):
            response = requests.post(f"{API_URL}/load-from-huggingface")

            if response.status_code == 200:
                st.session_state.upload_response = response.json()
                st.success("Dataset loaded and processed successfully.")
                st.json(response.json())
            else:
                st.error(response.text)

    with col_b:
        st.subheader("Dataset Status")

        if st.button("Check Dataset Info", use_container_width=True):
            response = requests.get(f"{API_URL}/dataset-info")

            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error(response.text)

    st.divider()

    st.subheader("Upload CSV Manually")

    uploaded_file = st.file_uploader(
        "Upload HR CSV file",
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
                st.success("Dataset uploaded and cleaned successfully.")
                st.json(response.json())
            else:
                st.error(response.text)


# =========================
# SEGMENTATION DASHBOARD
# =========================

elif page == "Segmentation Dashboard":
    st.title("Employee Segmentation")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Run Segmentation", use_container_width=True):
            response = requests.post(f"{API_URL}/run-segmentation")

            if response.status_code == 200:
                st.session_state.segmentation_response = response.json()
                st.success("Segmentation completed successfully.")
            else:
                st.error(response.text)

    with col2:
        st.info("This page shows employee clusters, PCA visualization, cluster profiles, and model evaluation.")

    st.divider()

    render_segmentation_dashboard(st.session_state.segmentation_response)


# =========================
# RISK DASHBOARD
# =========================

elif page == "Risk Dashboard":
    st.title("Risk Detection & Burnout Intelligence")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Run Risk Detection", use_container_width=True):
            response = requests.post(f"{API_URL}/run-risk-detection")

            if response.status_code == 200:
                st.session_state.risk_response = response.json()
                st.success("Risk detection completed successfully.")
                st.json(response.json())
            else:
                st.error(response.text)

    with col2:
        st.info("This page shows attrition risk, burnout scores, anomaly detection, alerts, and employee explanations.")

    st.divider()

    render_risk_dashboard()


# =========================
# RECOMMENDATIONS DASHBOARD
# =========================

elif page == "Recommendations Dashboard":
    st.title("Strategic HR Recommendations")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Run Recommendations", use_container_width=True):
            response = requests.post(f"{API_URL}/run-recommendations")

            if response.status_code == 200:
                st.session_state.recommendation_response = response.json()
                st.success("Recommendations generated successfully.")
            else:
                st.error(response.text)

    with col2:
        st.info("This page shows HR actions, retention strategies, priority levels, and recommendation insights.")

    st.divider()

    render_recommendation_dashboard(st.session_state.recommendation_response)


# =========================
# FULL HR INTELLIGENCE DASHBOARD
# =========================

elif page == "Full HR Intelligence Dashboard":
    render_full_hr_intelligence_dashboard()


# =========================
# DATASET VERSIONS
# =========================

elif page == "Dataset Versions":
    st.title("Dataset Versions & API Status")

    response = requests.get(f"{API_URL}/dataset-info")

    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error(response.text)

    st.divider()

    st.subheader("Expected Local Outputs")

    expected_files = {
        "Clustered Dataset": "backend/data/clustered/clustered_data.csv",
        "Risk Scored Dataset": "data/processed/V5_risk_scored_dataset.csv",
        "Final Recommendation Dataset": "data/processed/V6_final_recommendation_dataset.csv"
    }

    for label, path in expected_files.items():
        if os.path.exists(path):
            st.success(f"{label}: {path}")
        else:
            st.warning(f"{label} not found: {path}")