import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# =========================
# WorkPulse Configuration
# =========================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

APP_NAME = "WorkPulse"

FRONTEND_STORE = Path("frontend/.saved_data")
SAVED_RAW_PATH = FRONTEND_STORE / "current_uploaded_dataset.csv"
SAVED_META_PATH = FRONTEND_STORE / "current_dataset_meta.txt"

PROCESSED_DIR = Path("backend/data/processed")

OUTPUT_PATHS = {
    "raw": [
        "frontend/.saved_data/current_uploaded_dataset.csv",
        "backend/data/processed/hr_dataset_v0_raw.csv",
    ],
    "cleaned": [
        "backend/data/processed/cleaned_data.csv",
    ],
    "encoded": [
        "backend/data/processed/encoded_data.csv",
    ],
    "scaled": [
        "backend/data/processed/scaled_data.csv",
    ],
    "clustered": [
        "backend/data/processed/clustered_data.csv",
    ],
    "risk": [
        "backend/data/processed/V5_risk_scored_dataset.csv",
    ],
    "recommendations": [
        "backend/data/processed/V6_recommendations.csv",
    ],
}


# =========================
# Page Setup
# =========================

st.set_page_config(
    page_title="WorkPulse",
    page_icon="💼",
    layout="wide"
)

st.markdown(
    """
<style>
.stApp {
    background: #f8fafc;
    color: #0f172a;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0f172a;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: linear-gradient(135deg, #0f172a, #2563eb);
    color: white;
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 22px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
}

.hero h1 {
    margin: 0;
    font-size: 38px;
    font-weight: 800;
}

.hero p {
    margin-top: 8px;
    color: #dbeafe;
    font-size: 16px;
}

.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

.explain-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e3a8a;
    border-radius: 16px;
    padding: 14px 16px;
    margin-top: 8px;
    margin-bottom: 18px;
    font-size: 14px;
}

.action-card {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.warning-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.danger-card {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
}

.small-title {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Utility Functions
# =========================

def show_hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_store():
    FRONTEND_STORE.mkdir(parents=True, exist_ok=True)


def safe_request(method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
    try:
        return requests.request(method, f"{API_URL}{endpoint}", timeout=300, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error("Backend is not running. Start FastAPI first on http://127.0.0.1:8000")
    except Exception as exc:
        st.error(f"Backend error: {exc}")
    return None


def parse_response(response: Optional[requests.Response], ok_message: str):
    if response is None:
        return None

    try:
        data = response.json()
    except Exception:
        data = {"raw_response": response.text}

    if response.status_code in [200, 201]:
        st.success(ok_message)
        return data

    st.error(data.get("detail", response.text))
    return None


def first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Cannot read file: {path}\n{exc}")
        return None


def get_output_df(name: str) -> Optional[pd.DataFrame]:
    return read_csv(first_existing(OUTPUT_PATHS.get(name, [])))


def get_backend_info() -> Dict[str, Any]:
    response = safe_request("GET", "/dataset-info")
    if response is not None and response.status_code == 200:
        try:
            return response.json()
        except Exception:
            pass
    return {"status": "empty"}


def has_uploaded_dataset() -> bool:
    info = get_backend_info()
    return SAVED_RAW_PATH.exists() or info.get("status") == "available"


def save_uploaded_file(uploaded_file) -> pd.DataFrame:
    ensure_store()

    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    if df.empty:
        raise ValueError("CSV file is empty.")

    clear_outputs_only()

    df.to_csv(SAVED_RAW_PATH, index=False)
    SAVED_META_PATH.write_text(uploaded_file.name, encoding="utf-8")

    return df


def clear_outputs_only():
    for key, paths in OUTPUT_PATHS.items():
        for path in paths:
            try:
                if path != str(SAVED_RAW_PATH) and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    for key in list(st.session_state.keys()):
        if key.startswith(("pipeline_", "upload_", "filter_", "dashboard_")):
            del st.session_state[key]


def clear_all_data():
    if FRONTEND_STORE.exists():
        shutil.rmtree(FRONTEND_STORE, ignore_errors=True)

    if PROCESSED_DIR.exists():
        for file in PROCESSED_DIR.glob("*.csv"):
            try:
                file.unlink()
            except Exception:
                pass

    st.session_state.clear()


def metric_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, help_text = item
        col.metric(label, value, help=help_text)


def basic_metrics(df: pd.DataFrame):
    metric_row([
        ("Rows", f"{len(df):,}", "Total records"),
        ("Columns", f"{df.shape[1]:,}", "Total columns"),
        ("Missing Cells", f"{int(df.isna().sum().sum()):,}", "Missing values"),
        ("Duplicates", f"{int(df.duplicated().sum()):,}", "Duplicate rows"),
    ])


def download_df(df: pd.DataFrame, file_name: str, label: str = "Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def count_values(df: pd.DataFrame, column: str) -> Optional[pd.DataFrame]:
    if df is None or column not in df.columns:
        return None

    temp = df[column].fillna("Unknown").astype(str).value_counts().reset_index()
    temp.columns = [column, "Count"]
    return temp


def plot_count(df: pd.DataFrame, column: str, title: str, chart_type: str = "bar"):
    counts = count_values(df, column)

    if counts is None or counts.empty:
        st.info(f"{column} is not available.")
        return

    if chart_type == "pie":
        fig = px.pie(
            counts,
            names=column,
            values="Count",
            title=title,
            hole=0.35,
        )
    else:
        fig = px.bar(
            counts,
            x=column,
            y="Count",
            text="Count",
            title=title,
        )
        fig.update_traces(textposition="outside")

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def explanation(text: str):
    st.markdown(
        f"""
        <div class="explain-card">
            💡 <b>Insight:</b> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_box(text: str):
    st.markdown(
        f"""
        <div class="action-card">
            ✅ <b>Recommended Action:</b> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    if df is None:
        return None

    lower_map = {c.lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def filter_dataframe(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    filtered = df.copy()

    candidate_cols = [
        "Department",
        "JobRole",
        "EducationField",
        "Gender",
        "MaritalStatus",
        "OverTime",
        "risk_category",
        "burnout_category",
        "anomaly_status",
        "priority",
        "cluster",
        "cluster_label",
    ]

    available_cols = [c for c in candidate_cols if c in filtered.columns]

    st.markdown("### 🔎 Dashboard Filters")

    if not available_cols:
        st.info("No filter columns are available in this dataset.")
        return filtered

    with st.container():
        cols = st.columns(3)

        for i, col_name in enumerate(available_cols):
            values = (
                filtered[col_name]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

            options = ["All"] + values

            selected = cols[i % 3].selectbox(
                label=col_name,
                options=options,
                key=f"{prefix}_{col_name}",
            )

            if selected != "All":
                filtered = filtered[filtered[col_name].astype(str) == selected]

    st.caption(f"Showing {len(filtered):,} employees after filters.")
    return filtered


def merge_outputs() -> Optional[pd.DataFrame]:
    rec_df = get_output_df("recommendations")
    risk_df = get_output_df("risk")
    clustered_df = get_output_df("clustered")
    cleaned_df = get_output_df("cleaned")
    raw_df = get_output_df("raw")

    if rec_df is not None:
        dashboard_df = rec_df.copy()
    elif risk_df is not None:
        dashboard_df = risk_df.copy()
    elif clustered_df is not None:
        dashboard_df = clustered_df.copy()
    elif cleaned_df is not None:
        dashboard_df = cleaned_df.copy()
    elif raw_df is not None:
        dashboard_df = raw_df.copy()
    else:
        return None

    return dashboard_df


def generate_simple_summary(df: pd.DataFrame) -> List[str]:
    summary = []

    risk_col = detect_column(df, ["risk_category", "Risk Category"])
    burnout_col = detect_column(df, ["burnout_category", "Burnout Category"])
    priority_col = detect_column(df, ["priority", "Priority"])
    dept_col = detect_column(df, ["Department"])
    anomaly_col = detect_column(df, ["anomaly_status", "Anomaly Status"])

    total = len(df)

    if risk_col:
        high_count = int(df[risk_col].astype(str).str.lower().eq("high risk").sum())
        summary.append(f"{high_count} out of {total} employees are classified as high risk.")

    if burnout_col:
        severe_count = int(df[burnout_col].astype(str).str.lower().eq("severe burnout").sum())
        summary.append(f"{severe_count} employees show severe burnout indicators.")

    if priority_col:
        critical_count = int(df[priority_col].astype(str).str.lower().eq("critical").sum())
        summary.append(f"{critical_count} employees require critical or urgent HR attention.")

    if anomaly_col:
        anomaly_count = int(df[anomaly_col].astype(str).str.lower().str.contains("anomaly").sum())
        summary.append(f"{anomaly_count} employees were detected as unusual or anomalous cases.")

    if dept_col and risk_col:
        temp = df[df[risk_col].astype(str).str.lower().eq("high risk")]
        if not temp.empty:
            top_dept = temp[dept_col].astype(str).value_counts().idxmax()
            summary.append(f"The department with the highest number of high-risk employees is {top_dept}.")

    if not summary:
        summary.append("The dashboard is ready, but risk and recommendation columns were not found.")

    return summary


def recommended_action_text(row: pd.Series) -> str:
    priority = str(row.get("priority", "")).lower()
    risk = str(row.get("risk_category", "")).lower()
    burnout = str(row.get("burnout_category", "")).lower()

    existing_action = row.get("recommended_action", None)
    if pd.notna(existing_action) and str(existing_action).strip():
        return str(existing_action)

    strategy = row.get("retention_strategy", None)
    if pd.notna(strategy) and str(strategy).strip():
        return str(strategy)

    if "critical" in priority or "high" in risk:
        return "Schedule an urgent HR check-in, review workload, and prepare a personalized retention plan."

    if "severe" in burnout:
        return "Reduce workload pressure, offer wellness support, and follow up with the manager."

    if "medium" in risk:
        return "Monitor engagement, review satisfaction factors, and provide career development support."

    return "Keep monitoring employee engagement and maintain regular feedback conversations."


# =========================
# Sidebar
# =========================

st.sidebar.markdown("## 💼 WorkPulse")
st.sidebar.caption("HR Talent Mining & Workforce Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📤 Upload Dataset",
        "🚀 Run Full Pipeline",
        "📊 Full Dashboard",
        "🧾 Data Explorer",
        "📄 Reports",
        "🗑️ Delete Data",
    ],
)
st.sidebar.divider()

if SAVED_RAW_PATH.exists():
    saved_name = (
        SAVED_META_PATH.read_text(encoding="utf-8")
        if SAVED_META_PATH.exists()
        else "Saved dataset"
    )
    st.sidebar.success(f"Saved dataset: {saved_name}")
else:
    st.sidebar.warning("No saved dataset")

# =========================
# HOME PAGE / LANDING PAGE
# =========================

if page == "🏠 Home":

    show_hero(
        "💼 WorkPulse HR Talent Mining Platform",
        "AI-powered workforce intelligence platform for employee analytics, attrition prediction, burnout detection, and strategic HR decision-making."
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🚀 About WorkPulse")

    st.write("""
    WorkPulse is an advanced HR analytics platform designed to help organizations transform raw workforce data into actionable strategic insights.

    Using data mining, machine learning, and predictive analytics, WorkPulse enables HR teams to:

    - Upload and analyze company HR datasets
    - Automatically clean and preprocess workforce data
    - Segment employees into meaningful workforce clusters
    - Detect high-risk attrition employees
    - Identify burnout indicators
    - Generate personalized retention strategies
    - Support data-driven HR decision making
    """)

    explanation(
        "WorkPulse combines data engineering, employee segmentation, burnout analysis, anomaly detection, and recommendation systems into one unified platform."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # CORE FEATURES
    # ======================

    st.subheader("🔍 Core Platform Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
        <h4>📤 Smart Dataset Upload</h4>
        <p>Upload HR datasets securely with automatic validation, cleaning, and preprocessing.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h4>🧠 Employee Segmentation</h4>
        <p>Use K-Means clustering, PCA, and workforce analytics to discover employee segments.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
        <h4>⚠ Risk & Burnout Detection</h4>
        <p>Predict attrition risk, severe burnout, and workforce anomalies before they escalate.</p>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="card">
        <h4>📈 HR Dashboard</h4>
        <p>Interactive visualizations for workforce KPIs, department risk, and strategic metrics.</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="card">
        <h4>🎯 Retention Recommendations</h4>
        <p>Generate personalized HR actions and employee retention strategies.</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="card">
        <h4>📄 Reports & Exports</h4>
        <p>Download processed data, recommendations, and reports for executive use.</p>
        </div>
        """, unsafe_allow_html=True)

    # ======================
    # PIPELINE FLOW
    # ======================

    st.subheader("⚙️ WorkPulse Full Pipeline")

    st.markdown("""
    <div class="card">
    <h4>System Workflow:</h4>

    **Upload Dataset**  
    ↓  
    **Data Cleaning & Preprocessing**  
    ↓  
    **Employee Segmentation**  
    ↓  
    **Risk Detection & Burnout Analysis**  
    ↓  
    **Recommendation Engine**  
    ↓  
    **Executive Dashboard & Reports**
    </div>
    """, unsafe_allow_html=True)

    action_box(
        "Organizations can run the complete pipeline automatically or explore individual modules independently."
    )

    # ======================
    # BUSINESS VALUE
    # ======================

    st.subheader("🏢 Business Impact")

    metric_row([
        ("Attrition Prevention", "High", "Reduce employee turnover"),
        ("Burnout Detection", "Advanced", "Protect employee wellbeing"),
        ("HR Efficiency", "Optimized", "Automate workforce analysis"),
        ("Strategic Decision Support", "Data-Driven", "Enable executive insights"),
    ])

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("""
    ### Why WorkPulse?

    Traditional HR systems often rely on reactive processes.

    WorkPulse shifts HR toward **predictive intelligence**, allowing organizations to:

    - Improve employee retention
    - Reduce workforce instability
    - Detect hidden operational risks
    - Optimize HR investments
    - Increase organizational resilience
    """)

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # TEAM / PROJECT NOTE
    # ======================

    st.subheader("🎓 Academic & Professional Value")

    st.info("""
    This platform represents a complete end-to-end HR data mining system integrating:

    - Data Engineering
    - Machine Learning
    - Predictive Analytics
    - Workforce Intelligence
    - Strategic HR Automation
    """)

    st.success("WorkPulse is built to function as both an academic data mining project and a real-world HR intelligence platform.")
# =========================
# Page 1: Upload
# =========================

if page == "📤 Upload Dataset":
    show_hero(
        "📤 WorkPulse Dataset Upload",
        "Upload the company HR dataset once. WorkPulse saves it locally and uses it for the full pipeline.",
    )

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload HR CSV")

        uploaded_file = st.file_uploader("Choose company CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = save_uploaded_file(uploaded_file)
                st.success("Dataset saved successfully. You do not need to upload it again.")
                basic_metrics(df)
                st.dataframe(df.head(30), use_container_width=True)

                if st.button("Send Dataset to Backend + Clean", type="primary", use_container_width=True):
                    uploaded_file.seek(0)
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv",
                        )
                    }

                    data = parse_response(
                        safe_request("POST", "/upload-dataset", files=files),
                        "Dataset uploaded and cleaned successfully.",
                    )

                    if data:
                        st.session_state.upload_result = data
                        st.json(data)

            except Exception as exc:
                st.error(f"Upload failed: {exc}")

        else:
            saved_df = read_csv(str(SAVED_RAW_PATH))

            if saved_df is not None:
                st.success("Saved company dataset found.")
                basic_metrics(saved_df)
                st.dataframe(saved_df.head(30), use_container_width=True)

                if st.button("Send Saved Dataset to Backend + Clean", type="primary", use_container_width=True):
                    with open(SAVED_RAW_PATH, "rb") as file:
                        files = {
                            "file": (
                                SAVED_META_PATH.read_text(encoding="utf-8")
                                if SAVED_META_PATH.exists()
                                else "saved_dataset.csv",
                                file.read(),
                                "text/csv",
                            )
                        }

                    data = parse_response(
                        safe_request("POST", "/upload-dataset", files=files),
                        "Saved dataset sent to backend and cleaned.",
                    )

                    if data:
                        st.session_state.upload_result = data
                        st.json(data)
            else:
                st.info("Upload a company CSV file to start.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("WorkPulse Flow")

        st.markdown(
            """
            1. Upload HR dataset  
            2. Clean and prepare data  
            3. Run segmentation  
            4. Detect risk and burnout  
            5. Generate recommendations  
            6. View dashboard and reports  
            """
        )

        st.subheader("Backend Status")
        st.json(get_backend_info())

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Page 2: Full Pipeline
# =========================

elif page == "🚀 Run Full Pipeline":
    show_hero(
        "🚀 Run WorkPulse Full Pipeline",
        "Run cleaning, segmentation, risk detection, burnout analysis, and recommendation generation.",
    )

    if not has_uploaded_dataset():
        st.warning("Upload a company dataset first.")
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Full Pipeline Execution")

    st.write(
        """
        This process generates all WorkPulse outputs:

        **Cleaned Data → Encoded Data → Scaled Data → Clusters → Risk Scores → Recommendations**
        """
    )

    if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running WorkPulse full pipeline..."):
            data = parse_response(
                safe_request("POST", "/run-full-pipeline"),
                "Full pipeline completed successfully.",
            )

            if data:
                st.session_state.pipeline_result = data
                st.json(data)

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Generated Output Files")

    rows = []
    for name, paths in OUTPUT_PATHS.items():
        path = first_existing(paths)
        rows.append(
            {
                "Output": name,
                "Available": "✅ Yes" if path else "❌ No",
                "Path": path or "—",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =========================
# Page 3: Full Dashboard
# =========================
elif page == "📊 Full Dashboard":

    show_hero(
        "📊 WorkPulse HR Intelligence Dashboard",
        "Dashboard based only on V6_recommendations.csv"
    )

    from pathlib import Path
    import pandas as pd
    import plotly.express as px

    BASE_DIR = Path(__file__).resolve().parents[1]
    path = BASE_DIR / "backend" / "data" / "processed" / "V6_recommendations.csv"

    if not path.exists():
        st.error(f"File not found: {path}")
        st.stop()

    dashboard_df = pd.read_csv(path)
    st.success("Recommendations loaded ✅")

    filtered_df = filter_dataframe(dashboard_df, "dashboard")

    total = len(filtered_df)

    high_risk = int((filtered_df["risk_category"] == "High Risk").sum()) if "risk_category" in filtered_df.columns else 0
    medium_risk = int((filtered_df["risk_category"] == "Medium Risk").sum()) if "risk_category" in filtered_df.columns else 0
    severe_burnout = int((filtered_df["burnout_category"] == "Severe Burnout").sum()) if "burnout_category" in filtered_df.columns else 0
    anomalies = int((filtered_df["anomaly_status"] == "Anomalous Employee").sum()) if "anomaly_status" in filtered_df.columns else 0
    critical = int((filtered_df["priority"] == "Critical").sum()) if "priority" in filtered_df.columns else 0

    metric_row([
        ("Employees", f"{total:,}", "Total employees"),
        ("High Risk", high_risk, "High risk employees"),
        ("Medium Risk", medium_risk, "Medium risk employees"),
        ("Severe Burnout", severe_burnout, "Severe burnout employees"),
        ("Anomalies", anomalies, "Anomalous employees"),
        ("Critical Actions", critical, "Critical priority"),
    ])

    st.divider()

    st.subheader("🧠 Executive Summary")
    st.markdown(f"""
    WorkPulse analyzed **{total} employees** from the final recommendations file.

    - **{high_risk}** employees are High Risk  
    - **{medium_risk}** employees are Medium Risk  
    - **{severe_burnout}** employees have Severe Burnout  
    - **{anomalies}** employees are anomalous  
    - **{critical}** employees need Critical HR actions  
    """)

    st.divider()

    st.subheader("1. Risk & Burnout")

    c1, c2 = st.columns(2)

    with c1:
        if "risk_category" in filtered_df.columns:
            plot_count(filtered_df, "risk_category", "Risk Distribution", "pie")
            st.caption("Shows employees by attrition risk level.")

    with c2:
        if "burnout_category" in filtered_df.columns:
            plot_count(filtered_df, "burnout_category", "Burnout Distribution", "pie")
            st.caption("Shows employees by burnout level.")

    st.subheader("2. Priority & Retention Strategy")

    c3, c4 = st.columns(2)

    with c3:
        if "priority" in filtered_df.columns:
            plot_count(filtered_df, "priority", "Priority Distribution")
            st.caption("Shows urgency of HR actions.")

    with c4:
        if "retention_strategy" in filtered_df.columns:
            top_strategy = (
                filtered_df["retention_strategy"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )
            top_strategy.columns = ["Retention Strategy", "Count"]

            fig = px.bar(
                top_strategy,
                x="Count",
                y="Retention Strategy",
                orientation="h",
                text="Count",
                title="Top Retention Strategies"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows the most common HR retention strategies.")

    st.subheader("3. Department & Job Role Analysis")

    c5, c6 = st.columns(2)

    with c5:
        if {"Department", "risk_category"}.issubset(filtered_df.columns):
            dept_risk = (
                filtered_df
                .groupby(["Department", "risk_category"])
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                dept_risk,
                x="Department",
                y="Count",
                color="risk_category",
                barmode="group",
                title="Risk by Department"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows which departments have more risky employees.")

    with c6:
        if {"JobRole", "risk_category"}.issubset(filtered_df.columns):
            role_risk = (
                filtered_df
                .groupby(["JobRole", "risk_category"])
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                role_risk,
                x="JobRole",
                y="Count",
                color="risk_category",
                barmode="stack",
                title="Risk by Job Role"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows risk concentration by job role.")

    st.subheader("4. Anomaly & Scores Analysis")

    c7, c8 = st.columns(2)

    with c7:
        if "anomaly_status" in filtered_df.columns:
            plot_count(filtered_df, "anomaly_status", "Normal vs Anomalous Employees")
            st.caption("Shows unusual employee cases.")

    with c8:
        if {"attrition_score", "burnout_score", "priority"}.issubset(filtered_df.columns):
            fig = px.scatter(
                filtered_df,
                x="attrition_score",
                y="burnout_score",
                color="priority",
                size="priority_score" if "priority_score" in filtered_df.columns else None,
                hover_data=[
                    c for c in [
                        "EmployeeNumber",
                        "Department",
                        "JobRole",
                        "risk_category",
                        "burnout_category",
                        "retention_strategy"
                    ]
                    if c in filtered_df.columns
                ],
                title="Attrition vs Burnout Score"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows employees with high attrition and burnout scores.")

    st.subheader("5. Peer Gap Warnings")

    if "gap_count" in filtered_df.columns:
        plot_count(filtered_df, "gap_count", "Peer Gap Warning Count")
        st.caption("Shows how many peer comparison gaps were detected.")

    st.subheader("6. Recommended HR Actions")

    action_cols = [
        "EmployeeNumber",
        "Department",
        "JobRole",
        "risk_category",
        "burnout_category",
        "anomaly_status",
        "attrition_score",
        "burnout_score",
        "priority_score",
        "priority",
        "retention_strategy",
        "hr_actions",
        "peer_gap_warnings",
        "action_count",
        "gap_count",
        "peer_avg_income",
        "peer_avg_satisfaction",
        "peer_avg_wlb",
    ]

    available_cols = [c for c in action_cols if c in filtered_df.columns]

    st.dataframe(filtered_df[available_cols], use_container_width=True, height=450)

    download_df(filtered_df[available_cols], "workpulse_recommended_actions.csv")

    with st.expander("🧾 View Full Recommendations Data"):
        st.dataframe(filtered_df, use_container_width=True)
        download_df(filtered_df, "workpulse_full_recommendations.csv")
# =========================
# Page 4: Data Explorer
# =========================

elif page == "🧾 Data Explorer":
    show_hero(
        "🧾 WorkPulse Data Explorer",
        "Inspect raw, cleaned, encoded, scaled, clustered, risk, and recommendation outputs.",
    )

    tabs = st.tabs([
        "Raw",
        "Cleaned",
        "Encoded",
        "Scaled",
        "Clustered",
        "Risk",
        "Recommendations",
    ])

    keys = [
        "raw",
        "cleaned",
        "encoded",
        "scaled",
        "clustered",
        "risk",
        "recommendations",
    ]

    for tab, key in zip(tabs, keys):
        with tab:
            df = get_output_df(key)

            if df is None:
                st.info(f"{key.title()} data is not available yet.")
            else:
                basic_metrics(df)
                filtered = filter_dataframe(df, f"explorer_{key}")
                st.dataframe(filtered, use_container_width=True, height=500)
                download_df(filtered, f"workpulse_{key}_data.csv")


# =========================
# Page 5: Reports
# =========================

elif page == "📄 Reports":
    show_hero(
        "📄 WorkPulse Reports",
        "Download all generated CSV outputs from the HR intelligence pipeline.",
    )

    for name, paths in OUTPUT_PATHS.items():
        path = first_existing(paths)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        col1.write(f"### {name.title()} Dataset")
        col1.caption(path or "Not available")

        df = read_csv(path)

        if df is not None:
            col1.write(f"Rows: {len(df):,} | Columns: {df.shape[1]:,}")
            with col2:
                download_df(df, os.path.basename(path), "Download")
        else:
            col2.warning("Missing")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Page 6: Delete
# =========================

elif page == "🗑️ Delete Data":
    show_hero(
        "🗑️ Delete WorkPulse Data",
        "Delete saved uploaded data and generated pipeline outputs when starting with a new company dataset.",
    )

    st.markdown(
        """
        <div class="danger-card">
            This action deletes the saved frontend dataset and all generated backend CSV files.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("❌ Delete Uploaded Dataset + All Outputs", type="primary", use_container_width=True):
        clear_all_data()
        st.success("All saved data and generated outputs were deleted.")
        st.rerun()