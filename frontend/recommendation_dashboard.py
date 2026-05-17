import os
import ast
import pandas as pd
import streamlit as st
import plotly.express as px


RECOMMENDATION_PATHS = [
    "data/processed/V6_final_recommendation_dataset.csv",
    "data/processed/final_recommendation_dataset.csv",
    "data/processed/recommendation_dataset.csv"
]


def _load_recommendation_dataframe(api_response=None):
    if api_response and isinstance(api_response, dict):
        if "recommendations" in api_response:
            return pd.DataFrame(api_response["recommendations"])

    for path in RECOMMENDATION_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)

    return None


def _parse_list_value(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return parsed

        except Exception:
            return [value]

    if pd.isna(value):
        return []

    return [str(value)]


def render_recommendation_dashboard(api_response=None):
    st.subheader("Strategic HR Recommendations Dashboard")
    st.write("Retention strategies, HR action plans, employee priorities, and recommendation insights.")

    df = _load_recommendation_dataframe(api_response)

    if df is None or df.empty:
        st.warning("No recommendation dataset found. Please run /run-recommendations first.")
        return

    required_columns = [
        "priority",
        "EmployeeNumber",
        "retention_strategy",
        "hr_actions"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        st.error(f"Recommendation dataset is missing required columns: {missing}")
        st.dataframe(df.head(20), use_container_width=True)
        return

    total = len(df)

    priority_counts = (
        df["priority"]
        .value_counts()
        .reindex(["Critical", "High", "Medium", "Low"])
        .fillna(0)
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Employees", total)
    c2.metric("Critical", int(priority_counts["Critical"]))
    c3.metric("High", int(priority_counts["High"]))
    c4.metric("Medium", int(priority_counts["Medium"]))
    c5.metric("Low", int(priority_counts["Low"]))

    st.divider()

    st.subheader("Priority Distribution")

    fig_priority = px.bar(
        x=priority_counts.index,
        y=priority_counts.values,
        text=priority_counts.values,
        labels={
            "x": "Priority Level",
            "y": "Employee Count"
        },
        color=priority_counts.index,
        title="Employee Retention Priority Levels"
    )

    st.plotly_chart(fig_priority, use_container_width=True)

    st.divider()

    st.subheader("Recommendation Analytics")

    col1, col2 = st.columns(2)

    with col1:
        if "Department" in df.columns:
            dept_counts = (
                df["Department"]
                .value_counts()
                .reset_index()
            )

            dept_counts.columns = ["Department", "Count"]

            fig_dept = px.pie(
                dept_counts,
                names="Department",
                values="Count",
                title="Employees by Department"
            )

            st.plotly_chart(fig_dept, use_container_width=True)
        else:
            st.info("Department column is not available.")

    with col2:
        if {"priority", "MonthlyIncome"}.issubset(df.columns):
            income_df = (
                df.groupby("priority")["MonthlyIncome"]
                .mean()
                .round(2)
                .reset_index()
            )

            fig_income = px.bar(
                income_df,
                x="priority",
                y="MonthlyIncome",
                color="priority",
                text="MonthlyIncome",
                title="Average Monthly Income by Priority"
            )

            st.plotly_chart(fig_income, use_container_width=True)
        else:
            st.info("MonthlyIncome or priority column is not available.")

    st.divider()

    if {"Department", "risk_category"}.issubset(df.columns):
        st.subheader("Departmental Risk Distribution")

        dept_risk = (
            df.groupby(["Department", "risk_category"])
            .size()
            .reset_index(name="Count")
        )

        fig_dept_risk = px.bar(
            dept_risk,
            x="Department",
            y="Count",
            color="risk_category",
            barmode="group",
            title="Risk Counts by Department"
        )

        st.plotly_chart(fig_dept_risk, use_container_width=True)

    st.divider()

    st.subheader("Employee Recommendation Explorer")

    col1, col2 = st.columns(2)

    with col1:
        selected_priority = st.selectbox(
            "Filter by Priority",
            ["All", "Critical", "High", "Medium", "Low"],
            key="recommendation_priority_filter"
        )

    with col2:
        if "Department" in df.columns:
            departments = ["All"] + sorted(df["Department"].dropna().unique().tolist())
            selected_department = st.selectbox(
                "Filter by Department",
                departments,
                key="recommendation_department_filter"
            )
        else:
            selected_department = "All"

    filtered = df.copy()

    if selected_priority != "All":
        filtered = filtered[
            filtered["priority"] == selected_priority
        ]

    if selected_department != "All":
        filtered = filtered[
            filtered["Department"] == selected_department
        ]

    st.write(f"Total Filtered Employees: {len(filtered)}")

    st.dataframe(filtered, use_container_width=True)

    st.divider()

    st.subheader("Employee Strategy Card")

    if filtered.empty:
        st.info("No employees found for the selected filters.")
        return

    employee_options = []

    for _, row in filtered.iterrows():
        employee_id = row.get("EmployeeNumber", "N/A")
        role = row.get("JobRole", "N/A")
        employee_options.append(f"{employee_id} - {role}")

    selected_employee = st.selectbox(
        "Select Employee",
        employee_options,
        key="recommendation_employee_select"
    )

    selected_id = selected_employee.split(" - ")[0]

    selected_row = filtered[
        filtered["EmployeeNumber"].astype(str) == str(selected_id)
    ].iloc[0]

    st.markdown(f"### Employee #{selected_id}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Priority", selected_row.get("priority", "N/A"))

    with col2:
        st.metric("Risk Category", selected_row.get("risk_category", "N/A"))

    with col3:
        st.metric("Burnout Category", selected_row.get("burnout_category", "N/A"))

    st.markdown("### Retention Strategy")
    st.info(selected_row.get("retention_strategy", "No retention strategy available."))

    st.markdown("### Recommended HR Actions")

    actions = _parse_list_value(
        selected_row.get("hr_actions", [])
    )

    if actions:
        for action in actions:
            st.markdown(f"- {action}")
    else:
        st.write("No HR actions available.")

    st.markdown("### Peer Gap Warnings")

    gaps = _parse_list_value(
        selected_row.get("peer_gap_warnings", [])
    )

    if gaps:
        for gap in gaps:
            st.warning(gap)
    else:
        st.success("No major peer gaps detected.")

    st.divider()

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Final Recommendation Dataset",
        data=csv,
        file_name="V6_final_recommendation_dataset.csv",
        mime="text/csv"
    )