import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def show_recommendation_dashboard(df: pd.DataFrame, api_raw=None):

    st.metric("Total Records", len(df))

    # PRIORITY DISTRIBUTION

    st.subheader("Priority Distribution Overview")
    priority_counts = df["priority"].value_counts().reindex(
        ["Critical", "High", "Medium", "Low"]
    ).fillna(0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical", int(priority_counts["Critical"]))
    c2.metric("High",     int(priority_counts["High"]))
    c3.metric("Medium",   int(priority_counts["Medium"]))
    c4.metric("Low",      int(priority_counts["Low"]))

    fig_priority = px.bar(
        x=priority_counts.index,
        y=priority_counts.values,
        text=priority_counts.values,
        labels={"x": "Priority Level", "y": "Count"},
        color=priority_counts.index,
        color_discrete_map={
            "Critical": "#D32F2F",
            "High":     "#F57C00",
            "Medium":   "#1976D2",
            "Low":      "#388E3C",
        },
        title="Priority Level Comparison",
    )
    st.plotly_chart(fig_priority, use_container_width=True)
    st.divider()

    # ANALYTICS

    st.subheader("Analytical Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Burnout Level vs. Average Monthly Income**")
        avg_income_burnout = (
            df.groupby("burnout_category")["MonthlyIncome"].mean().reset_index()
        )
        fig_income = px.bar(
            avg_income_burnout,
            x="burnout_category",
            y="MonthlyIncome",
            color="burnout_category",
            title="Avg Income per Burnout Category",
        )
        st.plotly_chart(fig_income, use_container_width=True)

    with col2:
        st.markdown("**Departmental Risk Distribution**")
        dept_risk = (
            df.groupby(["Department", "risk_category"]).size().reset_index(name="Count")
        )
        fig_dept = px.bar(
            dept_risk,
            x="Department",
            y="Count",
            color="risk_category",
            barmode="group",
            title="Risk Counts by Department",
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    st.divider()

    
    # FILTER + RECOMMENDATIONS
    st.subheader("Data Explorer and Strategy Engine")

    selected_priority = st.selectbox(
        "Select Priority Group:",
        options=["All", "Critical", "High", "Medium", "Low"],
        key="rec_priority_filter",
    )

    if selected_priority == "All":
        filtered = df.copy()
    else:
        filtered = df[df["priority"] == selected_priority].copy()

    st.write(f"Displaying **{len(filtered)}** records for: **{selected_priority}**")
    st.dataframe(filtered, use_container_width=True)

    st.divider()

    # Pick an employee ID to see their recommendation
    
    st.markdown(f"### Strategy and Recommendations: {selected_priority}")

    if filtered.empty:
        st.info("No data available for the selected priority.")
    else:
        # Build dropdown options from filtered employees
        employee_options = [
            f"ID: {row['EmployeeNumber']} | Role: {row.get('JobRole', 'N/A')}"
            for _, row in filtered.iterrows()
        ]

        selected_employee_label = st.selectbox(
            "Select an Employee to view their recommendation:",
            options=employee_options,
            key="rec_employee_select",
        )

        # Get the selected employee's row
        selected_id = int(selected_employee_label.split("ID: ")[1].split(" |")[0])
        selected_row = filtered[filtered["EmployeeNumber"] == selected_id].iloc[0]

        # Display the recommendation card
        st.markdown(f"#### Employee {selected_id} — {selected_row.get('JobRole', 'N/A')}")

        card_col1, card_col2 = st.columns(2)

        with card_col1:
            st.markdown("**Retention Strategy**")
            st.info(selected_row.get("retention_strategy", "N/A"))

        with card_col2:
            st.markdown("**Peer Gap Analysis**")
            gaps = selected_row.get("peer_gap_warnings", [])
            if isinstance(gaps, list) and gaps:
                for g in gaps:
                    st.warning(g)
            else:
                st.write("No significant peer gaps identified.")

        st.markdown("**Recommended HR Actions**")
        actions = selected_row.get("hr_actions", [])
        if isinstance(actions, list) and actions:
            for a in actions:
                st.markdown(f"- {a}")
        else:
            st.write(actions)