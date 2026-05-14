import streamlit as st
import pandas as pd
import plotly.express as px


def show_cleaned_recommendation_dashboard(
    df: pd.DataFrame,
    api_raw=None
):

    st.metric("Total Employees", len(df))

    # PRIORITY DISTRIBUTION

    st.subheader("Priority Distribution")

    priority_counts = (
        df["priority"]
        .value_counts()
        .reindex(["Critical", "High", "Medium", "Low"])
        .fillna(0)
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Critical", int(priority_counts["Critical"]))
    c2.metric("High", int(priority_counts["High"]))
    c3.metric("Medium", int(priority_counts["Medium"]))
    c4.metric("Low", int(priority_counts["Low"]))

    fig = px.bar(
        x=priority_counts.index,
        y=priority_counts.values,
        text=priority_counts.values,
        color=priority_counts.index,
        title="Employee Retention Priority Levels"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ANALYTICS

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Department Distribution")

        dept = (
            df["Department"]
            .value_counts()
            .reset_index()
        )

        dept.columns = ["Department", "Count"]

        fig2 = px.pie(
            dept,
            names="Department",
            values="Count"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with col2:

        st.subheader("Average Monthly Income")

        income = (
            df.groupby("priority")["MonthlyIncome"]
            .mean()
            .reset_index()
        )

        fig3 = px.bar(
            income,
            x="priority",
            y="MonthlyIncome",
            color="priority"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    st.divider()

    # FILTERS

    st.subheader("Employee Recommendation Explorer")

    selected_priority = st.selectbox(
        "Filter by Priority",
        ["All", "Critical", "High", "Medium", "Low"]
    )

    if selected_priority == "All":

        filtered = df.copy()

    else:

        filtered = df[
            df["priority"] == selected_priority
        ]

    st.write(f"Total Filtered Employees: {len(filtered)}")

    st.dataframe(
        filtered,
        use_container_width=True
    )

    st.divider()

    # EMPLOYEE DETAILS

    if not filtered.empty:

        employee_options = [

            f"{row['EmployeeNumber']} - {row['JobRole']}"

            for _, row in filtered.iterrows()
        ]

        selected_employee = st.selectbox(
            "Select Employee",
            employee_options
        )

        selected_id = int(
            selected_employee.split(" - ")[0]
        )

        row = filtered[
            filtered["EmployeeNumber"] == selected_id
        ].iloc[0]

        st.subheader(
            f"Employee #{selected_id}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Retention Strategy")

            st.info(
                row["retention_strategy"]
            )

        with col2:

            st.markdown("### Priority Score")

            st.metric(
                "Priority Score",
                row["priority_score"]
            )

        st.markdown("### HR Actions")

        actions = row["hr_actions"]

        if isinstance(actions, list):

            for action in actions:
                st.markdown(f"- {action}")

        else:
            st.write(actions)

        st.markdown("### Peer Gap Warnings")

        gaps = row["peer_gap_warnings"]

        if isinstance(gaps, list) and len(gaps) > 0:

            for g in gaps:
                st.warning(g)

        else:
            st.success(
                "No major peer gaps detected."
            )