
from io import BytesIO

import streamlit as st
import pandas as pd
import plotly.express as px

from question_router import route_question
from ai_engine import understand_question
from data_analyzer import analyze_data
from data_profiler import profile_dataset
from insight_generator import generate_insights
from visualization_engine import (
    create_intelligent_chart,
    create_dataset_visualization,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_grouped_dataframe(values):
    """
    Convert grouped analysis output into a standard DataFrame.

    Supports:
    1. DataFrame output from the current analyzer.
    2. Dictionary output from older analyzer versions.
    3. List-like grouped results.
    """

    if isinstance(values, pd.DataFrame):
        result_df = values.copy()

    elif isinstance(values, dict):
        result_df = pd.DataFrame(
            list(values.items()),
            columns=["Category", "Value"],
        )

    elif isinstance(values, (list, tuple)):
        result_df = pd.DataFrame(values)

    else:
        return pd.DataFrame()

    # Standardize column names when possible.
    if len(result_df.columns) >= 2:
        result_df = result_df.iloc[:, :2].copy()
        result_df.columns = ["Category", "Value"]

    return result_df.reset_index(drop=True)


def is_percentage_question(question):
    """
    Identify questions asking for proportions or percentages.
    """

    keywords = [
        "percentage",
        "percent",
        "proportion",
        "share",
        "composition",
    ]

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in keywords
    )


def add_percentage_column(result_df):
    """
    Calculate each category's percentage of the total.
    """

    result_df = result_df.copy()

    result_df["Value"] = pd.to_numeric(
        result_df["Value"],
        errors="coerce",
    )

    total = result_df["Value"].sum()

    if total != 0:
        result_df["Percentage"] = (
            result_df["Value"] / total * 100
        )
    else:
        result_df["Percentage"] = 0.0

    return result_df


def format_grouped_dataframe(result_df, percentage_mode=False):
    """
    Format grouped results for presentation.
    """

    display_df = result_df.copy()

    if "Value" in display_df.columns:
        display_df["Value"] = display_df["Value"].map(
            lambda value: (
                f"{value:,.2f}"
                if isinstance(value, (float, int))
                and not float(value).is_integer()
                else f"{value:,.0f}"
                if isinstance(value, (float, int))
                else value
            )
        )

    if (
        percentage_mode
        and "Percentage" in display_df.columns
    ):
        display_df["Percentage"] = (
            display_df["Percentage"].map(
                lambda value: f"{value:.2f}%"
            )
        )

    return display_df


def dataframe_to_excel_bytes(dataframe, sheet_name="Results"):
    """Serialize a DataFrame to an in-memory Excel workbook."""
    output = BytesIO()
    safe_sheet_name = str(sheet_name)[:31] or "Results"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=safe_sheet_name)
    return output.getvalue()


def exportable_result_dataframe(result):
    """Return a safe tabular representation of a successful analysis result."""
    if not isinstance(result, dict) or not result.get("success"):
        return None

    value = result.get("value")
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return pd.DataFrame(list(value.items()), columns=["Category", "Value"])
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(value)
    if value is not None and result.get("result_type") in {
        "number", "percentage"
    }:
        return pd.DataFrame({"Result": [value]})
    return None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🤖 AI Chat With Your Data")

st.write(
    "Upload a dataset and ask questions "
    "in plain English."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file",
    type=["csv", "xlsx"],
)


# ============================================================
# DATASET PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        file_bytes = uploaded_file.getvalue()

        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            workbook = pd.ExcelFile(BytesIO(file_bytes))
            sheet_names = workbook.sheet_names

            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Choose an Excel worksheet",
                    options=sheet_names,
                    key=f"sheet_selector_{uploaded_file.name}",
                    help="Select which worksheet to analyze.",
                )
            else:
                selected_sheet = sheet_names[0]

            df = pd.read_excel(
                BytesIO(file_bytes),
                sheet_name=selected_sheet,
            )

    except Exception as e:

        st.error(f"Unable to read the file: {e}")
        st.stop()

    if df.empty:

        st.warning(
            "The uploaded dataset is empty. "
            "Please upload a file containing data."
        )
        st.stop()

    # ========================================================
    # DATASET PROFILING
    # ========================================================

    try:

        profile = profile_dataset(df)

        insights = generate_insights(
            df,
            profile,
        )

    except Exception as e:

        st.error(
            f"Unable to profile the dataset: {e}"
        )
        st.stop()

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{profile['rows']:,}")

    with col2:
        st.metric("Columns", f"{profile['columns']:,}")

    with col3:
        st.metric(
            "Missing Values",
            f"{profile['total_missing']:,}",
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            f"{profile['duplicate_rows']:,}",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Numeric Columns",
            f"{len(profile['numeric_columns']):,}",
        )

    with col2:
        st.metric(
            "Categorical Columns",
            f"{len(profile['categorical_columns']):,}",
        )

    with col3:
        st.metric(
            "Date/Time Columns",
            f"{len(profile['datetime_columns']):,}",
        )

    # ========================================================
    # DATASET HEALTH
    # ========================================================

    st.subheader("🩺 Dataset Health")

    total_issues = 0

    if profile["total_missing"] > 0:
        total_issues += 1

    if profile["duplicate_rows"] > 0:
        total_issues += 1

    if profile["constant_columns"]:
        total_issues += 1

    if profile["high_cardinality_columns"]:
        total_issues += 1

    if total_issues == 0:

        st.success(
            "Dataset health looks good. "
            "No major data-quality issues were detected."
        )

    else:

        st.warning(
            f"{total_issues} potential data-quality "
            "issue(s) detected. Review the sections below."
        )

    # ========================================================
    # QUALITY SUMMARY CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if profile["total_missing"] == 0:
            st.success("Missing Data\n\nNone detected")
        else:
            st.warning(
                f"Missing Data\n\n"
                f"{profile['total_missing']:,} values"
            )

    with col2:

        if profile["duplicate_rows"] == 0:
            st.success("Duplicates\n\nNone detected")
        else:
            st.warning(
                f"Duplicates\n\n"
                f"{profile['duplicate_rows']:,} rows"
            )

    with col3:

        if profile["constant_columns"]:
            st.warning(
                "Constant Columns\n\n"
                f"{len(profile['constant_columns'])}"
            )
        else:
            st.success("Constant Columns\n\nNone")

    with col4:

        if profile["high_cardinality_columns"]:
            st.info(
                "High Cardinality\n\n"
                f"{len(profile['high_cardinality_columns'])}"
            )
        else:
            st.success("High Cardinality\n\nNone")

    # ========================================================
    # AUTOMATIC DATASET INSIGHTS
    # ========================================================

    st.subheader("💡 Automatic Dataset Insights")

    if insights:

        for insight in insights:
            st.info(f"💡 {insight}")

    else:

        st.info(
            "No automatic insights were generated."
        )

    # ========================================================
    # MISSING VALUE ANALYSIS
    # ========================================================

    if profile["missing_columns"]:

        st.subheader("⚠️ Missing Value Analysis")

        missing_df = pd.DataFrame(
            [
                {
                    "Column": column,
                    "Missing Values": details["count"],
                    "Missing %": details["percentage"],
                }
                for column, details
                in profile["missing_columns"].items()
            ]
        )

        st.dataframe(
            missing_df,
            use_container_width=True,
        )

        chart = px.bar(
            missing_df,
            x="Column",
            y="Missing Values",
            text="Missing Values",
            title="Missing Values by Column",
        )

        chart.update_traces(
            textposition="outside"
        )

        chart.update_layout(
            xaxis_title="Column",
            yaxis_title="Missing Values",
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
            key="missing_values_chart",
        )

    else:

        with st.expander(
            "⚠️ Missing Value Analysis",
            expanded=False,
        ):

            st.success(
                "No missing values were detected "
                "in the dataset."
            )

    # ========================================================
    # DUPLICATE ANALYSIS
    # ========================================================

    with st.expander(
        "🔍 Duplicate Analysis",
        expanded=False,
    ):

        if profile["duplicate_rows"] > 0:

            st.warning(
                f"{profile['duplicate_rows']:,} "
                "duplicate row(s) detected."
            )

        else:

            st.success(
                "No duplicate rows detected."
            )

    # ========================================================
    # CONSTANT COLUMN ANALYSIS
    # ========================================================

    with st.expander(
        "📌 Constant Column Analysis",
        expanded=False,
    ):

        if profile["constant_columns"]:

            st.warning(
                "The following columns contain "
                "only one unique value:"
            )

            st.write(profile["constant_columns"])

        else:

            st.success(
                "No constant columns detected."
            )

    # ========================================================
    # HIGH CARDINALITY ANALYSIS
    # ========================================================

    with st.expander(
        "🧬 High Cardinality Analysis",
        expanded=False,
    ):

        if profile["high_cardinality_columns"]:

            st.info(
                "The following columns have a "
                "very high percentage of unique values:"
            )

            st.write(
                profile["high_cardinality_columns"]
            )

            st.caption(
                "High cardinality is not necessarily "
                "an error. Identifier columns often "
                "have high cardinality."
            )

        else:

            st.success(
                "No high-cardinality columns detected."
            )

    # ========================================================
    # LOW-CARDINALITY NUMERIC COLUMNS
    # ========================================================

    with st.expander(
        "🔢 Low-Cardinality Numeric Columns",
        expanded=False,
    ):

        if profile["low_cardinality_numeric"]:

            st.info(
                "These numeric columns contain "
                "relatively few unique values:"
            )

            st.write(
                profile["low_cardinality_numeric"]
            )

            st.caption(
                "These columns may represent "
                "categories, flags, scores, or codes."
            )

        else:

            st.success(
                "No low-cardinality numeric "
                "columns detected."
            )

    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    with st.expander(
        "🔍 Preview Dataset",
        expanded=False,
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

        dataset_csv = df.to_csv(index=False).encode("utf-8")
        dataset_excel = dataframe_to_excel_bytes(df, "Dataset")
        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download full dataset (CSV)",
                data=dataset_csv,
                file_name="dataset_export.csv",
                mime="text/csv",
                key=f"download_dataset_csv_{uploaded_file.name}",
            )
        with download_col2:
            st.download_button(
                "Download full dataset (Excel)",
                data=dataset_excel,
                file_name="dataset_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_dataset_xlsx_{uploaded_file.name}",
            )

    # ========================================================
    # COLUMN INFORMATION
    # ========================================================

    with st.expander(
        "📋 Column Information",
        expanded=False,
    ):

        st.dataframe(
            profile["column_details"],
            use_container_width=True,
        )

    # ========================================================
    # NUMERIC STATISTICS
    # ========================================================

    with st.expander(
        "📈 Numeric Statistics",
        expanded=False,
    ):

        if not profile["numeric_summary"].empty:

            st.dataframe(
                profile["numeric_summary"],
                use_container_width=True,
            )

        else:

            st.info(
                "No numeric columns were detected."
            )

    # ========================================================
    # CHAT HISTORY INITIALIZATION
    # ========================================================

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ========================================================
    # ASK YOUR DATASET
    # ========================================================

    st.subheader("💬 Ask Your Dataset")

    question = st.chat_input(
        "Example: Show employee count by department"
    )

    # ========================================================
    # QUESTION PROCESSING
    # ========================================================

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        try:

            # ------------------------------------------------
            # ROUTE QUESTION
            # ------------------------------------------------

            route = route_question(question)

            # ------------------------------------------------
            # UNDERSTAND QUESTION
            # ------------------------------------------------

            if route["route"] == "direct":

                request = {
                    "intent": route["intent"],
                }

            else:

                request = understand_question(
                    question,
                    df,
                )

            # ------------------------------------------------
            # ANALYZE DATA
            # ------------------------------------------------

            result = analyze_data(
                df,
                request,
            )

            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            if result.get("success"):

                answer = result.get(
                    "answer",
                    "Analysis completed.",
                )

            else:

                # When the interpreter deliberately requests clarification,
                # show that specific guidance instead of the analyzer's
                # generic "could not understand" fallback.
                semantic_warning = request.get("semantic_warning")
                if (
                    request.get("intent") == "unknown"
                    and semantic_warning
                ):
                    answer = semantic_warning
                else:
                    answer = result.get(
                        "answer",
                        "Unable to analyze the question.",
                    )

        except Exception as e:

            result = {
                "success": False,
                "result_type": "error",
                "answer": (
                    "An error occurred while processing "
                    f"your question: {e}"
                ),
            }

            answer = result["answer"]

        # ----------------------------------------------------
        # SAVE AI RESPONSE
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # ====================================================
        # AI ANSWER
        # ====================================================

        st.markdown("### 🤖 AI Answer")
        st.markdown(answer)

        # ====================================================
        # NUMBER RESULT
        # ====================================================

        if (
            result.get("success")
            and result.get("result_type") == "table"
        ):

            table_df = result.get("value")

            if isinstance(table_df, pd.DataFrame) and not table_df.empty:
                st.subheader("📋 Average of Numeric Columns")
                st.dataframe(
                    table_df,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("The analysis completed, but no table was returned.")

        elif (
            result.get("success")
            and result.get("result_type") == "number"
        ):

            value = result["value"]

            if isinstance(value, (int, float)):

                display_value = (
                    f"{value:,.2f}"
                    if isinstance(value, float)
                    and not value.is_integer()
                    else f"{value:,.0f}"
                )

            else:

                display_value = str(value)

            st.metric(
                "Result",
                display_value,
            )

        # ====================================================
        # PERCENTAGE RESULT
        # ====================================================

        elif (
            result.get("success")
            and result.get("result_type") == "percentage"
        ):

            st.metric(
                "Percentage",
                f"{result['value']:.2f}%",
            )

        # ====================================================
        # GROUPED RESULT
        # ====================================================

        elif (
            result.get("success")
            and result.get("result_type") == "dictionary"
        ):

            result_df = prepare_grouped_dataframe(
                result.get("value")
            )

            if result_df.empty:

                st.info(
                    "The analysis completed, but no "
                    "grouped results were available."
                )

            else:

                percentage_mode = is_percentage_question(
                    question
                )

                # --------------------------------------------
                # CALCULATE PERCENTAGES WHEN REQUESTED
                # --------------------------------------------

                if percentage_mode:

                    result_df = add_percentage_column(
                        result_df
                    )

                    st.info(
                        "Percentages are calculated using "
                        "the total of the displayed groups."
                    )

                # --------------------------------------------
                # DETAILED RESULTS
                # --------------------------------------------

                st.subheader("📋 Detailed Results")

                display_df = format_grouped_dataframe(
                    result_df,
                    percentage_mode,
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                # --------------------------------------------
                # VISUALIZATION
                # --------------------------------------------

                st.subheader("📈 Visualization")

                chart, chart_type = (
                    create_intelligent_chart(
                        result_df,
                        question,
                    )
                )

                if chart is not None:

                    st.caption(
                        "Automatically selected chart: "
                        f"**{chart_type.capitalize()} Chart**"
                    )

                    chart_key = (
                        f"analysis_chart_"
                        f"{len(st.session_state.chat_history)}_"
                        f"{chart_type}"
                    )

                    st.plotly_chart(
                        chart,
                        use_container_width=True,
                        key=chart_key,
                    )

                else:

                    st.info(
                        "No suitable visualization "
                        "could be generated for this result."
                    )

                # --------------------------------------------
                # HIGHEST ATTRITION
                # --------------------------------------------

                if result.get("highest_group") is not None:

                    highest_group = result[
                        "highest_group"
                    ]

                    highest_value = result[
                        "highest_value"
                    ]

                    st.info(
                        f"📌 Highest attrition: "
                        f"**{highest_group}** "
                        f"({highest_value:.2f}%)"
                    )

        # ====================================================
        # DISTRIBUTION RESULT
        # ====================================================

        elif (
            result.get("success")
            and result.get("result_type") == "distribution"
        ):

            st.subheader("📈 Visualization")

            chart, chart_type = (
                create_dataset_visualization(
                    df,
                    result,
                    question,
                )
            )

            if chart is not None:

                st.caption(
                    "Automatically selected chart: "
                    f"**{chart_type.capitalize()} Chart**"
                )

                chart_key = (
                    f"dataset_chart_"
                    f"{len(st.session_state.chat_history)}_"
                    f"{chart_type}"
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    key=chart_key,
                )

            else:

                st.info(
                    "No suitable visualization "
                    "could be generated."
                )

        # ====================================================
        # RELATIONSHIP RESULT
        # ====================================================

        elif (
            result.get("success")
            and result.get("result_type") == "relationship"
        ):

            st.subheader("📈 Visualization")

            chart, chart_type = (
                create_dataset_visualization(
                    df,
                    result,
                    question,
                )
            )

            if chart is not None:

                st.caption(
                    "Automatically selected chart: "
                    f"**{chart_type.capitalize()} Chart**"
                )

                chart_key = (
                    f"dataset_chart_"
                    f"{len(st.session_state.chat_history)}_"
                    f"{chart_type}"
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    key=chart_key,
                )

            else:

                st.info(
                    "No suitable visualization "
                    "could be generated."
                )

        # ====================================================
        # EXPORT CURRENT ANALYSIS RESULT
        # ====================================================

        export_df = exportable_result_dataframe(result)
        if export_df is not None:
            st.subheader("⬇️ Export Analysis Result")
            export_csv = export_df.to_csv(index=False).encode("utf-8")
            export_xlsx = dataframe_to_excel_bytes(export_df, "Analysis")
            export_col1, export_col2 = st.columns(2)
            export_key = len(st.session_state.chat_history)
            with export_col1:
                st.download_button(
                    "Download result (CSV)",
                    data=export_csv,
                    file_name="analysis_result.csv",
                    mime="text/csv",
                    key=f"download_result_csv_{export_key}",
                )
            with export_col2:
                st.download_button(
                    "Download result (Excel)",
                    data=export_xlsx,
                    file_name="analysis_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_result_xlsx_{export_key}",
                )

    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    if st.session_state.chat_history:

        st.divider()

        st.subheader("🕘 Conversation History")

        for message in st.session_state.chat_history:

            if message["role"] == "user":

                st.markdown(
                    f"**👤 You:** {message['content']}"
                )

            else:

                st.markdown(
                    f"**🤖 AI:** {message['content']}"
                )

else:

    st.info(
        "Upload a CSV or Excel dataset to get started."
    )
