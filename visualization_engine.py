import pandas as pd
import plotly.express as px

from dataset_understanding import understand_dataset


# ============================================================
# SEMANTIC VISUALIZATION CONTEXT
# ============================================================

def get_visualization_context(df):
    """
    Build semantic information about the dataset so that
    visualization choices avoid identifiers, constants, and
    unsuitable columns.
    """

    understanding = understand_dataset(df)

    return {
        "column_roles": understanding.get("column_roles", {}),
        "identifier_columns": understanding.get("identifier_columns", []),
        "constant_columns": understanding.get("constant_columns", []),
        "numeric_columns": understanding.get("numeric_columns", []),
        "categorical_columns": understanding.get("categorical_columns", []),
        "ordinal_columns": understanding.get("ordinal_columns", []),
        "datetime_columns": understanding.get("datetime_columns", []),
        "usable_numeric_columns": understanding.get(
            "usable_numeric_columns", []
        ),
        "usable_group_columns": understanding.get(
            "usable_group_columns", []
        ),
    }


# ============================================================
# COLUMN VALIDATION
# ============================================================

def is_valid_chart_dimension(column, context):
    """
    Check whether a dataset column is appropriate for use
    as a chart dimension.

    Valid:
        - categorical
        - ordinal
        - datetime

    Invalid:
        - identifiers
        - constant columns
        - numeric measures
        - unknown columns
    """

    if column is None:
        return False

    if column in context.get("identifier_columns", []):
        return False

    if column in context.get("constant_columns", []):
        return False

    role = context.get("column_roles", {}).get(column)

    return role in ["categorical", "ordinal", "datetime"]


def is_valid_numeric_measure(column, context):
    """
    Check whether a dataset column is appropriate for use
    as a numeric chart measure.
    """

    if column is None:
        return False

    if column in context.get("identifier_columns", []):
        return False

    if column in context.get("constant_columns", []):
        return False

    if column in context.get("usable_numeric_columns", []):
        return True

    role = context.get("column_roles", {}).get(column)

    return role == "numeric"


# ============================================================
# BASIC CHART FUNCTIONS
# ============================================================

def create_bar_chart(
    result_df,
    x_column="Category",
    y_column="Value",
    title="Analysis Result"
):
    """
    Create a bar chart from an analysis result dataframe.
    """

    if result_df is None or result_df.empty:
        return None

    if x_column not in result_df.columns:
        return None

    if y_column not in result_df.columns:
        return None

    return px.bar(
        result_df,
        x=x_column,
        y=y_column,
        title=title
    )


def create_pie_chart(
    result_df,
    names_column="Category",
    values_column="Value",
    title="Analysis Result"
):
    """
    Create a pie chart from an analysis result dataframe.
    """

    if result_df is None or result_df.empty:
        return None

    if names_column not in result_df.columns:
        return None

    if values_column not in result_df.columns:
        return None

    return px.pie(
        result_df,
        names=names_column,
        values=values_column,
        title=title
    )


def create_line_chart(
    result_df,
    x_column="Category",
    y_column="Value",
    title="Analysis Result"
):
    """
    Create a line chart from an analysis result dataframe.
    """

    if result_df is None or result_df.empty:
        return None

    if x_column not in result_df.columns:
        return None

    if y_column not in result_df.columns:
        return None

    return px.line(
        result_df,
        x=x_column,
        y=y_column,
        title=title,
        markers=True
    )


def create_histogram(
    df,
    column,
    title="Distribution"
):
    """
    Create a histogram for a numeric dataset column.
    """

    if df is None or df.empty:
        return None

    if column not in df.columns:
        return None

    return px.histogram(
        df,
        x=column,
        title=title
    )


def create_scatter_chart(
    df,
    x_column,
    y_column,
    title="Relationship Between Variables"
):
    """
    Create a scatter plot between two numeric columns.
    """

    if df is None or df.empty:
        return None

    if x_column not in df.columns:
        return None

    if y_column not in df.columns:
        return None

    return px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=title
    )


# ============================================================
# CHART TYPE DETECTION
# ============================================================

def detect_chart_type(result_df, question=""):
    """
    Automatically choose an appropriate chart type based
    on the question and result size.
    """

    question_lower = str(question).lower()

    # --------------------------------------------------------
    # Time-related questions
    # --------------------------------------------------------

    time_keywords = [
        "trend",
        "over time",
        "monthly",
        "month",
        "yearly",
        "year",
        "weekly",
        "week",
        "daily",
        "day",
        "timeline",
        "time series"
    ]

    if any(keyword in question_lower for keyword in time_keywords):
        return "line"

    # --------------------------------------------------------
    # Distribution-related questions
    # --------------------------------------------------------

    distribution_keywords = [
        "distribution",
        "spread",
        "histogram",
        "frequency distribution"
    ]

    if any(
        keyword in question_lower
        for keyword in distribution_keywords
    ):
        return "histogram"

    # --------------------------------------------------------
    # Proportion / share questions
    # --------------------------------------------------------

    proportion_keywords = [
        "share",
        "proportion",
        "percentage",
        "percent",
        "composition",
        "breakdown"
    ]

    if (
        any(
            keyword in question_lower
            for keyword in proportion_keywords
        )
        and result_df is not None
        and len(result_df) <= 8
    ):
        return "pie"

    # --------------------------------------------------------
    # Default grouped result
    # --------------------------------------------------------

    return "bar"


# ============================================================
# INTELLIGENT RESULT VISUALIZATION
# ============================================================

def create_intelligent_chart(
    result_df,
    question="",
    df=None
):
    """
    Automatically create a visualization for an analysis result.

    Standardized analysis results normally contain:

        Category
        Value

    These are result-level columns, not necessarily original
    dataset columns.

    Semantic validation is therefore applied differently:

    1. Original dataset columns are checked against the
       semantic dataset profile.

    2. Standardized result columns such as Category and Value
       are accepted because they represent already-aggregated
       analysis output.
    """

    if result_df is None or result_df.empty:
        return None, "none"

    context = None

    if df is not None:
        context = get_visualization_context(df)

    # --------------------------------------------------------
    # Identify standardized result columns
    # --------------------------------------------------------

    result_columns = list(result_df.columns)

    x_column = None
    y_column = None

    # Standard analysis-result format
    if "Category" in result_columns:
        x_column = "Category"

    if "Value" in result_columns:
        y_column = "Value"

    # --------------------------------------------------------
    # Fallback for non-standard result names
    # --------------------------------------------------------

    if x_column is None:
        possible_x_columns = [
            column
            for column in result_columns
            if column != "Value"
        ]

        if possible_x_columns:
            candidate = possible_x_columns[0]

            # If original dataset is available, validate it.
            if context is None:
                x_column = candidate
            elif is_valid_chart_dimension(candidate, context):
                x_column = candidate

    if y_column is None and len(result_columns) >= 2:
        possible_y_columns = [
            column
            for column in result_columns
            if column != x_column
        ]

        for candidate in possible_y_columns:
            if context is None:
                if pd.api.types.is_numeric_dtype(
                    result_df[candidate]
                ):
                    y_column = candidate
                    break
            else:
                # Standard result values are already aggregated.
                if candidate == "Value":
                    y_column = candidate
                    break

                if is_valid_numeric_measure(
                    candidate,
                    context
                ):
                    y_column = candidate
                    break

    # --------------------------------------------------------
    # Validate standardized result
    # --------------------------------------------------------

    if x_column is None:
        return None, "none"

    if y_column is None:
        return None, "none"

    if x_column not in result_df.columns:
        return None, "none"

    if y_column not in result_df.columns:
        return None, "none"

    # --------------------------------------------------------
    # Detect chart type
    # --------------------------------------------------------

    chart_type = detect_chart_type(
        result_df,
        question
    )

    title = question.strip()

    if not title:
        title = "Analysis Result"

    # --------------------------------------------------------
    # Create chart
    # --------------------------------------------------------

    if chart_type == "bar":
        chart = create_bar_chart(
            result_df,
            x_column,
            y_column,
            title
        )

    elif chart_type == "pie":
        chart = create_pie_chart(
            result_df,
            x_column,
            y_column,
            title
        )

    elif chart_type == "line":
        chart = create_line_chart(
            result_df,
            x_column,
            y_column,
            title
        )

    else:
        return None, "none"

    if chart is None:
        return None, "none"

    return chart, chart_type


# ============================================================
# DATASET-LEVEL VISUALIZATION
# ============================================================

def create_dataset_visualization(
    df,
    result,
    question=""
):
    """
    Create a visualization directly from the original dataset
    for distribution and relationship analysis results.
    """

    if df is None or df.empty:
        return None, "none"

    if result is None:
        return None, "none"

    context = get_visualization_context(df)

    result_type = result.get("result_type")

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    if result_type == "distribution":

        column = result.get("column")

        if not is_valid_numeric_measure(
            column,
            context
        ):
            return None, "none"

        chart = create_histogram(
            df,
            column,
            f"Distribution of {column}"
        )

        if chart is None:
            return None, "none"

        return chart, "histogram"

    # ========================================================
    # RELATIONSHIP
    # ========================================================

    if result_type == "relationship":

        x_column = result.get("x_column")
        y_column = result.get("y_column")

        if not is_valid_numeric_measure(
            x_column,
            context
        ):
            return None, "none"

        if not is_valid_numeric_measure(
            y_column,
            context
        ):
            return None, "none"

        chart = create_scatter_chart(
            df,
            x_column,
            y_column,
            f"{y_column} vs {x_column}"
        )

        if chart is None:
            return None, "none"

        return chart, "scatter"

    # ========================================================
    # GROUPED / AGGREGATED RESULTS
    # ========================================================

    if result_type == "dictionary":

        result_df = result.get("value")

        if isinstance(result_df, pd.DataFrame):
            return create_intelligent_chart(
                result_df,
                question,
                df
            )

    return None, "none"
