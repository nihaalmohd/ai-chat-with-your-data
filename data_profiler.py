import pandas as pd


# ==========================================================
# SEMANTIC COLUMN ROLE DETECTION
# ==========================================================

def detect_column_roles(df):
    """
    Detect the likely semantic role of each column.

    Possible roles:
    - identifier
    - numeric
    - categorical
    - ordinal
    - datetime

    This is a heuristic-based system.
    It does not modify the original pandas data types.
    """

    column_roles = {}

    identifier_columns = []

    ordinal_columns = []

    rows = len(df)


    # ======================================================
    # CHECK EACH COLUMN
    # ======================================================

    for column in df.columns:

        series = df[column]

        column_lower = str(column).lower().strip()

        unique_count = series.nunique(
            dropna=True
        )


        # ==================================================
        # DATETIME
        # ==================================================

        if pd.api.types.is_datetime64_any_dtype(series):

            column_roles[column] = "datetime"

            continue


        # ==================================================
        # IDENTIFIER DETECTION
        # ==================================================

        identifier_keywords = [

            "id",

            "identifier",

            "number",

            "code"

        ]


        looks_like_identifier = any(

            keyword in column_lower

            for keyword in identifier_keywords

        )


        high_uniqueness = (

            rows > 0

            and unique_count / rows >= 0.90

            and unique_count > 10

        )


        if (

            looks_like_identifier

            and high_uniqueness

        ):

            column_roles[column] = "identifier"

            identifier_columns.append(
                column
            )

            continue


        # ==================================================
        # CATEGORICAL TEXT
        # ==================================================

        if pd.api.types.is_object_dtype(series):

            column_roles[column] = "categorical"

            continue


        # ==================================================
        # BOOLEAN
        # ==================================================

        if pd.api.types.is_bool_dtype(series):

            column_roles[column] = "categorical"

            continue


        # ==================================================
        # NUMERIC
        # ==================================================

        if pd.api.types.is_numeric_dtype(series):


            # ----------------------------------------------
            # ORDINAL DETECTION
            # ----------------------------------------------

            ordinal_keywords = [

                "level",

                "rating",

                "satisfaction",

                "education",

                "involvement",

                "performance",

                "worklife",

                "stockoption"

            ]


            looks_like_ordinal = any(

                keyword in column_lower

                for keyword in ordinal_keywords

            )


            if (

                looks_like_ordinal

                and 1 < unique_count <= 10

            ):

                column_roles[column] = "ordinal"

                ordinal_columns.append(
                    column
                )

            else:

                column_roles[column] = "numeric"


            continue


        # ==================================================
        # FALLBACK
        # ==================================================

        column_roles[column] = "categorical"


    # ======================================================
    # RETURN SEMANTIC INFORMATION
    # ======================================================

    return {

        "column_roles":
            column_roles,

        "identifier_columns":
            identifier_columns,

        "ordinal_columns":
            ordinal_columns

    }


# ==========================================================
# DATASET PROFILER
# ==========================================================

def profile_dataset(df):
    """
    Create an automatic profile of any dataset.

    The profile contains:

    - Basic dataset information
    - Missing values
    - Duplicate rows
    - Data types
    - Unique values
    - Column details
    - Numeric summary
    - Data quality issues
    - Semantic column roles
    """


    # ======================================================
    # BASIC DATASET INFORMATION
    # ======================================================

    rows = int(
        df.shape[0]
    )

    columns = int(
        df.shape[1]
    )


    # ======================================================
    # MISSING VALUES
    # ======================================================

    missing_by_column = (
        df.isnull().sum()
    )

    total_missing = int(
        missing_by_column.sum()
    )

    columns_with_missing = int(
        (missing_by_column > 0).sum()
    )


    # ======================================================
    # DUPLICATE ROWS
    # ======================================================

    duplicate_rows = int(
        df.duplicated().sum()
    )


    # ======================================================
    # DATA TYPES
    # ======================================================

    numeric_columns = (

        df.select_dtypes(
            include="number"
        )

        .columns

        .tolist()

    )


    categorical_columns = (

        df.select_dtypes(

            include=[
                "object",
                "category",
                "bool"
            ]

        )

        .columns

        .tolist()

    )


    datetime_columns = (

        df.select_dtypes(
            include=["datetime"]
        )

        .columns

        .tolist()

    )


    # ======================================================
    # UNIQUE VALUES
    # ======================================================

    unique_values = {

        column:

            int(
                df[column].nunique()
            )

        for column in df.columns

    }


    # ======================================================
    # COLUMN DETAILS
    # ======================================================

    column_details = []


    for column in df.columns:

        column_details.append({

            "Column":
                column,

            "Data Type":
                str(
                    df[column].dtype
                ),

            "Missing Values":
                int(
                    df[column].isnull().sum()
                ),

            "Unique Values":
                int(
                    df[column].nunique()
                ),

            "Unique %":
                round(

                    (

                        df[column].nunique()

                        / rows

                        * 100

                    )

                    if rows > 0

                    else 0,

                    2

                )

        })


    column_details_df = pd.DataFrame(
        column_details
    )


    # ======================================================
    # NUMERIC SUMMARY
    # ======================================================

    numeric_summary = pd.DataFrame()


    if len(numeric_columns) > 0:

        numeric_summary = (

            df[numeric_columns]

            .describe()

            .T

            .reset_index()

            .rename(

                columns={
                    "index": "Column"
                }

            )

        )


    # ======================================================
    # DATA QUALITY ISSUES
    # ======================================================

    quality_issues = []


    # ------------------------------------------------------
    # MISSING DATA
    # ------------------------------------------------------

    if total_missing > 0:

        quality_issues.append({

            "type":
                "warning",

            "title":
                "Missing Values",

            "message":

                f"Dataset contains "

                f"{total_missing:,} missing values "

                f"across "

                f"{columns_with_missing} columns."

        })

    else:

        quality_issues.append({

            "type":
                "success",

            "title":
                "Missing Values",

            "message":
                "No missing values detected."

        })


    # ------------------------------------------------------
    # DUPLICATES
    # ------------------------------------------------------

    if duplicate_rows > 0:

        quality_issues.append({

            "type":
                "warning",

            "title":
                "Duplicate Rows",

            "message":

                f"Dataset contains "

                f"{duplicate_rows:,} duplicate rows."

        })

    else:

        quality_issues.append({

            "type":
                "success",

            "title":
                "Duplicate Rows",

            "message":
                "No duplicate rows detected."

        })


    # ======================================================
    # CONSTANT COLUMNS
    # ======================================================

    constant_columns = []


    for column in df.columns:

        if (

            df[column].nunique(
                dropna=False
            )

            <= 1

        ):

            constant_columns.append(
                column
            )


    if constant_columns:

        quality_issues.append({

            "type":
                "warning",

            "title":
                "Constant Columns",

            "message":

                "Constant columns detected: "

                + ", ".join(
                    constant_columns
                )

        })

    else:

        quality_issues.append({

            "type":
                "success",

            "title":
                "Constant Columns",

            "message":
                "No constant columns detected."

        })


    # ======================================================
    # HIGH CARDINALITY COLUMNS
    # ======================================================

    high_cardinality_columns = []


    if rows > 0:

        for column in df.columns:

            unique_count = (

                df[column].nunique()

            )


            if (

                unique_count / rows >= 0.9

                and unique_count > 10

            ):

                high_cardinality_columns.append(
                    column
                )


    if high_cardinality_columns:

        quality_issues.append({

            "type":
                "info",

            "title":
                "High Cardinality",

            "message":

                "High-cardinality columns detected: "

                + ", ".join(
                    high_cardinality_columns
                )

        })

    else:

        quality_issues.append({

            "type":
                "success",

            "title":
                "High Cardinality",

            "message":
                "No high-cardinality columns detected."

        })


    # ======================================================
    # LOW CARDINALITY NUMERIC COLUMNS
    # ======================================================

    low_cardinality_numeric = []


    for column in numeric_columns:

        unique_count = (

            df[column].nunique()

        )


        if (

            unique_count <= 10

            and unique_count > 1

        ):

            low_cardinality_numeric.append(
                column
            )


    # ======================================================
    # MISSING VALUE DETAILS
    # ======================================================

    missing_columns = {}


    for column in df.columns:

        missing_count = int(

            df[column].isnull().sum()

        )


        if missing_count > 0:

            missing_percentage = (

                missing_count

                / rows

                * 100

            ) if rows > 0 else 0


            missing_columns[column] = {

                "count":
                    missing_count,

                "percentage":
                    round(
                        missing_percentage,
                        2
                    )

            }


    # ======================================================
    # SEMANTIC COLUMN ROLES
    # ======================================================

    semantic_roles = (

        detect_column_roles(df)

    )


    column_roles = (

        semantic_roles[
            "column_roles"
        ]

    )


    identifier_columns = (

        semantic_roles[
            "identifier_columns"
        ]

    )


    ordinal_columns = (

        semantic_roles[
            "ordinal_columns"
        ]

    )


    # ======================================================
    # ROLE SUMMARY
    # ======================================================

    role_summary = {

        "identifier":
            identifier_columns,

        "numeric": [

            column

            for column in df.columns

            if column_roles.get(column)
            == "numeric"

        ],

        "categorical": [

            column

            for column in df.columns

            if column_roles.get(column)
            == "categorical"

        ],

        "ordinal":
            ordinal_columns,

        "datetime": [

            column

            for column in df.columns

            if column_roles.get(column)
            == "datetime"

        ]

    }


    # ======================================================
    # RETURN PROFILE
    # ======================================================

    return {

        "rows":
            rows,

        "columns":
            columns,

        "total_missing":
            total_missing,

        "columns_with_missing":
            columns_with_missing,

        "duplicate_rows":
            duplicate_rows,

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "datetime_columns":
            datetime_columns,

        "unique_values":
            unique_values,

        "column_details":
            column_details_df,

        "numeric_summary":
            numeric_summary,

        "quality_issues":
            quality_issues,

        "constant_columns":
            constant_columns,

        "high_cardinality_columns":
            high_cardinality_columns,

        "low_cardinality_numeric":
            low_cardinality_numeric,

        "missing_columns":
            missing_columns,

        # --------------------------------------------------
        # SEMANTIC INFORMATION
        # --------------------------------------------------

        "column_roles":
            column_roles,

        "identifier_columns":
            identifier_columns,

        "ordinal_columns":
            ordinal_columns,

        "role_summary":
            role_summary

    }
