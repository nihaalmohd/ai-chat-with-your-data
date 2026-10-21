import pandas as pd

from data_profiler import profile_dataset


# ==========================================================
# DATASET UNDERSTANDING
# ==========================================================

def understand_dataset(df):
    """
    Build a higher-level understanding of a dataset.

    This function uses the automatic dataset profiler and
    converts the profiling information into a structure
    that can be used by the AI engine, analysis engine,
    visualization engine, and Streamlit application.

    The function does NOT modify the original dataframe.
    """

    # ======================================================
    # CREATE DATASET PROFILE
    # ======================================================

    profile = profile_dataset(df)


    # ======================================================
    # GET SEMANTIC ROLE INFORMATION
    # ======================================================

    column_roles = profile.get(
        "column_roles",
        {}
    )

    role_summary = profile.get(
        "role_summary",
        {}
    )


    identifier_columns = role_summary.get(
        "identifier",
        []
    )

    numeric_columns = role_summary.get(
        "numeric",
        []
    )

    categorical_columns = role_summary.get(
        "categorical",
        []
    )

    ordinal_columns = role_summary.get(
        "ordinal",
        []
    )

    datetime_columns = role_summary.get(
        "datetime",
        []
    )


    # ======================================================
    # USABLE NUMERIC MEASURES
    # ======================================================
    #
    # Identifier columns should not normally be treated
    # as analytical measures.
    #
    # Constant columns are also excluded because they
    # provide no analytical variation.
    # ======================================================

    constant_columns = profile.get(
        "constant_columns",
        []
    )


    usable_numeric_columns = [

        column

        for column in numeric_columns

        if column not in identifier_columns

        and column not in constant_columns

    ]


    # ======================================================
    # USABLE GROUPING COLUMNS
    # ======================================================
    #
    # Categorical and ordinal columns are generally useful
    # for grouping and comparison.
    #
    # Identifier columns are excluded.
    # ======================================================

    usable_group_columns = [

        column

        for column in (

            categorical_columns
            + ordinal_columns

        )

        if column not in identifier_columns

        and column not in constant_columns

    ]


    # ======================================================
    # CHART DIMENSIONS
    # ======================================================

    chart_dimension_columns = [

        column

        for column in (

            categorical_columns
            + ordinal_columns

        )

        if column not in identifier_columns

        and column not in constant_columns

    ]


    # ======================================================
    # CHART MEASURES
    # ======================================================

    chart_measure_columns = [

        column

        for column in usable_numeric_columns

    ]


    # ======================================================
    # FILTER COLUMNS
    # ======================================================
    #
    # Both categorical and ordinal fields can normally be
    # useful for filtering.
    # ======================================================

    filter_columns = [

        column

        for column in (

            categorical_columns
            + ordinal_columns

        )

        if column not in identifier_columns

    ]


    # ======================================================
    # HIGH CARDINALITY COLUMNS
    # ======================================================

    high_cardinality_columns = profile.get(
        "high_cardinality_columns",
        []
    )


    # ======================================================
    # MISSING VALUE INFORMATION
    # ======================================================

    missing_columns = profile.get(
        "missing_columns",
        {}
    )


    # ======================================================
    # DATASET WARNINGS
    # ======================================================

    warnings = []


    # ------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------

    if profile.get(
        "total_missing",
        0
    ) > 0:

        warnings.append({

            "type":
                "missing_values",

            "message":
                (
                    f"Dataset contains "
                    f"{profile['total_missing']:,} "
                    f"missing values."
                )

        })


    # ------------------------------------------------------
    # DUPLICATES
    # ------------------------------------------------------

    if profile.get(
        "duplicate_rows",
        0
    ) > 0:

        warnings.append({

            "type":
                "duplicate_rows",

            "message":
                (
                    f"Dataset contains "
                    f"{profile['duplicate_rows']:,} "
                    f"duplicate rows."
                )

        })


    # ------------------------------------------------------
    # CONSTANT COLUMNS
    # ------------------------------------------------------

    if constant_columns:

        warnings.append({

            "type":
                "constant_columns",

            "message":
                (
                    "Constant columns should generally "
                    "not be used for analytical comparisons: "
                    + ", ".join(
                        constant_columns
                    )
                )

        })


    # ------------------------------------------------------
    # HIGH CARDINALITY
    # ------------------------------------------------------

    if high_cardinality_columns:

        warnings.append({

            "type":
                "high_cardinality",

            "message":
                (
                    "High-cardinality columns may not be "
                    "suitable for standard categorical charts: "
                    + ", ".join(
                        high_cardinality_columns
                    )
                )

        })


    # ======================================================
    # COLUMN INFORMATION
    # ======================================================

    column_information = {}


    for column in df.columns:

        column_information[column] = {

            "role":
                column_roles.get(
                    column,
                    "unknown"
                ),

            "data_type":
                str(
                    df[column].dtype
                ),

            "unique_values":
                int(
                    df[column].nunique()
                ),

            "missing_values":
                int(
                    df[column].isnull().sum()
                )

        }


    # ======================================================
    # DATASET SUMMARY
    # ======================================================

    dataset_summary = {

        "rows":
            profile.get(
                "rows",
                0
            ),

        "columns":
            profile.get(
                "columns",
                0
            ),

        "total_missing":
            profile.get(
                "total_missing",
                0
            ),

        "duplicate_rows":
            profile.get(
                "duplicate_rows",
                0
            ),

        "identifier_count":
            len(
                identifier_columns
            ),

        "numeric_count":
            len(
                numeric_columns
            ),

        "categorical_count":
            len(
                categorical_columns
            ),

        "ordinal_count":
            len(
                ordinal_columns
            ),

        "datetime_count":
            len(
                datetime_columns
            )

    }


    # ======================================================
    # RETURN DATASET UNDERSTANDING
    # ======================================================

    return {

        # --------------------------------------------------
        # ORIGINAL PROFILE
        # --------------------------------------------------

        "profile":
            profile,


        # --------------------------------------------------
        # DATASET SUMMARY
        # --------------------------------------------------

        "dataset_summary":
            dataset_summary,


        # --------------------------------------------------
        # COLUMN ROLES
        # --------------------------------------------------

        "column_roles":
            column_roles,


        # --------------------------------------------------
        # COLUMN INFORMATION
        # --------------------------------------------------

        "column_information":
            column_information,


        # --------------------------------------------------
        # SEMANTIC GROUPS
        # --------------------------------------------------

        "identifier_columns":
            identifier_columns,

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "ordinal_columns":
            ordinal_columns,

        "datetime_columns":
            datetime_columns,


        # --------------------------------------------------
        # ANALYTICAL COLUMNS
        # --------------------------------------------------

        "usable_numeric_columns":
            usable_numeric_columns,

        "usable_group_columns":
            usable_group_columns,

        "filter_columns":
            filter_columns,


        # --------------------------------------------------
        # VISUALIZATION COLUMNS
        # --------------------------------------------------

        "chart_dimension_columns":
            chart_dimension_columns,

        "chart_measure_columns":
            chart_measure_columns,


        # --------------------------------------------------
        # QUALITY INFORMATION
        # --------------------------------------------------

        "constant_columns":
            constant_columns,

        "high_cardinality_columns":
            high_cardinality_columns,

        "missing_columns":
            missing_columns,

        "warnings":
            warnings

    }
