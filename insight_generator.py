import pandas as pd


def generate_insights(df, profile):
    """
    Generate automatic dataset insights
    from the profiler results.

    This function is dataset-agnostic.
    It does not assume HR-specific columns.
    """

    insights = []


    # ==================================================
    # BASIC DATASET SIZE
    # ==================================================

    rows = profile["rows"]
    columns = profile["columns"]


    if rows > 0:

        insights.append(
            f"The dataset contains "
            f"{rows:,} records across "
            f"{columns:,} columns."
        )

    else:

        insights.append(
            "The uploaded dataset is empty."
        )


    # ==================================================
    # NUMERIC AND CATEGORICAL STRUCTURE
    # ==================================================

    numeric_count = len(
        profile["numeric_columns"]
    )

    categorical_count = len(
        profile["categorical_columns"]
    )

    datetime_count = len(
        profile["datetime_columns"]
    )


    if numeric_count > 0:

        insights.append(
            f"The dataset contains "
            f"{numeric_count:,} numeric "
            f"column(s)."
        )


    if categorical_count > 0:

        insights.append(
            f"The dataset contains "
            f"{categorical_count:,} "
            f"categorical column(s)."
        )


    if datetime_count > 0:

        insights.append(
            f"The dataset contains "
            f"{datetime_count:,} "
            f"date/time column(s)."
        )


    # ==================================================
    # MISSING DATA
    # ==================================================

    total_missing = profile[
        "total_missing"
    ]


    if total_missing > 0:

        missing_columns = profile[
            "missing_columns"
        ]


        highest_missing_column = None

        highest_missing_percentage = 0


        for column, details in (
            missing_columns.items()
        ):

            percentage = details[
                "percentage"
            ]


            if percentage > highest_missing_percentage:

                highest_missing_percentage = (
                    percentage
                )

                highest_missing_column = column


        if highest_missing_column:

            insights.append(

                f"The dataset contains "
                f"{total_missing:,} missing values. "
                f"The column with the highest "
                f"missing percentage is "
                f"'{highest_missing_column}' "
                f"at "
                f"{highest_missing_percentage:.2f}%."

            )

    else:

        insights.append(
            "No missing values were detected "
            "in the dataset."
        )


    # ==================================================
    # DUPLICATES
    # ==================================================

    duplicate_rows = profile[
        "duplicate_rows"
    ]


    if duplicate_rows > 0:

        duplicate_percentage = (

            duplicate_rows
            / rows
            * 100

        ) if rows > 0 else 0


        insights.append(

            f"The dataset contains "
            f"{duplicate_rows:,} duplicate "
            f"row(s), representing approximately "
            f"{duplicate_percentage:.2f}% "
            f"of the dataset."

        )

    else:

        insights.append(
            "No duplicate rows were detected."
        )


    # ==================================================
    # CONSTANT COLUMNS
    # ==================================================

    constant_columns = profile[
        "constant_columns"
    ]


    if constant_columns:

        insights.append(

            f"{len(constant_columns):,} "
            f"constant column(s) were detected: "
            + ", ".join(
                constant_columns
            )
            + ". These columns contain "
              "little or no variation."

        )

    else:

        insights.append(
            "No constant columns were detected."
        )


    # ==================================================
    # HIGH CARDINALITY
    # ==================================================

    high_cardinality_columns = profile[
        "high_cardinality_columns"
    ]


    if high_cardinality_columns:

        insights.append(

            f"{len(high_cardinality_columns):,} "
            f"high-cardinality column(s) "
            f"were detected: "
            + ", ".join(
                high_cardinality_columns
            )
            + ". These columns may require "
              "additional investigation, "
              "especially if they represent "
              "identifiers."

        )

    else:

        insights.append(
            "No high-cardinality columns "
            "were detected."
        )


    # ==================================================
    # LOW-CARDINALITY NUMERIC COLUMNS
    # ==================================================

    low_cardinality_numeric = profile[
        "low_cardinality_numeric"
    ]


    if low_cardinality_numeric:

        insights.append(

            f"{len(low_cardinality_numeric):,} "
            f"numeric column(s) have relatively "
            f"few unique values: "
            + ", ".join(
                low_cardinality_numeric
            )
            + ". These may represent "
              "flags, codes, scores, or "
              "categorical-like variables."

        )


    # ==================================================
    # CATEGORICAL DOMINANCE
    # ==================================================

    categorical_dominance = []


    for column in profile[
        "categorical_columns"
    ]:

        if rows == 0:

            continue


        value_counts = (
            df[column]
            .value_counts(
                dropna=True
            )
        )


        if value_counts.empty:

            continue


        top_value = value_counts.index[0]

        top_count = int(
            value_counts.iloc[0]
        )


        percentage = (
            top_count
            / rows
            * 100
        )


        if percentage >= 80:

            categorical_dominance.append({

                "column": column,

                "value": str(
                    top_value
                ),

                "percentage": round(
                    percentage,
                    2
                )

            })


    # ==================================================
    # ADD CATEGORICAL INSIGHTS
    # ==================================================

    for item in categorical_dominance:

        insights.append(

            f"The category "
            f"'{item['value']}' accounts for "
            f"{item['percentage']:.2f}% of "
            f"the values in "
            f"'{item['column']}'. "
            f"This column may be highly "
            f"imbalanced."

        )


    # ==================================================
    # NUMERIC RANGE INSIGHTS
    # ==================================================

    numeric_range_insights = []


    for column in profile[
        "numeric_columns"
    ]:

        numeric_data = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()


        if numeric_data.empty:

            continue


        minimum = numeric_data.min()

        maximum = numeric_data.max()


        if minimum == maximum:

            continue


        numeric_range_insights.append({

            "column": column,

            "minimum": float(
                minimum
            ),

            "maximum": float(
                maximum
            )

        })


    # ==================================================
    # ADD NUMERIC RANGE INSIGHTS
    # ==================================================

    # Limit this section to avoid
    # overwhelming the user.

    for item in numeric_range_insights[:5]:

        insights.append(

            f"'{item['column']}' ranges "
            f"from "
            f"{item['minimum']:,.2f} "
            f"to "
            f"{item['maximum']:,.2f}."

        )


    # ==================================================
    # RETURN INSIGHTS
    # ==================================================

    return insights
