import pandas as pd


# =========================================================
# FIND COLUMN
# =========================================================

def find_column(
    df,
    requested_column
):

    if requested_column is None:
        return None

    requested_column = str(
        requested_column
    ).strip().lower()

    for column in df.columns:

        if (
            str(column)
            .strip()
            .lower()
            == requested_column
        ):

            return column

    return None


# =========================================================
# FIND VALUE
# =========================================================

def find_value(
    series,
    requested_value
):

    if requested_value is None:
        return None

    requested_value = str(
        requested_value
    ).strip().lower()

    for value in series.dropna().unique():

        if (
            str(value)
            .strip()
            .lower()
            == requested_value
        ):

            return value

    return None


# =========================================================
# ANALYZE DATA
# =========================================================

def analyze_data(df, request):

    if request is None:

        return {
            "success": False,
            "answer": "No analysis request was provided.",
            "result_type": "error"
        }

    intent = request.get("intent")


    # =========================================================
    # 1. TOTAL EMPLOYEES
    # =========================================================

    if intent == "total_employees":

        count = len(df)

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{count:,} records."
            ),
            "result_type": "number",
            "value": count
        }


    # =========================================================
    # 2. EMPLOYEES WHO LEFT
    # =========================================================

    if intent == "employees_left":

        attrition_column = find_column(
            df,
            "Attrition"
        )

        if attrition_column is None:

            return {
                "success": False,
                "answer": (
                    "The dataset does not contain "
                    "an Attrition column."
                ),
                "result_type": "error"
            }

        yes_value = find_value(
            df[attrition_column],
            "Yes"
        )

        if yes_value is None:

            return {
                "success": False,
                "answer": (
                    "The dataset does not contain "
                    "an Attrition value of Yes."
                ),
                "result_type": "error"
            }

        count = int(
            (
                df[attrition_column]
                == yes_value
            ).sum()
        )

        return {
            "success": True,
            "answer": (
                f"{count:,} employees "
                "left the company."
            ),
            "result_type": "number",
            "value": count
        }


    # =========================================================
    # 3. ATTRITION RATE
    # =========================================================

    if intent == "attrition_rate":

        attrition_column = find_column(
            df,
            "Attrition"
        )

        if attrition_column is None:

            return {
                "success": False,
                "answer": (
                    "The dataset does not contain "
                    "an Attrition column."
                ),
                "result_type": "error"
            }

        yes_value = find_value(
            df[attrition_column],
            "Yes"
        )

        if yes_value is None:

            return {
                "success": False,
                "answer": (
                    "The dataset does not contain "
                    "an Attrition value of Yes."
                ),
                "result_type": "error"
            }

        left_count = int(
            (
                df[attrition_column]
                == yes_value
            ).sum()
        )

        total_count = len(df)

        if total_count == 0:

            return {
                "success": False,
                "answer": "The dataset is empty.",
                "result_type": "error"
            }

        rate = (
            left_count
            / total_count
        ) * 100

        return {
            "success": True,
            "answer": (
                f"The attrition rate is "
                f"{rate:.2f}%."
            ),
            "result_type": "percentage",
            "value": round(rate, 2)
        }


    # =========================================================
    # 4. COLUMN COUNT
    # =========================================================

    if intent == "column_count":

        count = len(df.columns)

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{count:,} columns."
            ),
            "result_type": "number",
            "value": count
        }


    # =========================================================
    # 5. ROW COUNT
    # =========================================================

    if intent == "row_count":

        count = len(df)

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{count:,} rows."
            ),
            "result_type": "number",
            "value": count
        }


    # =========================================================
    # 6. MISSING VALUES
    # =========================================================

    if intent == "missing_values":

        total_missing = int(
            df.isnull()
            .sum()
            .sum()
        )

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{total_missing:,} "
                "missing values."
            ),
            "result_type": "number",
            "value": total_missing
        }


    # =========================================================
    # 7. GENERIC COUNT
    # =========================================================

    if intent == "count":

        count = len(df)

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{count:,} records."
            ),
            "result_type": "number",
            "value": count
        }


    # =========================================================
    # 8. FILTERED COUNT
    # =========================================================

    if intent == "filtered_count":

        requested_column = request.get(
            "filter_column"
        )

        requested_value = request.get(
            "filter_value"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        actual_value = find_value(
            df[actual_column],
            requested_value
        )

        if actual_value is None:

            return {
                "success": False,
                "answer": (
                    f"The value "
                    f"'{requested_value}' "
                    f"was not found in column "
                    f"'{actual_column}'."
                ),
                "result_type": "error"
            }

        filtered_df = df[
            df[actual_column]
            == actual_value
        ]

        count = len(filtered_df)

        return {
            "success": True,
            "answer": (
                f"There are {count:,} records "
                f"where '{actual_column}' "
                f"is '{actual_value}'."
            ),
            "result_type": "number",
            "value": count,
            "filter_column": actual_column,
            "filter_value": actual_value
        }


    # =========================================================
    # 9. AVERAGE
    # =========================================================

    if intent == "average":

        requested_column = request.get(
            "column"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        average_value = (
            df[actual_column]
            .mean()
        )

        return {
            "success": True,
            "answer": (
                f"The average of "
                f"'{actual_column}' "
                f"is {average_value:,.2f}."
            ),
            "result_type": "number",
            "value": average_value,
            "column": actual_column
        }


    # =========================================================
    # 10. MAXIMUM
    # =========================================================

    if intent == "maximum":

        requested_column = request.get(
            "column"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        maximum_value = (
            df[actual_column]
            .max()
        )

        return {
            "success": True,
            "answer": (
                f"The maximum value of "
                f"'{actual_column}' "
                f"is {maximum_value:,.2f}."
            ),
            "result_type": "number",
            "value": maximum_value,
            "column": actual_column
        }


    # =========================================================
    # 11. MINIMUM
    # =========================================================

    if intent == "minimum":

        requested_column = request.get(
            "column"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        minimum_value = (
            df[actual_column]
            .min()
        )

        return {
            "success": True,
            "answer": (
                f"The minimum value of "
                f"'{actual_column}' "
                f"is {minimum_value:,.2f}."
            ),
            "result_type": "number",
            "value": minimum_value,
            "column": actual_column
        }


    # =========================================================
    # DISTINCT VALUES
    # =========================================================

    if intent == "unique_values":

        requested_column = request.get("column")
        actual_column = find_column(df, requested_column)

        if actual_column is None:
            return {
                "success": False,
                "answer": (
                    f"The column '{requested_column}' was not found "
                    "in the dataset."
                ),
                "result_type": "error"
            }

        unique_counts = (
            df[actual_column]
            .value_counts(dropna=False)
            .rename_axis("Category")
            .reset_index(name="Value")
        )

        return {
            "success": True,
            "answer": (
                f"Here are the distinct values in '{actual_column}', "
                "with the number of records for each value."
            ),
            "result_type": "dictionary",
            "value": unique_counts,
            "column": actual_column
        }


    # =========================================================
    # 12. GROUP COUNT
    # =========================================================

    if intent == "group_count":

        requested_group = request.get(
            "group_by"
        )

        actual_group = find_column(
            df,
            requested_group
        )

        if actual_group is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_group}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        grouped = (
            df[actual_group]
            .value_counts(
                dropna=False
            )
            .reset_index()
        )

        grouped.columns = [
            "Category",
            "Value"
        ]

        return {
            "success": True,
            "answer": (
                f"Here is the count by "
                f"'{actual_group}'."
            ),
            "result_type": "dictionary",
            "value": grouped
        }


    # =========================================================
    # 13. GROUP AVERAGE
    # =========================================================

    if intent == "group_average":

        requested_column = request.get(
            "column"
        )

        requested_group = request.get(
            "group_by"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        actual_group = find_column(
            df,
            requested_group
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if actual_group is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_group}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        grouped = (
            df.groupby(
                actual_group
            )[actual_column]
            .mean()
            .reset_index()
        )

        grouped.columns = [
            "Category",
            "Value"
        ]

        return {
            "success": True,
            "answer": (
                f"Here is the average of "
                f"'{actual_column}' by "
                f"'{actual_group}'."
            ),
            "result_type": "dictionary",
            "value": grouped
        }


    # =========================================================
    # 14. GROUP SUM
    # =========================================================
    #
    # NEW IN STEP 12K.4
    #
    # Example:
    #
    # Show total purchase amount by category
    #
    # intent:
    # group_sum
    #
    # column:
    # Purchase Amount (USD)
    #
    # group_by:
    # Category
    #
    # Pandas performs the actual calculation.
    # =========================================================

    if intent == "group_sum":

        requested_column = request.get(
            "column"
        )

        requested_group = request.get(
            "group_by"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        actual_group = find_column(
            df,
            requested_group
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if actual_group is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_group}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        grouped = (
            df.groupby(
                actual_group
            )[actual_column]
            .sum()
            .reset_index()
        )

        grouped.columns = [
            "Category",
            "Value"
        ]

        return {
            "success": True,
            "answer": (
                f"Here is the total of "
                f"'{actual_column}' by "
                f"'{actual_group}'."
            ),
            "result_type": "dictionary",
            "value": grouped
        }


    # =========================================================
    # 15. GROUP ATTRITION
    # =========================================================

    if intent == "group_attrition":

        requested_group = request.get(
            "group_by"
        )

        actual_group = find_column(
            df,
            requested_group
        )

        attrition_column = find_column(
            df,
            "Attrition"
        )

        if actual_group is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_group}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if attrition_column is None:

            return {
                "success": False,
                "answer": (
                    "The dataset does not contain "
                    "an Attrition column."
                ),
                "result_type": "error"
            }

        yes_value = find_value(
            df[attrition_column],
            "Yes"
        )

        if yes_value is None:

            return {
                "success": False,
                "answer": (
                    "The dataset does not contain "
                    "an Attrition value of Yes."
                ),
                "result_type": "error"
            }

        grouped = (
            df.groupby(
                actual_group
            )
            .apply(
                lambda group:
                (
                    group[attrition_column]
                    == yes_value
                ).sum()
                / len(group)
                * 100,
                include_groups=False
            )
            .reset_index()
        )

        grouped.columns = [
            "Category",
            "Value"
        ]

        grouped = grouped.sort_values(
            "Value",
            ascending=False
        )

        highest_group = grouped.iloc[0]

        return {
            "success": True,
            "answer": (
                f"'{highest_group['Category']}' "
                f"has the highest attrition rate "
                f"at {highest_group['Value']:.2f}%."
            ),
            "result_type": "dictionary",
            "value": grouped
        }


    # =========================================================
    # 16. DISTRIBUTION
    # =========================================================

    if intent == "distribution":

        requested_column = request.get(
            "column"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        return {
            "success": True,
            "answer": (
                f"Here is the distribution of "
                f"'{actual_column}'."
            ),
            "result_type": "distribution",
            "column": actual_column,
            "value": df[
                [actual_column]
            ].copy()
        }


    # =========================================================
    # 17. RELATIONSHIP
    # =========================================================

    if intent == "relationship":

        requested_x = request.get(
            "x_column"
        )

        requested_y = request.get(
            "y_column"
        )

        x_column = find_column(
            df,
            requested_x
        )

        y_column = find_column(
            df,
            requested_y
        )

        if x_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_x}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if y_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_y}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[x_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{x_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[y_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{y_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        relationship_df = (
            df[
                [
                    x_column,
                    y_column
                ]
            ]
            .dropna()
        )

        return {
            "success": True,
            "answer": (
                f"Here is the relationship between "
                f"'{x_column}' and '{y_column}'."
            ),
            "result_type": "relationship",
            "x_column": x_column,
            "y_column": y_column,
            "value": relationship_df
        }


    # =========================================================
    # 18. UNKNOWN
    # =========================================================

    if intent == "unknown":

        return {
            "success": False,
            "answer": (
                "I could not understand the "
                "requested analysis."
            ),
            "result_type": "error"
        }


    # =========================================================
    # 19. ERROR
    # =========================================================

    if intent == "error":

        return {
            "success": False,
            "answer": (
                "There was an error while "
                "understanding the question."
            ),
            "result_type": "error"
        }


    # =========================================================
    # UNSUPPORTED INTENT
    # =========================================================

    return {
        "success": False,
        "answer": (
            "This type of analysis is "
            "not supported yet."
        ),
        "result_type": "error"
    }
