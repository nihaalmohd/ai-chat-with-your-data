import json
import os
import re

import pandas as pd
import ollama

from dataset_understanding import (
    understand_dataset
)


def chat_with_model(messages, temperature=0):
    """Call hosted Groq inference when configured, otherwise local Ollama.

    Streamlit Community Cloud cannot access Ollama running on a developer's
    computer, so deployments can set GROQ_API_KEY. Local development keeps
    using Ollama without requiring a hosted API key. Both paths return the
    same small response shape consumed by the interpreter below.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if api_key:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return {
            "message": {
                "content": completion.choices[0].message.content
            }
        }

    return ollama.chat(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        messages=messages,
        format="json",
        options={"temperature": temperature},
    )


# ==========================================================
# DATASET PROFILE
# ==========================================================

def get_dataset_profile(df):

    profile = {

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "column_names":
            df.columns.tolist(),

        "column_types": {

            column:
                str(df[column].dtype)

            for column in df.columns

        },

        "sample_values": {}

    }


    for column in df.columns:

        # Avoid materializing every distinct value in high-cardinality
        # columns (IDs, timestamps, transaction keys, etc.). The prompt
        # only needs a small representative sample.
        values = (
            df[column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        profile["sample_values"][str(column)] = values


    return profile


# ==========================================================
# CLEAN NULL VALUES
# ==========================================================

def clean_null_values(result):

    if not isinstance(result, dict):

        return result


    for key in result:

        if result[key] == "null":

            result[key] = None


    return result


# ==========================================================
# FIND COLUMN
# ==========================================================

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


# ==========================================================
# FIND VALUE
# ==========================================================

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


# ==========================================================
# FIND COLUMN BY VALUE
# ==========================================================

def find_column_by_value(
    df,
    requested_value
):

    if requested_value is None:

        return None, None


    requested_value = str(
        requested_value
    ).strip()


    for column in df.columns:

        actual_value = find_value(
            df[column],
            requested_value
        )


        if actual_value is not None:

            return (
                column,
                actual_value
            )


    return None, None


# ==========================================================
# SEMANTIC VALIDATION
# ==========================================================

def validate_semantic_result(
    result,
    understanding
):
    """
    Validate the AI-generated intent using the
    deterministic dataset-understanding layer.

    Ollama interprets the natural-language question.

    This function verifies whether the selected columns
    make analytical sense according to their semantic roles.
    """

    if not isinstance(result, dict):

        return result


    if not isinstance(
        understanding,
        dict
    ):

        return result


    column_roles = understanding.get(
        "column_roles",
        {}
    )


    identifier_columns = understanding.get(
        "identifier_columns",
        []
    )


    constant_columns = understanding.get(
        "constant_columns",
        []
    )


    usable_numeric_columns = understanding.get(
        "usable_numeric_columns",
        []
    )


    usable_group_columns = understanding.get(
        "usable_group_columns",
        []
    )


    # ======================================================
    # GROUP BY VALIDATION
    # ======================================================

    group_by = result.get(
        "group_by"
    )


    if group_by is not None:

        role = column_roles.get(
            group_by
        )


        # --------------------------------------------------
        # IDENTIFIER AS GROUPING COLUMN
        # --------------------------------------------------

        if group_by in identifier_columns:

            result["semantic_warning"] = (

                f"'{group_by}' is identified as an "
                "identifier column and is not recommended "
                "as an analytical grouping column."

            )


            # ------------------------------------------------
            # If this is a group-count question, remove the
            # invalid grouping instruction rather than
            # allowing the visualization layer to chart IDs.
            # ------------------------------------------------

            if result.get("intent") in [
                "group_count",
                "group_average",
                "group_sum",
                "group_attrition"
            ]:

                result["intent"] = "unknown"

                result["group_by"] = None

                result["column"] = None

                result["x_column"] = None

                result["y_column"] = None


            return result


        # --------------------------------------------------
        # CONSTANT COLUMN AS GROUPING COLUMN
        # --------------------------------------------------

        if group_by in constant_columns:

            result["semantic_warning"] = (

                f"'{group_by}' is a constant column and "
                "does not provide useful analytical variation."

            )


            if result.get("intent") in [
                "group_count",
                "group_average",
                "group_sum",
                "group_attrition"
            ]:

                result["intent"] = "unknown"

                result["group_by"] = None


            return result


        # --------------------------------------------------
        # INVALID GROUPING ROLE
        # --------------------------------------------------

        if (

            group_by not in usable_group_columns

            and role not in [
                "categorical",
                "ordinal"
            ]

        ):

            result["semantic_warning"] = (

                f"'{group_by}' is not a suitable "
                "categorical or ordinal grouping column."

            )


    # ======================================================
    # MEASURE VALIDATION
    # ======================================================

    column = result.get(
        "column"
    )


    if column is not None:

        role = column_roles.get(
            column
        )


        # --------------------------------------------------
        # IDENTIFIER USED AS MEASURE
        # --------------------------------------------------

        if column in identifier_columns:

            result["semantic_warning"] = (

                f"'{column}' is identified as an "
                "identifier column and should not normally "
                "be used as an analytical measure."

            )


            if result.get("intent") in [
                "average",
                "maximum",
                "minimum",
                "group_average"
            ]:

                result["intent"] = "unknown"

                result["column"] = None

                result["group_by"] = None


            return result


        # --------------------------------------------------
        # CONSTANT COLUMN AS MEASURE
        # --------------------------------------------------

        if column in constant_columns:

            result["semantic_warning"] = (

                f"'{column}' is a constant column and "
                "does not provide useful analytical variation."

            )


            if result.get("intent") in [
                "average",
                "maximum",
                "minimum",
                "group_average"
            ]:

                result["intent"] = "unknown"

                result["column"] = None


            return result


        # --------------------------------------------------
        # NUMERIC MEASURE VALIDATION
        # --------------------------------------------------

        if result.get("intent") in [
            "average",
            "maximum",
            "minimum",
            "group_average",
            "group_sum"
        ]:

            if (

                column not in usable_numeric_columns

                and role != "ordinal"

            ):

                result["semantic_warning"] = (

                    f"'{column}' is not recognized as a "
                    "suitable numeric measure for this operation."

                )


                result["intent"] = "unknown"

                result["column"] = None

                result["group_by"] = None


    # ======================================================
    # X / Y VALIDATION FOR RELATIONSHIPS
    # ======================================================

    if result.get("intent") == "relationship":

        x_column = result.get(
            "x_column"
        )

        y_column = result.get(
            "y_column"
        )


        if x_column in identifier_columns:

            result["semantic_warning"] = (

                f"'{x_column}' is an identifier and "
                "should not normally be used as an "
                "analytical relationship variable."

            )


            result["x_column"] = None


        if y_column in identifier_columns:

            result["semantic_warning"] = (

                f"'{y_column}' is an identifier and "
                "should not normally be used as an "
                "analytical relationship variable."

            )


            result["y_column"] = None


    return result


# ==========================================================
# NORMALIZE RESULT
# ==========================================================

def normalize_result(
    result,
    question,
    df
):

    if not isinstance(
        result,
        dict
    ):

        result = {}


    question_lower = str(
        question
    ).strip().lower()


    normalized = {

        "intent":
            result.get(
                "intent",
                "unknown"
            ),

        "column":
            result.get(
                "column"
            ),

        "group_by":
            result.get(
                "group_by"
            ),

        "filter_column":
            result.get(
                "filter_column"
            ),

        "filter_value":
            result.get(
                "filter_value"
            ),

        "x_column":
            result.get(
                "x_column"
            ),

        "y_column":
            result.get(
                "y_column"
            )

    }


    result = normalized


    result = clean_null_values(
        result
    )


    # A clearly unfiltered dataset-wide count takes precedence over a
    # contradictory filter invented by the language model. Keep this check
    # here as well as before the model call so normalization remains safe
    # when invoked independently by tests or future callers.
    if detect_plain_count_question(question, df):
        return {
            "intent": "count",
            "column": None,
            "group_by": None,
            "filter_column": None,
            "filter_value": None,
            "x_column": None,
            "y_column": None,
        }


    # ======================================================
    # COUNT RESULT NORMALIZATION
    # ======================================================
    #
    # A normal count question only needs the "count" intent.
    #
    # Ollama may sometimes return an irrelevant identifier
    # such as "Customer ID" or "EmployeeNumber" in y_column.
    #
    # Count is performed deterministically by Pandas, so no
    # analytical column is required for a normal record/entity
    # count.
    #
    # This rule is dataset-agnostic and applies to any dataset.
    # ======================================================

    if result.get("intent") == "count":

        result["column"] = None

        result["group_by"] = None

        result["filter_column"] = None

        result["filter_value"] = None

        result["x_column"] = None

        result["y_column"] = None


    # ======================================================
    # NORMALIZE COLUMN
    # ======================================================

    if result.get(
        "column"
    ) is not None:

        actual_column = find_column(
            df,
            result["column"]
        )


        if actual_column is not None:

            result["column"] = (
                actual_column
            )

        else:

            result["column"] = None


    # ======================================================
    # NORMALIZE GROUP BY
    # ======================================================

    if result.get(
        "group_by"
    ) is not None:

        actual_group = find_column(
            df,
            result["group_by"]
        )


        if actual_group is not None:

            result["group_by"] = (
                actual_group
            )

        else:

            result["group_by"] = None


    # ======================================================
    # NORMALIZE FILTER COLUMN
    # ======================================================

    if result.get(
        "filter_column"
    ) is not None:

        actual_filter_column = (
            find_column(
                df,
                result["filter_column"]
            )
        )


        if actual_filter_column is not None:

            result["filter_column"] = (
                actual_filter_column
            )

        else:

            result["filter_column"] = None


    # ======================================================
    # NORMALIZE FILTER VALUE
    # ======================================================

    if (

        result.get(
            "filter_column"
        ) is not None

        and result.get(
            "filter_value"
        ) is not None

    ):

        actual_value = find_value(

            df[
                result["filter_column"]
            ],

            result["filter_value"]

        )


        if actual_value is not None:

            result["filter_value"] = (
                actual_value
            )


    # ======================================================
    # FILTERED COUNT DETECTION
    # ======================================================

    filtered_count_phrases = [

        "how many employees have",

        "how many employees are from",

        "how many employees belong to",

        "how many employees work in",

        "how many records have",

        "how many records are from",

        "how many records belong to",

        "how many records match",

        "how many people are from",

        "how many people have",

        "how many customers are from",

        "how many customers belong to",

        "how many customers work in",

        "how many products are from",

        "how many products belong to",

        "how many orders are from",

        "how many orders belong to"

    ]


    is_filtered_count_question = any(

        phrase in question_lower

        for phrase in filtered_count_phrases

    )


    # ======================================================
    # GENERIC FILTER LANGUAGE
    # ======================================================

    filter_language_patterns = [

        r"\bare\s+from\b",

        r"\bbelong\s+to\b",

        r"\bbelong\s+in\b",

        r"\bwork\s+in\b",

        r"\bworks\s+in\b",

        r"\bequal\s+to\b",

        r"\bequals\b",

        r"\bwhere\b"

    ]


    if any(

        re.search(
            pattern,
            question_lower
        )

        for pattern
        in filter_language_patterns

    ):

        if (

            "how many"
            in question_lower

            or "count"
            in question_lower

        ):

            is_filtered_count_question = True


    # ======================================================
    # GENERIC "ARE" PATTERN
    # ======================================================

    generic_are_match = re.search(

        r"^how\s+many\s+.+?\s+are\s+"
        r"(.+?)(?:\?|$)",

        question,

        re.IGNORECASE

    )


    if generic_are_match:

        value_text = (

            generic_are_match
            .group(1)
            .strip()

        )


        value_text = value_text.rstrip(
            "?. "
        )


        if value_text.lower() not in [

            "there",

            "available",

            "present"

        ]:

            actual_column, actual_value = (
                find_column_by_value(
                    df,
                    value_text
                )
            )


            if actual_column is not None:

                is_filtered_count_question = True

                result["intent"] = (
                    "filtered_count"
                )

                result["column"] = None

                result["group_by"] = None

                result["filter_column"] = (
                    actual_column
                )

                result["filter_value"] = (
                    actual_value
                )

                result["x_column"] = None

                result["y_column"] = None


    # ======================================================
    # FILTERED COUNT EXTRACTION
    # ======================================================

    if is_filtered_count_question:

        result["intent"] = (
            "filtered_count"
        )

        result["column"] = None

        result["group_by"] = None

        result["x_column"] = None

        result["y_column"] = None


        # --------------------------------------------------
        # EQUAL TO / EQUALS
        # --------------------------------------------------

        equal_match = re.search(

            r"(?:have|has)\s+(.+?)\s+"
            r"(?:equal to|equals)\s+"
            r"(.+?)(?:\?|$)",

            question,

            re.IGNORECASE

        )


        if equal_match:

            column_text = (

                equal_match
                .group(1)
                .strip()

            )


            value_text = (

                equal_match
                .group(2)
                .strip()

            )


            value_text = value_text.rstrip(
                "?. "
            )


            actual_column = find_column(
                df,
                column_text
            )


            if actual_column is not None:

                result["filter_column"] = (
                    actual_column
                )


                actual_value = find_value(

                    df[actual_column],

                    value_text

                )


                if actual_value is not None:

                    result["filter_value"] = (
                        actual_value
                    )

                else:

                    result["filter_value"] = (
                        value_text
                    )


        # --------------------------------------------------
        # FROM
        # --------------------------------------------------

        if result[
            "filter_column"
        ] is None:

            from_match = re.search(

                r"\bare\s+from\s+"
                r"(.+?)(?:\?|$)",

                question,

                re.IGNORECASE

            )


            if from_match:

                value_text = (

                    from_match
                    .group(1)
                    .strip()

                )


                value_text = value_text.rstrip(
                    "?. "
                )


                (
                    actual_column,
                    actual_value
                ) = find_column_by_value(

                    df,
                    value_text

                )


                if actual_column is not None:

                    result[
                        "filter_column"
                    ] = actual_column

                    result[
                        "filter_value"
                    ] = actual_value


        # --------------------------------------------------
        # WORK IN
        # --------------------------------------------------

        if result[
            "filter_column"
        ] is None:

            work_match = re.search(

                r"\bwork(?:s)?\s+in\s+"
                r"(.+?)(?:\?|$)",

                question,

                re.IGNORECASE

            )


            if work_match:

                value_text = (

                    work_match
                    .group(1)
                    .strip()

                )


                value_text = value_text.rstrip(
                    "?. "
                )


                (
                    actual_column,
                    actual_value
                ) = find_column_by_value(

                    df,
                    value_text

                )


                if actual_column is not None:

                    result[
                        "filter_column"
                    ] = actual_column

                    result[
                        "filter_value"
                    ] = actual_value


        # --------------------------------------------------
        # BELONG TO / IN
        # --------------------------------------------------

        if result[
            "filter_column"
        ] is None:

            belong_match = re.search(

                r"\bbelong\s+(?:to|in)\s+"
                r"(.+?)(?:\?|$)",

                question,

                re.IGNORECASE

            )


            if belong_match:

                value_text = (

                    belong_match
                    .group(1)
                    .strip()

                )


                value_text = value_text.rstrip(
                    "?. "
                )


                (
                    actual_column,
                    actual_value
                ) = find_column_by_value(

                    df,
                    value_text

                )


                if actual_column is not None:

                    result[
                        "filter_column"
                    ] = actual_column

                    result[
                        "filter_value"
                    ] = actual_value


        # --------------------------------------------------
        # WHERE
        # --------------------------------------------------

        if result[
            "filter_column"
        ] is None:

            where_match = re.search(

                r"\bwhere\s+(.+?)\s*"
                r"(?:=|is|equals|equal to)\s+"
                r"(.+?)(?:\?|$)",

                question,

                re.IGNORECASE

            )


            if where_match:

                column_text = (

                    where_match
                    .group(1)
                    .strip()

                )


                value_text = (

                    where_match
                    .group(2)
                    .strip()

                )


                value_text = value_text.rstrip(
                    "?. "
                )


                actual_column = find_column(
                    df,
                    column_text
                )


                if actual_column is not None:

                    actual_value = find_value(

                        df[actual_column],

                        value_text

                    )


                    if actual_value is not None:

                        result[
                            "filter_column"
                        ] = actual_column

                        result[
                            "filter_value"
                        ] = actual_value


    # ======================================================
    # GROUP AVERAGE
    # ======================================================

    if (

        "average"
        in question_lower

        and " by "
        in question_lower

    ):

        result["intent"] = (
            "group_average"
        )

        result[
            "filter_column"
        ] = None

        result[
            "filter_value"
        ] = None


        by_position = (
            question_lower.rfind(
                " by "
            )
        )


        if by_position != -1:

            group_text = question[
                by_position + 4:
            ].strip()


            group_text = group_text.rstrip(
                "?."
            )


            for dataset_column in df.columns:

                if (

                    str(dataset_column)
                    .strip()
                    .lower()

                    == group_text.lower()

                ):

                    result["group_by"] = (
                        dataset_column
                    )

                    break


    # ======================================================
    # PROPORTION / PERCENTAGE / SHARE
    # ======================================================

    proportion_keywords = [
        "proportion",
        "percentage",
        "percent",
        "share",
        "composition"
    ]


    is_proportion_question = any(
        keyword in question_lower
        for keyword in proportion_keywords
    )


    if (
        is_proportion_question
        and " by " in question_lower
    ):

        result["intent"] = "group_count"

        result["column"] = None

        result["filter_column"] = None

        result["filter_value"] = None

        result["x_column"] = None

        result["y_column"] = None


        by_position = question_lower.rfind(" by ")


        if by_position != -1:

            group_text = question[
                by_position + 4:
            ].strip()


            group_text = group_text.rstrip("?. ")


            actual_group = find_column(
                df,
                group_text
            )


            if actual_group is not None:

                result["group_by"] = actual_group

            else:

                result["group_by"] = None


    # ======================================================
    # GROUP COUNT
    # ======================================================

    if (

        (

            "count by"
            in question_lower

            or "count of"
            in question_lower

            or "employee count by"
            in question_lower

            or "number of employees by"
            in question_lower

            or "how many employees by"
            in question_lower

            or "distribution by"
            in question_lower

        )

        and " by "
        in question_lower

    ):

        result["intent"] = (
            "group_count"
        )


        by_position = (
            question_lower.rfind(
                " by "
            )
        )


        if by_position != -1:

            group_text = question[
                by_position + 4:
            ].strip()


            group_text = group_text.rstrip(
                "?."
            )


            for dataset_column in df.columns:

                if (

                    str(dataset_column)
                    .strip()
                    .lower()

                    == group_text.lower()

                ):

                    result["group_by"] = (
                        dataset_column
                    )

                    break


    # ======================================================
    # GROUP ATTRITION / RATE-LIKE ANALYSIS NORMALIZATION
    # ======================================================
    # Do not hard-code HR grouping columns such as Department
    # or JobRole. If the user explicitly mentions attrition,
    # resolve the target/grouping columns from the current
    # dataset instead.
    # ======================================================

    if "attrition" in question_lower:

        if result.get("intent") == "group_attrition" or (
            "highest attrition" in question_lower
            or "highest rate of attrition" in question_lower
        ):

            result["intent"] = "group_attrition"

            # The target is determined from the dataset, not
            # from an assumed HR schema.
            attrition_column = find_column(
                df,
                "Attrition"
            )

            if attrition_column is not None:
                result["column"] = attrition_column

            # Recover the grouping dimension from wording such as
            # "which department has the highest attrition".
            group_match = re.search(
                r"\bwhich\s+(.+?)\s+has\s+"
                r"(?:the\s+)?highest\s+attrition\b",
                question_lower,
                re.IGNORECASE
            )

            if group_match:
                group_text = group_match.group(1).strip()

                actual_group = find_column(
                    df,
                    group_text
                )

                # Also support column names whose spacing differs
                # from the user's wording, e.g. "job role" vs
                # "JobRole".
                if actual_group is None:
                    requested_compact = re.sub(
                        r"[^a-z0-9]",
                        "",
                        group_text.lower()
                    )

                    for dataset_column in df.columns:
                        column_compact = re.sub(
                            r"[^a-z0-9]",
                            "",
                            str(dataset_column).lower()
                        )

                        if column_compact == requested_compact:
                            actual_group = dataset_column
                            break

                if actual_group is not None:
                    result["group_by"] = actual_group

            # A group-attrition request is not a filtered-count
            # request. Clear contradictory fields returned by the
            # language model.
            result["filter_column"] = None
            result["filter_value"] = None
            result["x_column"] = None
            result["y_column"] = None

    # ======================================================
    # DISTRIBUTION
    # ======================================================

    if "distribution" in question_lower:

        result["intent"] = (
            "distribution"
        )


    # ======================================================
    # RELATIONSHIP
    # ======================================================

    if (

        "relationship"
        in question_lower

        or "correlation"
        in question_lower

    ):

        result["intent"] = (
            "relationship"
        )


    # ======================================================
    # DETERMINISTIC NORMAL COUNT FALLBACK
    # ======================================================
    # Ollama can occasionally return "unknown" for simple
    # dataset-wide count questions. Recognize only clear,
    # unfiltered total-count wording here. Filtered and grouped
    # questions are deliberately excluded.
    # ======================================================

    count_question = re.sub(
        r"\s+",
        " ",
        question_lower
    ).strip().rstrip("?.! ")

    plain_count_patterns = [
        r"^how many .+? are there$",
        r"^how many (?:rows?|records?)$",
        r"^how many .+? do we have$",
        r"^(?:what is|what's) the total number of .+$",
        r"^total number of .+$",
        r"^count of (?:the )?(?:rows?|records?)$",
        r"^number of (?:rows?|records?)$",
    ]

    has_plain_count_wording = any(
        re.fullmatch(pattern, count_question, re.IGNORECASE)
        for pattern in plain_count_patterns
    )

    # These terms indicate that the user is asking for a subset,
    # breakdown, or special data-quality count—not the full row count.
    count_exclusions = [
        " by ",
        " where ",
        " with ",
        " without ",
        " missing",
        " null",
        " duplicate",
        " duplicates",
        " equal to ",
        " equals ",
        " greater than ",
        " less than ",
        " more than ",
        " fewer than ",
        " from ",
        " belong to ",
        " work in ",
        " are male",
        " are female",
        " attrition",
        " left",
        " resigned",
    ]

    has_exclusion = any(
        term in f" {count_question} "
        for term in count_exclusions
    )

    if (
        has_plain_count_wording
        and not has_exclusion
        and result.get("intent") in ["unknown", "count"]
        and result.get("filter_column") is None
        and result.get("filter_value") is None
        and result.get("group_by") is None
    ):
        result["intent"] = "count"
        result["column"] = None
        result["group_by"] = None
        result["filter_column"] = None
        result["filter_value"] = None
        result["x_column"] = None
        result["y_column"] = None

    # ======================================================
    # FINAL ANALYTICAL COMPLETENESS CHECK
    # ======================================================

    intent = result.get("intent")

    missing_required_fields = False


    if intent in [
        "average",
        "maximum",
        "minimum",
        "distribution",
        "unique_values"
    ]:

        missing_required_fields = (
            result.get("column") is None
        )


    elif intent in [
        "group_count",
        "group_average",
        "group_attrition"
    ]:

        missing_required_fields = (
            result.get("group_by") is None
        )


        if intent in [
            "group_average",
            "group_attrition"
        ]:

            missing_required_fields = (

                missing_required_fields

                or result.get(
                    "column"
                ) is None

            )


    elif intent == "relationship":

        missing_required_fields = (

            result.get(
                "x_column"
            ) is None

            or result.get(
                "y_column"
            ) is None

        )


    elif intent == "filtered_count":

        missing_required_fields = (

            result.get(
                "filter_column"
            ) is None

            or result.get(
                "filter_value"
            ) is None

        )


    if missing_required_fields:

        result["intent"] = "unknown"

        result["column"] = None

        result["group_by"] = None

        result["x_column"] = None

        result["y_column"] = None


        if "semantic_warning" not in result:

            result["semantic_warning"] = (

                "The requested analysis is incomplete. Please "
                "specify all required dataset columns or filters."

            )


    return result


# ==========================================================
# INCOMPLETE QUESTION DETECTION
# ==========================================================

def is_incomplete_question(question):
    """Return True when the question asks for an operation
    but does not provide enough information to choose the
    analytical column or grouping dimension safely.
    """

    text = str(
        question or ""
    ).strip().lower()


    if not text:

        return True


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    text = text.rstrip(
        "?.! "
    )


    incomplete_patterns = [

        r"^(?:show|give|find|get|calculate|display)\s+"
        r"(?:the\s+)?"
        r"(?:average|mean|sum|total|maximum|max|minimum|min|median|mode)$",

        r"^(?:what is|what's|what are)\s+"
        r"(?:the\s+)?"
        r"(?:average|mean|sum|total|maximum|max|minimum|min|median|mode)$",

        r"^(?:show|give|find|get|calculate|display)\s+"
        r"(?:a\s+)?"
        r"(?:distribution|relationship|correlation)$"

    ]


    return any(

        re.fullmatch(
            pattern,
            text,
            re.IGNORECASE
        )

        for pattern in incomplete_patterns

    )


# ==========================================================
# DATASET-AWARE DETERMINISTIC QUESTION GUARDS
# ==========================================================
def detect_value_filtered_count_question(question, df):
    """Resolve count questions whose category/value is stated in either order.

    Example: "How many female customers are there?" and
    "How many customers are female?" both resolve to Gender=Female.
    The logic discovers values from the loaded dataset rather than naming
    a particular dataset or column in code.
    """
    text = re.sub(r"\s+", " ", str(question or "").strip().lower())
    if not text or not re.search(
        r"\b(?:how many|count of|number of|count)\b", text
    ):
        return None

    matches = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series.dtype):
            continue
        try:
            values = series.dropna().unique()
        except Exception:
            continue
        # A free-text / identifier column is not a useful source of
        # category values for this conservative deterministic shortcut.
        if len(values) == 0 or len(values) > min(100, max(20, int(len(df) * 0.1))):
            continue
        for value in values:
            value_text = str(value).strip()
            if len(value_text) < 3 or value_text.lower() in {"nan", "none", "null"}:
                continue
            if re.search(r"(?<!\w)" + re.escape(value_text.lower()) + r"(?!\w)", text):
                matches.append((column, value))

    # Only auto-resolve a unique column/value match. If the same word occurs
    # as a value in several columns, let the semantic interpreter decide.
    unique_matches = []
    seen = set()
    for column, value in matches:
        key = (str(column), str(value).lower())
        if key not in seen:
            seen.add(key)
            unique_matches.append((column, value))

    if len(unique_matches) != 1:
        return None

    column, value = unique_matches[0]
    return {
        "intent": "filtered_count",
        "column": None,
        "group_by": None,
        "filter_column": column,
        "filter_value": value,
        "x_column": None,
        "y_column": None,
    }


def resolve_average_by_entity_question(question, df):
    """Resolve explicit average-measure-by-each-entity questions conservatively.

    Uses column names and data types to identify the measure and entity
    identifier. It does not assume a particular dataset schema.
    """
    text = re.sub(r"\s+", " ", str(question or "").strip().lower())
    match = re.search(
        r"\b(?:average|mean)\b\s+(.+?)\s+\b(?:by|for)\s+(?:each|every)\s+([a-z][a-z _-]*?)\s*[?.!]*$",
        text,
    )
    if not match:
        return None

    measure_phrase = match.group(1)
    entity_phrase = match.group(2).strip()
    stop_words = {"show", "the", "a", "an", "of", "for", "per", "each", "every"}
    measure_tokens = {
        token.rstrip("s") for token in re.findall(r"[a-z]+", measure_phrase)
        if token not in stop_words
    }
    entity_tokens = {
        token.rstrip("s") for token in re.findall(r"[a-z]+", entity_phrase)
        if token not in stop_words
    }
    if not measure_tokens or not entity_tokens:
        return None

    numeric_columns = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col].dtype)
    ]
    if not numeric_columns:
        return None

    def tokens(value):
        return {word.rstrip("s") for word in re.findall(r"[a-z]+", str(value).lower())}

    measure_ranked = []
    money_or_amount_terms = {"amount", "price", "cost", "revenue", "sales", "spend", "value", "total"}
    count_terms = {"count", "number", "quantity", "previous", "frequency"}
    for column in numeric_columns:
        col_tokens = tokens(column)
        overlap = len(measure_tokens & col_tokens)
        if overlap == 0:
            continue
        score = overlap * 3
        if col_tokens & money_or_amount_terms:
            score += 4
        if col_tokens & count_terms:
            score -= 2
        measure_ranked.append((score, column))
    if not measure_ranked:
        return None
    measure_ranked.sort(key=lambda item: item[0], reverse=True)
    if len(measure_ranked) > 1 and measure_ranked[0][0] == measure_ranked[1][0]:
        return None
    measure_column = measure_ranked[0][1]

    entity_ranked = []
    for column in df.columns:
        col_tokens = tokens(column)
        overlap = len(entity_tokens & col_tokens)
        if overlap == 0:
            continue
        score = overlap * 3
        if col_tokens & {"id", "identifier", "key", "number"}:
            score += 3
        try:
            unique_ratio = df[column].nunique(dropna=True) / max(len(df), 1)
        except Exception:
            unique_ratio = 0
        if unique_ratio >= 0.8:
            score += 2
        entity_ranked.append((score, column))
    if not entity_ranked:
        return None
    entity_ranked.sort(key=lambda item: item[0], reverse=True)
    if len(entity_ranked) > 1 and entity_ranked[0][0] == entity_ranked[1][0]:
        return None
    group_column = entity_ranked[0][1]

    return {
        "intent": "group_average",
        "column": measure_column,
        "group_by": group_column,
        "filter_column": None,
        "filter_value": None,
        "x_column": None,
        "y_column": None,
    }


def detect_ambiguous_average_request(question, df):
    """Ask for clarification when a broad average noun matches multiple measures."""
    text = re.sub(r"\s+", " ", str(question or "").strip().lower())
    if not re.search(r"\b(?:average|mean)\b", text):
        return None
    if re.search(r"\b(?:by|for each|for every|per)\b", text):
        return None

    # Extract the metric phrase after average/mean and ignore generic wording.
    metric_match = re.search(r"\b(?:average|mean)\b\s+(?:of\s+)?(.+?)[?.!]*$", text)
    if not metric_match:
        return None
    metric_tokens = {
        token.rstrip("s") for token in re.findall(r"[a-z]+", metric_match.group(1))
        if token not in {"the", "a", "an", "of", "all", "overall"}
    }
    if not metric_tokens:
        return None

    candidates = []
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column].dtype):
            continue
        col_tokens = {word.rstrip("s") for word in re.findall(r"[a-z]+", str(column).lower())}
        if metric_tokens & col_tokens:
            candidates.append(str(column))

    if len(candidates) < 2:
        return None
    return {
        "intent": "unknown",
        "column": None,
        "group_by": None,
        "filter_column": None,
        "filter_value": None,
        "x_column": None,
        "y_column": None,
        "semantic_warning": (
            "Your question could refer to more than one numeric column: "
            + ", ".join(candidates[:5])
            + ". Please specify the column you want to average."
        ),
    }


# ==========================================================
# DETERMINISTIC PLAIN-COUNT DETECTION
# ==========================================================
def detect_plain_count_question(question, df=None):
    """Recognize unfiltered dataset-wide counts before calling the LLM.

    This deliberately uses whole-question patterns and rejects common
    filter/group/data-quality wording. It is dataset-agnostic: it does not
    depend on particular column names or values.
    """
    text = re.sub(r"\s+", " ", str(question or "").strip().lower())
    text = text.rstrip("?.! ")
    if not text:
        return False

    # If the question mentions a unique categorical value in the loaded data,
    # treat it as a possible filter rather than a dataset-wide count.
    if df is not None and detect_value_filtered_count_question(question, df) is not None:
        return False

    # These clauses usually indicate a subset, grouping, or a different
    # metric rather than the number of records in the complete dataset.
    exclusions = (
        " by ", " where ", " with ", " without ", " from ",
        " belong to ", " belong in ", " work in ", " works in ",
        " equal to ", " equals ", " greater than ", " less than ",
        " more than ", " fewer than ", " missing ", " null ",
        " duplicate ", " duplicates ", " attrition ", " left ",
        " resigned ", " are male", " are female", " are yes", " are no",
    )
    padded = f" {text} "
    if any(term in padded for term in exclusions):
        return False

    patterns = (
        r"^how many .+ are there$",
        r"^how many .+ are in (?:the )?dataset$",
        r"^how many .+ do we have$",
        r"^(?:what is|what's) the total number of .+$",
        r"^total number of .+$",
        r"^total .+$",
        r"^number of (?:rows?|records?|entries|observations|employees|people|customers|orders|products)$",
        r"^how many (?:rows?|records?|entries|observations) are there$",
        r"^how many (?:employees|people|workers|staff) are employed$",
    )
    return any(re.fullmatch(pattern, text, re.IGNORECASE) for pattern in patterns)


# ==========================================================
# UNDERSTAND QUESTION
# ==========================================================

def understand_question(
    question,
    df
):

    try:

        # ==================================================
        # REJECT INCOMPLETE QUESTIONS BEFORE LLM FALLBACK
        # ==================================================

        if is_incomplete_question(question):

            return {

                "intent":
                    "unknown",

                "column":
                    None,

                "group_by":
                    None,

                "filter_column":
                    None,

                "filter_value":
                    None,

                "x_column":
                    None,

                "y_column":
                    None,

                "semantic_warning":
                    (
                        "The question is incomplete. Please "
                        "specify the column or analysis dimension "
                        "you want to use."
                    )

            }


        # ==================================================
        # DETERMINISTIC CATEGORY FILTERS / EXPLICIT ENTITY AVERAGES
        # ==================================================
        # Resolve unambiguous questions from actual dataset values and column
        # metadata before the model can turn them into a plain count or select
        # an unrelated metric.
        filtered_count_request = detect_value_filtered_count_question(
            question, df
        )
        if filtered_count_request is not None:
            return filtered_count_request

        average_by_entity_request = resolve_average_by_entity_question(
            question, df
        )
        if average_by_entity_request is not None:
            return average_by_entity_request

        ambiguous_average_request = detect_ambiguous_average_request(
            question, df
        )
        if ambiguous_average_request is not None:
            return ambiguous_average_request

        # ==================================================
        # DETERMINISTIC DATASET-WIDE COUNT
        # ==================================================
        # Resolve unfiltered count questions without asking the LLM to
        # infer filters. This prevents hallucinated filters such as
        # Attrition=Yes from changing a total into a subset count.
        if detect_plain_count_question(question):
            return {
                "intent": "count",
                "column": None,
                "group_by": None,
                "filter_column": None,
                "filter_value": None,
                "x_column": None,
                "y_column": None,
            }

        # ==================================================
        # DATASET PROFILE
        # ==================================================

        profile = get_dataset_profile(
            df
        )


        # ==================================================
        # DATASET UNDERSTANDING
        # ==================================================

        understanding = understand_dataset(
            df
        )


        column_roles = understanding.get(
            "column_roles",
            {}
        )


        role_summary = understanding.get(
            "role_summary",
            {}
        )


        identifier_columns = understanding.get(
            "identifier_columns",
            []
        )


        numeric_columns = understanding.get(
            "numeric_columns",
            []
        )


        categorical_columns = understanding.get(
            "categorical_columns",
            []
        )


        ordinal_columns = understanding.get(
            "ordinal_columns",
            []
        )


        datetime_columns = understanding.get(
            "datetime_columns",
            []
        )


        usable_numeric_columns = understanding.get(
            "usable_numeric_columns",
            []
        )


        usable_group_columns = understanding.get(
            "usable_group_columns",
            []
        )


        # ==================================================
        # BUILD SEMANTIC PROFILE
        # ==================================================

        semantic_profile = {

            "column_roles":
                column_roles,

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

            "usable_numeric_columns":
                usable_numeric_columns,

            "usable_group_columns":
                usable_group_columns

        }


        # ==================================================
        # OLLAMA PROMPT
        # ==================================================

        prompt = f"""
You are an AI data-analysis question interpreter.

The dataset can be ANY structured dataset.

It may contain:

employees,
customers,
sales,
products,
orders,
insurance,
finance,
marketing,
healthcare,
logistics,
or any other structured data.

Do NOT assume that the dataset is an HR dataset.

Your job is ONLY to understand the user's question
and convert it into structured JSON.

Python/Pandas will perform the actual calculation.

DO NOT calculate the answer yourself.

============================================================
DATASET PROFILE
============================================================

Rows:
{profile["rows"]}

Columns:
{profile["columns"]}

Column names:
{profile["column_names"]}

Column types:
{profile["column_types"]}

Sample values:
{profile["sample_values"]}

============================================================
SEMANTIC COLUMN INFORMATION
============================================================

The system has automatically identified semantic roles
for the dataset columns.

Column roles:
{semantic_profile["column_roles"]}

Identifier columns:
{semantic_profile["identifier_columns"]}

Numeric columns:
{semantic_profile["numeric_columns"]}

Categorical columns:
{semantic_profile["categorical_columns"]}

Ordinal columns:
{semantic_profile["ordinal_columns"]}

Datetime columns:
{semantic_profile["datetime_columns"]}

Usable numeric columns:
{semantic_profile["usable_numeric_columns"]}

Usable grouping columns:
{semantic_profile["usable_group_columns"]}

============================================================
IMPORTANT SEMANTIC RULES
============================================================

1. Identifier columns are identifiers, not analytical
   grouping dimensions or measures.

2. Do NOT normally use an identifier column for:
   - group_count
   - group_average
   - group_attrition
   - average
   - maximum
   - minimum
   - relationship

3. Prefer categorical or ordinal columns for grouping.

4. Prefer numeric columns for mathematical measures.

5. Constant columns should not be used for analytical
   comparisons.

6. If the user asks for a count of records/entities,
   the count can be performed without selecting an
   identifier as the measure.

7. If the user explicitly asks for an identifier itself,
   preserve the user's requested column when it is
   necessary to answer the question, but do not treat
   the identifier as a meaningful analytical measure.

============================================================
SUPPORTED INTENTS
============================================================

Use one of:

count
unique_values
average
maximum
minimum
attrition_rate
group_count
group_average
group_sum
group_attrition
distribution
relationship
filtered_count
unknown

============================================================
FILTERED COUNT
============================================================

If the user asks:

"How many employees have Attrition equal to Yes?"

use:

{{
    "intent": "filtered_count",
    "column": null,
    "group_by": null,
    "filter_column": "Attrition",
    "filter_value": "Yes",
    "x_column": null,
    "y_column": null
}}

If the user asks:

"How many employees are from Sales?"

and Sales exists in Department,

use:

{{
    "intent": "filtered_count",
    "column": null,
    "group_by": null,
    "filter_column": "Department",
    "filter_value": "Sales",
    "x_column": null,
    "y_column": null
}}

If the user asks:

"How many customers are Male?"

and Male exists in a dataset column,

use:

{{
    "intent": "filtered_count",
    "column": null,
    "group_by": null,
    "filter_column": "Gender",
    "filter_value": "Male",
    "x_column": null,
    "y_column": null
}}

============================================================
GROUP COUNT
============================================================

For:

"Show employee count by department"

use:

{{
    "intent": "group_count",
    "column": null,
    "group_by": "Department",
    "filter_column": null,
    "filter_value": null,
    "x_column": null,
    "y_column": null
}}

For:

"Show customer count by category"

use:

{{
    "intent": "group_count",
    "column": null,
    "group_by": "Category",
    "filter_column": null,
    "filter_value": null,
    "x_column": null,
    "y_column": null
}}

Do NOT use an identifier as the group_by column for
a normal analytical count.

============================================================
GROUP AVERAGE
============================================================

For:

"Show average monthly income by department"

use:

{{
    "intent": "group_average",
    "column": "MonthlyIncome",
    "group_by": "Department",
    "filter_column": null,
    "filter_value": null,
    "x_column": null,
    "y_column": null
}}

For:

"Show average purchase amount by category"

use:

{{
    "intent": "group_average",
    "column": "Purchase Amount (USD)",
    "group_by": "Category",
    "filter_column": null,
    "filter_value": null,
    "x_column": null,
    "y_column": null
}}

============================================================
GROUP SUM
============================================================

For:

"Show total purchase amount by category"

use:

{{
    "intent": "group_sum",
    "column": "Purchase Amount (USD)",
    "group_by": "Category",
    "filter_column": null,
    "filter_value": null,
    "x_column": null,
    "y_column": null
}}

For:

"Show total sales by region"

select the numeric sales/revenue/amount column
and the appropriate categorical grouping column.

The selected measure must be numeric.

Python/Pandas will calculate the actual totals.

Do not calculate the totals yourself.

============================================================
RELATIONSHIP
============================================================

For relationship or correlation questions identify:

x_column
y_column

Prefer numeric columns.

Do not normally select identifier columns.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Do not add explanations outside the JSON.

============================================================
USER QUESTION
============================================================

{question}
"""


        # ==================================================
        # CALL OLLAMA
        # ==================================================

        response = chat_with_model(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )


        content = response[
            "message"
        ][
            "content"
        ]


        # ==================================================
        # PARSE JSON
        # ==================================================

        parsed_result = json.loads(
            content
        )


        # ==================================================
        # NORMALIZE RESULT
        # ==================================================

        parsed_result = normalize_result(

            parsed_result,

            question,

            df

        )


        # ==================================================
        # DETERMINISTIC SEMANTIC VALIDATION
        # ==================================================

        parsed_result = validate_semantic_result(

            parsed_result,

            understanding

        )


        # ==================================================
        # CONTROLLED INTERPRETATION RETRY
        # ==================================================
        # A local model can occasionally return valid JSON that
        # is semantically incomplete. Retry once with an explicit
        # correction instruction rather than allowing a transient
        # interpretation failure to reach the analyzer.
        # ==================================================

        if parsed_result.get("intent") in [
            "unknown",
            "error"
        ]:

            retry_prompt = prompt + """

============================================================
CORRECTION PASS
============================================================

The previous interpretation was incomplete or inconsistent.
Re-read the user question and the dataset semantic information.
Return the single best structured interpretation.

Rules:
- Use an exact column name from the provided dataset.
- Never invent a column.
- Do not use an identifier as an analytical grouping or measure.
- For a grouped rate/attrition question, provide both the
  grouping column and the target column when the target exists.
- For a normal count, no analytical column is required.
- Do not add filters unless the user explicitly asked for a filter.
- Return ONLY the required JSON object.
"""

            retry_response = chat_with_model(
                messages=[
                    {
                        "role": "user",
                        "content": retry_prompt,
                    }
                ],
                temperature=0,
            )

            retry_content = retry_response[
                "message"
            ][
                "content"
            ]

            retry_result = json.loads(
                retry_content
            )

            retry_result = normalize_result(
                retry_result,
                question,
                df
            )

            retry_result = validate_semantic_result(
                retry_result,
                understanding
            )

            if retry_result.get("intent") not in [
                "unknown",
                "error"
            ]:
                return retry_result

        return parsed_result


    except Exception as e:

        return {

            "intent":
                "error",

            "error":
                str(e)

        }
