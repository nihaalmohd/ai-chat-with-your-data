"""Grounded conversational follow-ups for prior data-analysis results.

This module deliberately explains only the last successful computed result.
It does not infer causation from descriptive statistics or invent new metrics.
"""
import math

import pandas as pd


FOLLOWUP_PHRASES = (
    "what does this mean",
    "what does that mean",
    "explain this",
    "explain that",
    "explain the result",
    "interpret this",
    "interpret that",
    "summarize this",
    "summarise this",
    "why is that",
    "why did that happen",
    "tell me more about this",
    "can you explain",
    "what can we conclude",
)


def is_followup_question(question):
    """Return True for short context-dependent explanation questions."""
    text = " ".join(str(question or "").lower().strip().split())
    return any(phrase in text for phrase in FOLLOWUP_PHRASES)


def _format_value(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return str(value)
        return f"{value:,.2f}" if not float(value).is_integer() else f"{value:,.0f}"
    return str(value)


def explain_previous_result(question, previous_question, result, dataframe):
    """Explain a prior successful result using its actual computed values."""
    if not isinstance(result, dict) or not result.get("success"):
        return (
            "I don't have a successful previous result to explain yet. "
            "Ask a data question first, such as 'How many records are there?'"
        )

    answer = str(result.get("answer", "The previous analysis completed."))
    result_type = result.get("result_type")
    value = result.get("value")
    source_question = str(previous_question or "your previous question")

    if result_type in {"number", "percentage"}:
        value_text = _format_value(value)
        if result_type == "percentage":
            explanation = (
                f"The previous question was: **{source_question}**. "
                f"The calculated result was **{value_text}%**. "
                "This is a descriptive percentage for the uploaded data, "
                "not evidence by itself of what caused the outcome."
            )
        else:
            explanation = (
                f"The previous question was: **{source_question}**. "
                f"The calculated result was **{value_text}**. "
                "Interpret it in the units and context of the column or count "
                "used in that question."
            )
        return explanation

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return (
                f"The previous question was: **{source_question}**. "
                "It returned an empty table, so there are no rows to interpret."
            )
        preview = value.head(3)
        rows = []
        for _, row in preview.iterrows():
            rows.append(", ".join(f"{column}: {_format_value(row[column])}" for column in preview.columns))
        summary = "; ".join(rows)
        remaining = len(value) - len(preview)
        suffix = f" There are {remaining:,} more row(s) not shown." if remaining > 0 else ""
        return (
            f"The previous question was: **{source_question}**. "
            f"The result is a table with {len(value):,} row(s). "
            f"The first result row(s) are: {summary}.{suffix} "
            "These are computed summaries, not proof of cause and effect."
        )

    if result_type == "dictionary" and isinstance(value, dict):
        if not value:
            return f"The previous question was: **{source_question}**. The result contains no groups."
        return (
            f"The previous question was: **{source_question}**. "
            f"The result contains {len(value):,} grouped value(s). "
            "Compare groups using the displayed values; check each group's size "
            "and missing data before drawing conclusions."
        )

    if result_type in {"distribution", "relationship"}:
        return (
            f"The previous question was: **{source_question}**. {answer} "
            "A distribution describes how values are spread. A relationship "
            "shows an association, which does not establish causation."
        )

    return (
        f"The previous question was: **{source_question}**. {answer} "
        f"The uploaded dataset currently has {len(dataframe):,} row(s) and "
        f"{len(dataframe.columns):,} column(s)."
    )
