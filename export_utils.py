"""Safe, testable export helpers shared by the Streamlit UI and regression tests."""
from io import BytesIO

import pandas as pd


def dataframe_to_excel_bytes(dataframe, sheet_name="Results"):
    """Serialize a DataFrame to a valid in-memory .xlsx workbook."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    output = BytesIO()
    safe_sheet_name = str(sheet_name)[:31] or "Results"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=safe_sheet_name)
    return output.getvalue()


def exportable_result_dataframe(result):
    """Convert a successful scalar, table, or grouped result to a DataFrame."""
    if not isinstance(result, dict) or not result.get("success"):
        return None

    value = result.get("value")
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return pd.DataFrame(list(value.items()), columns=["Category", "Value"])
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(value)
    if value is not None and result.get("result_type") in {"number", "percentage"}:
        return pd.DataFrame({"Result": [value]})
    return None
