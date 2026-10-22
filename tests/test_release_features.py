"""Regression tests for exports, grounded follow-ups, and release-readiness helpers."""
import io
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "ollama" not in sys.modules:
    sys.modules["ollama"] = types.SimpleNamespace(
        chat=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("These tests must not call an LLM")
        )
    )

from export_utils import dataframe_to_excel_bytes, exportable_result_dataframe
from followup_handler import is_followup_question, explain_previous_result


class ExportHelperTests(unittest.TestCase):
    def test_csv_export_round_trips_values(self):
        source = pd.DataFrame({"Region": ["East", "West"], "Sales": [12.5, 25.0]})
        csv_bytes = source.to_csv(index=False).encode("utf-8")
        restored = pd.read_csv(io.BytesIO(csv_bytes))
        pd.testing.assert_frame_equal(source, restored)

    def test_excel_export_is_readable_and_sheet_name_is_safe(self):
        source = pd.DataFrame({"Revenue": [1, 2], "Label": ["A", "B"]})
        workbook = dataframe_to_excel_bytes(source, "A sheet name that is much longer than thirty-one characters")
        restored = pd.read_excel(io.BytesIO(workbook), sheet_name=0)
        pd.testing.assert_frame_equal(source, restored)

    def test_excel_export_empty_sheet_name_uses_default(self):
        workbook = dataframe_to_excel_bytes(pd.DataFrame({"x": [1]}), "")
        self.assertGreater(len(workbook), 0)
        self.assertEqual(pd.read_excel(io.BytesIO(workbook)).iloc[0, 0], 1)

    def test_exportable_result_handles_scalar_table_and_errors(self):
        scalar = exportable_result_dataframe({"success": True, "result_type": "number", "value": 42})
        self.assertEqual(scalar.iloc[0, 0], 42)

        table = pd.DataFrame({"Column": ["Sales"], "Average": [10.0]})
        converted = exportable_result_dataframe({"success": True, "result_type": "table", "value": table})
        pd.testing.assert_frame_equal(converted, table)
        converted.iloc[0, 1] = 99
        self.assertEqual(table.iloc[0, 1], 10.0, "export conversion must not mutate source data")

        self.assertIsNone(exportable_result_dataframe({"success": False, "result_type": "error"}))
        self.assertIsNone(exportable_result_dataframe({"success": True, "result_type": "unknown", "value": object()}))


class FollowupHandlerTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"Sales": [100, 200, 300], "Region": ["East", "West", "East"]})

    def test_detects_context_dependent_explanation_questions(self):
        self.assertTrue(is_followup_question("What does this result mean?"))
        self.assertTrue(is_followup_question("Can you explain that?"))
        self.assertFalse(is_followup_question("What is the average Sales?"))

    def test_numeric_followup_uses_previous_result_and_avoids_causal_claims(self):
        result = {"success": True, "result_type": "number", "value": 200, "answer": "The average is 200."}
        answer = explain_previous_result("Explain this", "What is the average Sales?", result, self.df)
        self.assertIn("200", answer)
        self.assertIn("What is the average Sales?", answer)
        self.assertIn("not evidence by itself", answer)

    def test_table_followup_describes_real_rows_and_truncates_preview(self):
        table = pd.DataFrame({"Region": ["East", "West", "North", "South"], "Value": [2, 1, 3, 4]})
        result = {"success": True, "result_type": "table", "value": table, "answer": "Table ready."}
        answer = explain_previous_result("Explain the result", "Count by region", result, self.df)
        self.assertIn("4 row(s)", answer)
        self.assertIn("Region: East", answer)
        self.assertIn("1 more row(s)", answer)

    def test_no_previous_successful_result_gives_safe_guidance(self):
        answer = explain_previous_result("Explain this", None, None, self.df)
        self.assertIn("Ask a data question first", answer)

    def test_grouped_result_does_not_claim_causation(self):
        result = {
            "success": True,
            "result_type": "dictionary",
            "value": pd.DataFrame({"Category": ["East", "West"], "Value": [2, 1]}),
            "answer": "Grouped counts ready.",
        }
        answer = explain_previous_result("Explain that", "Count by Region", result, self.df)
        self.assertIn("computed summaries", answer)
        self.assertIn("not proof of cause and effect", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
