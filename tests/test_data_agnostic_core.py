"""Dataset-agnostic deterministic tests that run without private CSV fixtures."""
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid requiring a running local model for deterministic-path tests.
if "ollama" not in sys.modules:
    sys.modules["ollama"] = types.SimpleNamespace(
        chat=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("An LLM call is not expected in these tests")
        )
    )

from data_analyzer import analyze_data
from data_profiler import profile_dataset
from question_router import route_question
from dataset_understanding import understand_dataset


class DataAgnosticAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Revenue": [100.0, 200.0, None, 400.0],
            "Units": [1, 2, 3, 4],
            "Region": ["East", "West", "East", None],
            "Customer Code": ["A1", "A2", "A3", "A4"],
        })

    def test_total_row_and_column_counts(self):
        self.assertEqual(analyze_data(self.df, {"intent": "row_count"})["value"], 4)
        self.assertEqual(analyze_data(self.df, {"intent": "column_count"})["value"], 4)

    def test_average_skips_missing_values_and_is_dataset_agnostic(self):
        result = analyze_data(self.df, {"intent": "average", "column": "Revenue"})
        self.assertTrue(result["success"], result)
        self.assertAlmostEqual(result["value"], 700 / 3, places=8)

    def test_average_all_numeric_columns_excludes_text_columns(self):
        result = analyze_data(self.df, {"intent": "average_all_numeric"})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["result_type"], "table")
        self.assertEqual(result["value"]["Column"].tolist(), ["Revenue", "Units"])
        self.assertEqual(result["value"]["Average"].tolist(), [233.33, 2.5])

    def test_average_all_numeric_fails_cleanly_without_numeric_columns(self):
        result = analyze_data(pd.DataFrame({"A": ["x", "y"]}), {"intent": "average_all_numeric"})
        self.assertFalse(result["success"])
        self.assertEqual(result["result_type"], "error")

    def test_filtered_count_matches_case_insensitively(self):
        result = analyze_data(
            self.df,
            {"intent": "filtered_count", "filter_column": "region", "filter_value": "east"},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["value"], 2)

    def test_group_average_uses_actual_column_names(self):
        result = analyze_data(
            self.df,
            {"intent": "group_average", "column": "Revenue", "group_by": "Region"},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["result_type"], "dictionary")
        values = result["value"].set_index("Category")["Value"].to_dict()
        self.assertEqual(values["East"], 100.0)
        self.assertEqual(values["West"], 200.0)

    def test_invalid_column_returns_controlled_error(self):
        result = analyze_data(self.df, {"intent": "average", "column": "missing"})
        self.assertFalse(result["success"])
        self.assertEqual(result["result_type"], "error")

    def test_average_all_numeric_question_routes_without_model(self):
        for question in (
            "Show the average of every numeric column",
            "Give me the mean for all numerical fields",
            "Calculate avg of each number column",
        ):
            with self.subTest(question=question):
                route = route_question(question)
                self.assertEqual(route, {"route": "direct", "intent": "average_all_numeric"})

    def test_profile_and_understanding_support_arbitrary_schema(self):
        profile = profile_dataset(self.df)
        understanding = understand_dataset(self.df)
        self.assertEqual(profile["rows"], 4)
        self.assertEqual(profile["columns"], 4)
        self.assertEqual(len(profile["column_roles"]), 4)
        self.assertEqual(len(understanding["column_roles"]), 4)

    def test_empty_dataset_profile_is_safe(self):
        profile = profile_dataset(pd.DataFrame())
        self.assertEqual(profile["rows"], 0)
        self.assertEqual(profile["columns"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
