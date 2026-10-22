"""Offline regression tests for the data-agnostic analysis pipeline.

These tests intentionally do not require a running Ollama server. They verify
that deterministic paths, profiling, routing, and Pandas calculations behave
correctly before an optional live-model integration test is run.
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ai_engine imports the Ollama client at module import time. Provide a fail-fast
# test double so count questions prove they do not need an LLM call.
if "ollama" not in sys.modules:
    sys.modules["ollama"] = types.SimpleNamespace(
        chat=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Ollama should not be called for a plain total count")
        )
    )

from ai_engine import (
    chat_with_model,
    detect_plain_count_question,
    normalize_result,
    understand_question,
)
from data_analyzer import analyze_data
from data_profiler import profile_dataset
from dataset_understanding import understand_dataset
from question_router import route_question


class PlainCountRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [
            ROOT / "HR-Employee-Attrition.csv",
            ROOT / "customer_shopping_behavior.csv",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "Optional local validation dataset(s) not present: "
                + ", ".join(missing)
            )
        cls.hr = pd.read_csv(required[0])
        cls.customer = pd.read_csv(required[1])

    def test_previously_failing_total_count_questions(self):
        cases = [
            (self.hr, "What is the total number of employees?", 1470),
            (self.hr, "How many people are employed?", 1470),
            (self.hr, "How many employees are in the dataset?", 1470),
            (self.customer, "How many customers are in the dataset?", 3900),
            (self.customer, "How many records are there?", 3900),
        ]
        for df, question, expected in cases:
            with self.subTest(question=question):
                request = understand_question(question, df)
                self.assertEqual(request.get("intent"), "count")
                self.assertIsNone(request.get("filter_column"))
                self.assertIsNone(request.get("filter_value"))
                result = analyze_data(df, request)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["value"], expected)

    def test_normalizer_removes_hallucinated_filter_from_total_count(self):
        model_result = {
            "intent": "filtered_count",
            "filter_column": "Attrition",
            "filter_value": "Yes",
            "y_column": "EmployeeNumber",
        }
        normalized = normalize_result(
            model_result,
            "What is the total number of employees?",
            self.hr,
        )
        self.assertEqual(normalized["intent"], "count")
        self.assertIsNone(normalized["filter_column"])
        self.assertIsNone(normalized["filter_value"])
        self.assertIsNone(normalized["y_column"])

    def test_filtered_questions_are_not_plain_counts(self):
        questions = [
            "How many customers are Male?",
            "How many employees work in Sales?",
            "How many employees are there in Sales?",
            "How many records have missing values?",
            "How many employees left?",
            "Show employee count by department",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertFalse(detect_plain_count_question(question))

    def test_direct_router_total_and_filtered_routes(self):
        self.assertEqual(
            route_question("How many employees are in the dataset?")["intent"],
            "total_employees",
        )
        self.assertEqual(
            route_question("How many employees work in Sales?")["route"],
            "ollama",
        )

    def test_app_routing_handles_employees_left_without_ollama(self):
        route = route_question("How many employees left?")
        self.assertEqual(route["route"], "direct")
        self.assertEqual(route["intent"], "employees_left")
        result = analyze_data(self.hr, {"intent": route["intent"]})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["value"], 237)

    def test_female_customer_count_wording_variations(self):
        questions = [
            "How many female customers are there?",
            "How many customers are female?",
        ]
        for question in questions:
            with self.subTest(question=question):
                request = understand_question(question, self.customer)
                self.assertEqual(request.get("intent"), "filtered_count")
                self.assertEqual(request.get("filter_column"), "Gender")
                self.assertEqual(str(request.get("filter_value")).lower(), "female")
                result = analyze_data(self.customer, request)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["value"], 1248)

    def test_average_purchase_by_each_customer_resolves_columns(self):
        request = understand_question(
            "Show average purchase by each customer", self.customer
        )
        self.assertEqual(request.get("intent"), "group_average")
        self.assertEqual(request.get("column"), "Purchase Amount (USD)")
        self.assertEqual(request.get("group_by"), "Customer ID")
        result = analyze_data(self.customer, request)
        self.assertTrue(result["success"], result)
        self.assertEqual(result.get("result_type"), "dictionary")

    def test_ambiguous_average_purchases_requests_clarification(self):
        request = understand_question("Show average purchases", self.customer)
        self.assertEqual(request.get("intent"), "unknown")
        self.assertIn("Please specify the column", request.get("semantic_warning", ""))

    def test_chat_provider_uses_local_ollama_without_hosted_key(self):
        fake_response = {"message": {"content": '{"intent":"count"}'}}
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            with patch("ai_engine.ollama.chat", return_value=fake_response) as local_chat:
                response = chat_with_model([{"role": "user", "content": "test"}])
        self.assertEqual(response, fake_response)
        local_chat.assert_called_once()
        self.assertEqual(local_chat.call_args.kwargs["model"], "llama3.2:3b")

    def test_chat_provider_supports_hosted_groq_key(self):
        class FakeCompletion:
            choices = [types.SimpleNamespace(message=types.SimpleNamespace(content='{"intent":"count"}'))]

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeCompletion()

        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
        fake_groq_module = types.SimpleNamespace(Groq=lambda **kwargs: fake_client)
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
            with patch.dict(sys.modules, {"groq": fake_groq_module}):
                response = chat_with_model([{"role": "user", "content": "test"}])
        self.assertEqual(response["message"]["content"], '{"intent":"count"}')
        self.assertEqual(fake_client.chat.completions.kwargs["model"], "openai/gpt-oss-20b")
        self.assertEqual(fake_client.chat.completions.kwargs["response_format"], {"type": "json_object"})


class DataAgnosticProfilingTests(unittest.TestCase):
    def test_known_datasets_profile(self):
        cases = [
            ("HR-Employee-Attrition.csv", (1470, 35)),
            ("customer_shopping_behavior.csv", (3900, 18)),
        ]
        available_cases = [
            (filename, shape)
            for filename, shape in cases
            if (ROOT / filename).is_file()
        ]
        if not available_cases:
            self.skipTest(
                "Optional local validation datasets are not present; "
                "dataset-independent fixtures are tested separately."
            )

        for filename, expected_shape in available_cases:
            df = pd.read_csv(ROOT / filename)
            profile = profile_dataset(df)
            self.assertEqual((profile["rows"], profile["columns"]), expected_shape)
            self.assertIn("column_roles", profile)
            self.assertEqual(len(profile["column_roles"]), df.shape[1])
            understanding = understand_dataset(df)
            self.assertEqual(len(understanding["column_roles"]), df.shape[1])

    def test_non_string_column_names(self):
        df = pd.DataFrame({1: [10, 20], 2: ["A", "B"]})
        profile = profile_dataset(df)
        self.assertEqual(profile["rows"], 2)
        self.assertEqual(len(profile["column_roles"]), 2)

    def test_empty_dataframe_profile(self):
        profile = profile_dataset(pd.DataFrame())
        self.assertEqual(profile["rows"], 0)
        self.assertEqual(profile["columns"], 0)


class DeterministicAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hr_path = ROOT / "HR-Employee-Attrition.csv"
        customer_path = ROOT / "customer_shopping_behavior.csv"
        missing = [
            path.name for path in (hr_path, customer_path)
            if not path.is_file()
        ]
        if missing:
            raise unittest.SkipTest(
                "Optional local validation dataset(s) not present: "
                + ", ".join(missing)
            )
        cls.hr = pd.read_csv(hr_path)
        cls.customer = pd.read_csv(customer_path)

    def test_hr_core_results(self):
        cases = [
            (self.hr, {"intent": "total_employees"}, 1470),
            (self.hr, {"intent": "employees_left"}, 237),
            (self.hr, {"intent": "row_count"}, 1470),
            (self.hr, {"intent": "column_count"}, 35),
            (self.hr, {"intent": "filtered_count", "filter_column": "Department", "filter_value": "Sales"}, 446),
        ]
        for df, request, expected in cases:
            with self.subTest(request=request):
                result = analyze_data(df, request)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["value"], expected)

    def test_customer_filtered_count(self):
        result = analyze_data(
            self.customer,
            {"intent": "filtered_count", "filter_column": "Gender", "filter_value": "Male"},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["value"], 2652)

    def test_average_all_numeric_routes_and_calculates_without_llm(self):
        question = "Show the average of every numeric column in the Orders sheet"
        route = route_question(question)
        self.assertEqual(route, {
            "route": "direct",
            "intent": "average_all_numeric",
        })

        df = pd.DataFrame({
            "Sales": [100, 200, 300],
            "Quantity": [1, 2, 3],
            "Profit": [10, 20, 30],
            "Region": ["East", "West", "East"],
        })
        result = analyze_data(df, {"intent": route["intent"]})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["result_type"], "table")
        self.assertEqual(
            result["value"].to_dict("records"),
            [
                {"Column": "Sales", "Average": 200.0},
                {"Column": "Quantity", "Average": 2.0},
                {"Column": "Profit", "Average": 20.0},
            ],
        )

    def test_unique_values_intent(self):
        df = pd.DataFrame({"Color": ["red", "blue", "red", None]})
        result = analyze_data(df, {"intent": "unique_values", "column": "Color"})
        self.assertTrue(result["success"], result)
        self.assertEqual(set(result["value"].columns), {"Category", "Value"})
        self.assertEqual(int(result["value"]["Value"].sum()), 4)

    def test_missing_column_returns_controlled_error(self):
        result = analyze_data(self.customer, {"intent": "average", "column": "not_a_real_column"})
        self.assertFalse(result["success"])
        self.assertEqual(result["result_type"], "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
