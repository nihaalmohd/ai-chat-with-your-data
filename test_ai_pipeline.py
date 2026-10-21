
import pandas as pd
from ai_engine import understand_question
from data_analyzer import analyze_data
from question_router import route_question


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "HR": "HR-Employee-Attrition.csv",
    "Customer Shopping": "customer_shopping_behavior.csv",
}


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "dataset": "HR",
        "question": "How many employees are in the dataset?",
        # The Streamlit app handles this through its direct router.
        "expected_intent": "total_employees",
        "expected_value": 1470,
    },
    {
        "dataset": "HR",
        "question": "How many employees left?",
        "expected_intent": "employees_left",
    },
    {
        "dataset": "HR",
        "question": "Show employee count by department",
        "expected_intent": "group_count",
        "expected_group_by": "Department",
    },
    {
        "dataset": "HR",
        "question": "Show average monthly income by department",
        "expected_intent": "group_average",
        "expected_group_by": "Department",
    },
    {
        "dataset": "HR",
        "question": "Which department has highest attrition?",
        "expected_intent": "group_attrition",
        "expected_group_by": "Department",
    },
    {
        "dataset": "HR",
        "question": "Show distribution of MonthlyIncome",
        "expected_intent": "distribution",
    },
    {
        "dataset": "HR",
        "question": "Show relationship between Age and MonthlyIncome",
        "expected_intent": "relationship",
    },
    {
        "dataset": "Customer Shopping",
        "question": "How many customers are Male?",
        "expected_intent": "filtered_count",
        "expected_filter_column": "Gender",
    },
    {
        "dataset": "Customer Shopping",
        "question": "Show customer count by category",
        "expected_intent": "group_count",
        "expected_group_by": "Category",
    },
    {
        "dataset": "Customer Shopping",
        "question": "Show average purchase amount by category",
        "expected_intent": "group_average",
        "expected_group_by": "Category",
    },
    {
        "dataset": "Customer Shopping",
        "question": "Show distribution of Purchase Amount (USD)",
        "expected_intent": "distribution",
    },
    {
        "dataset": "Customer Shopping",
        "question": "Show relationship between Age and Purchase Amount (USD)",
        "expected_intent": "relationship",
    },
    {
        "dataset": "Customer Shopping",
        "question": "How many female customers are there?",
        "expected_intent": "filtered_count",
        "expected_filter_column": "Gender",
        "expected_value": 1248,
    },
    {
        "dataset": "Customer Shopping",
        "question": "Show average purchase by each customer",
        "expected_intent": "group_average",
        "expected_group_by": "Customer ID",
        "expected_column": "Purchase Amount (USD)",
    },
    {
        "dataset": "Customer Shopping",
        "question": "Show average purchases",
        "expected_intent": "unknown",
        "expected_success": False,
        "expected_warning": "Please specify the column",
    },
]


# ============================================================
# TEST EXECUTION
# ============================================================

def run_tests():

    loaded_datasets = {}
    passed = 0
    failed = 0
    results = []

    print("\n" + "=" * 70)
    print("AI CHATBOT — REGRESSION TEST SUITE")
    print("=" * 70)

    for name, path in DATASETS.items():

        try:
            loaded_datasets[name] = pd.read_csv(path)
            print(
                f"\nLoaded {name}: "
                f"{loaded_datasets[name].shape[0]} rows, "
                f"{loaded_datasets[name].shape[1]} columns"
            )

        except Exception as error:
            print(f"\nCould not load {name}: {error}")
            return

    for index, test in enumerate(TEST_CASES, start=1):

        dataset_name = test["dataset"]
        question = test["question"]
        df = loaded_datasets[dataset_name]

        print("\n" + "-" * 70)
        print(f"TEST {index}: {dataset_name}")
        print(f"QUESTION: {question}")

        try:

            # Mirror the same routing path used by app.py. Direct intents must
            # not be sent through Ollama, while semantic questions still use
            # the AI interpreter followed by deterministic Pandas analysis.
            route = route_question(question)

            if route.get("route") == "direct":
                request = {"intent": route.get("intent")}
            else:
                request = understand_question(question, df)

            result = analyze_data(df, request)

            errors = []

            actual_intent = request.get("intent")

            if actual_intent != test["expected_intent"]:
                errors.append(
                    f"Intent expected {test['expected_intent']}, "
                    f"got {actual_intent}"
                )

            if "expected_value" in test:
                actual_value = result.get("value")
                if actual_value != test["expected_value"]:
                    errors.append(
                        f"Value expected {test['expected_value']}, got {actual_value}"
                    )

            if "expected_group_by" in test:

                actual_group = request.get("group_by")

                if (
                    actual_group is None
                    or actual_group.lower()
                    != test["expected_group_by"].lower()
                ):
                    errors.append(
                        f"Group expected "
                        f"{test['expected_group_by']}, "
                        f"got {actual_group}"
                    )

            if "expected_column" in test:
                actual_column = request.get("column")
                if (
                    actual_column is None
                    or actual_column.lower() != test["expected_column"].lower()
                ):
                    errors.append(
                        f"Column expected {test['expected_column']}, got {actual_column}"
                    )

            if "expected_warning" in test:
                warning = request.get("semantic_warning", "")
                if test["expected_warning"].lower() not in warning.lower():
                    errors.append(
                        f"Expected clarification containing {test['expected_warning']!r}, got {warning!r}"
                    )

            if "expected_filter_column" in test:

                actual_filter = request.get("filter_column")

                if (
                    actual_filter is None
                    or actual_filter.lower()
                    != test["expected_filter_column"].lower()
                ):
                    errors.append(
                        f"Filter column expected "
                        f"{test['expected_filter_column']}, "
                        f"got {actual_filter}"
                    )

            expected_success = test.get("expected_success", True)
            if bool(result.get("success", False)) != expected_success:
                errors.append(
                    f"Analyzer success expected {expected_success}, "
                    f"got {result.get('success', False)}"
                )

            if errors:

                failed += 1
                status = "FAILED"

                for error in errors:
                    print("  ISSUE:", error)

            else:

                passed += 1
                status = "PASSED"

            print("AI INTENT:", actual_intent)
            print("RESULT TYPE:", result.get("result_type"))
            print(
                "ANSWER:",
                request.get("semantic_warning") or result.get("answer"),
            )
            print("STATUS:", status)

            results.append({
                "Test": index,
                "Dataset": dataset_name,
                "Question": question,
                "Intent": actual_intent,
                "Result Type": result.get("result_type"),
                "Status": status,
                "Issues": "; ".join(errors),
            })

        except Exception as error:

            failed += 1

            print("STATUS: FAILED")
            print("ERROR:", repr(error))

            results.append({
                "Test": index,
                "Dataset": dataset_name,
                "Question": question,
                "Intent": "ERROR",
                "Result Type": "ERROR",
                "Status": "FAILED",
                "Issues": repr(error),
            })

    # ========================================================
    # SUMMARY
    # ========================================================

    total = passed + failed

    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if total:
        print(f"Pass Rate: {passed / total * 100:.2f}%")

    report = pd.DataFrame(results)
    report.to_csv("test_report.csv", index=False)

    print("\nDetailed report saved to test_report.csv")

    if failed:
        print("\nSome tests failed. Review the issues above.")
    else:
        print("\nAll regression tests passed successfully.")


if __name__ == "__main__":
    run_tests()
