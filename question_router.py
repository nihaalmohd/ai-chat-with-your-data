import re


def normalize_question(question):
    question = str(question).strip().lower()
    question = re.sub(r"\s+", " ", question)
    return question


def route_question(question):

    question = normalize_question(
        question
    )

    # Route averages across every numeric column deterministically.
    # This must run before any LLM-backed interpretation.
    asks_average_all_numeric = (
        re.search(r"\b(?:average|avg|mean)\b", question)
        and re.search(r"\b(?:all|every|each)\b", question)
        and re.search(r"\b(?:numeric|numerical|number)\b", question)
        and re.search(r"\b(?:columns?|fields?|variables?)\b", question)
    )
    if asks_average_all_numeric:
        return {
            "route": "direct",
            "intent": "average_all_numeric",
        }

    # ========================================================
    # 1. GROUP COUNT
    # ========================================================
    # Must come before total employee patterns.
    #
    # Examples:
    # "employee count by department"
    # "show customer count by category"
    # "distribution by category"
    #
    # These require dataset-aware analysis.

    count_by_patterns = [
        "employee count by",
        "number of employees by",
        "how many employees by",
        "count of employees by",
        "count by",
        "distribution by"
    ]

    for pattern in count_by_patterns:

        if pattern in question:

            return {
                "route": "ollama",
                "intent": "group_count"
            }


    # ========================================================
    # 2. EMPLOYEES WHO LEFT
    # ========================================================
    # Must come BEFORE total employee patterns.
    #
    # Example:
    # "How many employees left?"
    #
    # should NEVER become:
    # "total_employees"

    employees_left_patterns = [
        "how many employees left",
        "how many employees have left",
        "employees who left",
        "employees that left",
        "number of employees who left",
        "number of employees that left",
        "how many people left",
        "people who left",
        "employee attrition count"
    ]

    for pattern in employees_left_patterns:

        if pattern in question:

            return {
                "route": "direct",
                "intent": "employees_left"
            }


    # ========================================================
    # 3. AVERAGE BY GROUP
    # ========================================================

    average_by_patterns = [
        "average by",
        "average salary by",
        "average income by",
        "average monthly income by",
        "mean by"
    ]

    for pattern in average_by_patterns:

        if pattern in question:

            return {
                "route": "ollama",
                "intent": "group_average"
            }


    # ========================================================
    # 4. GROUP ATTRITION
    # ========================================================

    highest_attrition_patterns = [
        "highest attrition",
        "highest attrition by",
        "department with the highest attrition",
        "job role with the highest attrition",
        "which department has the highest attrition",
        "which job role has the highest attrition"
    ]

    for pattern in highest_attrition_patterns:

        if pattern in question:

            return {
                "route": "ollama",
                "intent": "group_attrition"
            }


    # ========================================================
    # 5. ATTRITION RATE
    # ========================================================

    attrition_patterns = [
        "attrition rate",
        "what is the attrition",
        "employee attrition percentage",
        "percentage of employees who left",
        "percentage of employees that left"
    ]

    for pattern in attrition_patterns:

        if pattern in question:

            return {
                "route": "direct",
                "intent": "attrition_rate"
            }


    # ========================================================
    # 6. TOTAL EMPLOYEES
    # ========================================================
    #
    # IMPORTANT:
    #
    # Do NOT use a broad substring such as:
    #
    #     "how many employees"
    #
    # because that incorrectly captures filtered questions:
    #
    #     "How many employees are in Sales?"
    #     "How many employees work in Finance?"
    #     "How many employees have a high salary?"
    #
    # Total employee questions must describe the entire dataset.
    #
    # Examples that belong here:
    #
    #     "How many employees are there?"
    #     "How many employees are in the dataset?"
    #     "Total number of employees"
    #     "Total employees"
    #     "Number of employees"
    #     "Employee count"
    #
    # Questions containing additional conditions are allowed to
    # continue to the AI analysis route.

    total_employee_patterns = [
        "how many employees are there",
        "how many employees are in the dataset",
        "total number of employees",
        "total employees",
        "number of employees",
        "employee count"
    ]

    # --------------------------------------------------------
    # Exact/whole-question protection
    # --------------------------------------------------------
    #
    # "number of employees" should be treated as a total only
    # when it is not followed by another analytical condition.
    #
    # This prevents:
    #
    # "number of employees in Sales"
    #
    # from becoming total_employees.

    total_employee_regex_patterns = [
        r"^how\s+many\s+employees\s+are\s+there\??$",
        r"^how\s+many\s+employees\s+are\s+in\s+the\s+dataset\??$",
        r"^total\s+number\s+of\s+employees\??$",
        r"^total\s+employees\??$",
        r"^number\s+of\s+employees\??$",
        r"^employee\s+count\??$"
    ]

    for pattern in total_employee_regex_patterns:

        if re.search(
            pattern,
            question,
            re.IGNORECASE
        ):

            return {
                "route": "direct",
                "intent": "total_employees"
            }


    # ========================================================
    # 7. COLUMN COUNT
    # ========================================================

    column_patterns = [
        "how many columns",
        "number of columns",
        "total columns",
        "column count"
    ]

    for pattern in column_patterns:

        if pattern in question:

            return {
                "route": "direct",
                "intent": "column_count"
            }


    # ========================================================
    # 8. ROW COUNT
    # ========================================================

    row_patterns = [
        "how many rows",
        "number of rows",
        "total rows",
        "row count",
        "records in the dataset",
        "how many records"
    ]

    for pattern in row_patterns:

        if pattern in question:

            return {
                "route": "direct",
                "intent": "row_count"
            }


    # ========================================================
    # 9. MISSING VALUES
    # ========================================================

    missing_patterns = [
        "missing values",
        "missing data",
        "null values",
        "null data",
        "how many missing"
    ]

    for pattern in missing_patterns:

        if pattern in question:

            return {
                "route": "direct",
                "intent": "missing_values"
            }


    # ========================================================
    # 10. UNIQUE VALUES
    # ========================================================

    unique_patterns = [
        "unique values",
        "distinct values",
        "how many unique"
    ]

    for pattern in unique_patterns:

        if pattern in question:

            return {
                "route": "ollama",
                "intent": "unique_values"
            }


    # ========================================================
    # 11. DEFAULT AI ANALYSIS
    # ========================================================

    return {
        "route": "ollama",
        "intent": "ai_analysis"
    }
