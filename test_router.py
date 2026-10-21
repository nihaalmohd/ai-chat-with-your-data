from question_router import route_question


questions = [
    "How many employees are in the dataset?",
    "How many employees left?",
    "What is the attrition rate?",
    "How many columns are there?",
    "How many rows are there?",
    "How many missing values are there?",
    "What is the average monthly income?",
    "Which department has the highest attrition?"
]


for question in questions:

    result = route_question(question)

    print("----------------------------------------")
    print("Question:", question)
    print("Route:", result["route"])
    print("Intent:", result["intent"])
