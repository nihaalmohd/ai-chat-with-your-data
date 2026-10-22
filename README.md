# AI Chat With Your Data

An AI-powered Streamlit application that lets users explore structured datasets using natural-language questions. The AI interprets the question, while Pandas performs deterministic calculations on the uploaded data.

The project is designed to support varied datasets and column names instead of relying on fixed HR or customer-specific schemas.

## Current Capabilities

- Upload CSV and Excel (`.xlsx`) files, with worksheet selection for multi-sheet workbooks.
- Download the uploaded dataset or successful analysis results as CSV or Excel.
- Cache file parsing briefly to reduce repeated work during chat interactions.
- Display schema-aware example questions to help users explore a new dataset.
- Explore dataset dimensions, column types, missing values, duplicates, and inferred column roles.
- Ask natural-language questions using local Ollama or hosted Groq inference.
- Perform supported calculations, including counts, filtered counts, averages, minimums, maximums, grouped counts, grouped averages, sums, distributions, and relationships.
- Generate Plotly visualizations for supported analysis results.
- Validate analysis behavior using regression tests and multiple datasets.

## Important Scope and Data Notes

**Dataset-agnostic design:** The analysis logic is designed to work with varied column names and data domains. However, no application can guarantee correct interpretation of every possible question or file schema.

The current interface supports CSV and `.xlsx` files. The AI interpretation mode and supported analysis operations depend on the question, dataset structure, and configured model.

**Validation datasets are not included in this public repository.** The HR Employee Attrition and Customer Shopping datasets were used during development and local testing, but their original sources and redistribution permissions have not been verified. Do not assume that these files are licensed for redistribution.

To reproduce dataset-dependent tests, place your legally obtained copies of the required validation datasets in the project directory using the filenames expected by the relevant test scripts. Do not commit private or unlicensed datasets to GitHub.

## Setup on Windows PowerShell

Open PowerShell in the project directory.

Create a virtual environment and install the dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run Locally with Ollama

Install Ollama separately using its official installer. Download the model used by the application:

```powershell
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

Keep Ollama running. In a second PowerShell window, navigate to the project directory and start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app should open in your browser. Upload a dataset and try a question supported by the current analysis pipeline.

## Automated Regression Checks

GitHub Actions runs syntax compilation and the offline regression suite on pushes to the stabilization branches and on pull requests to `main`. The core tests use synthetic datasets so they do not depend on private files or an external model. Tests that validate the HR and customer examples are skipped when those optional local datasets are absent.

## Run Tests

Run the following commands from the project directory:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py ai_engine.py data_analyzer.py data_profiler.py dataset_understanding.py insight_generator.py question_router.py visualization_engine.py
```

Run the regression test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run the router tests:

```powershell
.\.venv\Scripts\python.exe test_router.py
```

The core dataset-agnostic tests run with synthetic fixtures. Some historical HR/customer regression tests are optional and will be skipped when those local validation datasets are not present.

For the AI pipeline tests, make sure the configured inference provider is available. If running with local Ollama, start Ollama and ensure `llama3.2:3b` is installed:

```powershell
.\.venv\Scripts\python.exe test_ai_pipeline.py
```

Model-generated interpretations can vary. Review the detected intent, selected columns, and calculated values when assessing test results.

## Deploy on Streamlit Community Cloud

The application supports two inference modes.

- **Local development:** Use Ollama with `llama3.2:3b` when no `GROQ_API_KEY` is configured.
- **Cloud deployment:** Configure `GROQ_API_KEY` in Streamlit Community Cloud's app secrets. The application can then use Groq-hosted inference without requiring Ollama to run on your personal computer.

The current hosted-model default is `openai/gpt-oss-20b`. Model availability, provider settings, and API quotas may change. The application also supports overriding the model with `GROQ_MODEL` where configured in the code.

### Deployment Checklist

1. Push the application source code, dependency file, README, and appropriate tests to GitHub.
2. Do not upload virtual environments, API credentials, `.streamlit/secrets.toml`, or validation datasets whose redistribution permissions have not been established.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app using your GitHub repository.
4. Select `app.py` as the application entry point.
5. In the app's secrets settings, configure your Groq API key:

   ```toml
   GROQ_API_KEY = "your-provider-api-key"
   ```

   Replace the placeholder with your actual key in the hosting platform's secrets settings. Never commit the real key to GitHub or share it publicly.

6. Deploy the application and inspect the build logs.
7. Test file upload, a total-count question, a filtered-count question, a grouped-average question, and an ambiguous question that should request clarification.

### Official Documentation

- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit secrets management](https://docs.streamlit.io/deploy/concepts/secrets)
- [Groq Console](https://console.groq.com/)

## Project Principles

- Use AI to interpret natural-language questions, not to fabricate calculated results.
- Use Pandas for deterministic data calculations.
- Avoid hard-coded assumptions about specific datasets.
- Validate changes against multiple datasets when available.
- Keep credentials, private data, and local development artifacts out of the public repository.
