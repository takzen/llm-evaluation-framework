# Advanced LLM Evaluation Framework

### An interactive, enterprise-grade framework for evaluating, red-teaming, and benchmarking Large Language Models (LLMs), featuring automated test set generation, multi-metric AI-powered judging, and advanced data visualization.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-orange?logo=streamlit) ![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.5-blue?logo=google-gemini) ![Plotly](https://img.shields.io/badge/Plotly-6.3.1-blue?logo=plotly)

## 🚀 Overview

This project moves beyond simple LLM application development into the critical domain of **AI Quality Assurance and MLOps**. It provides a robust, interactive tool for systematically evaluating and comparing the performance of Large Language Models on specific, context-aware tasks.

The framework automates the entire evaluation pipeline, from generating high-quality test data to scoring model responses using an advanced AI judge. It is designed to be a practical tool for prompt engineers, AI developers, and data scientists to make data-driven decisions about which model or system prompt is best suited for their specific use case.

## ✨ Key Features & Techniques

*   **Meta-AI (AI-Powered Judging):** Implements a state-of-the-art evaluation strategy where a powerful "judge" LLM (`Gemini 2.5 Pro`) quantitatively scores the responses of other models. This demonstrates a deep understanding of modern LLM evaluation techniques.
*   **Multi-Metric Evaluation:** The AI judge assesses responses across four critical dimensions: **Factual Consistency, Helpfulness, Relevance, and Completeness**, providing a nuanced and comprehensive view of model quality.
*   **Automated Test Set Generation:** Users can provide any text-based context, and the application uses a powerful LLM to automatically generate a relevant and diverse set of questions and ground-truth answers for testing.
*   **Model & Prompt Benchmarking:** The interface allows users to select from multiple available models (e.g., `Gemini 2.0 Pro`, `Gemini 2.0 Flash`) and customize the system prompt, enabling direct, A/B testing of different configurations.
*   **Advanced Data Visualization:** The dashboard presents evaluation results using sophisticated **Plotly** visualizations, including:
    *   A **Radar Chart** for a high-level overview of a model's strengths and weaknesses across all metrics.
    *   **Grouped Bar Charts** for a detailed, per-question breakdown of scores.
*   **Robust Engineering Practices:**
    *   **API Resilience:** Implements a `safe_api_call` wrapper with rate limiting and an exponential backoff retry mechanism to handle API errors gracefully.
    *   **Error Handling:** Includes robust error handling for JSON parsing and API failures.
    *   **User Experience:** A clean, intuitive UI with spinners, progress bars, and success/warning messages provides clear feedback to the user.
*   **Data Export:** Allows users to download the detailed evaluation results as either a **CSV or JSON** file for further offline analysis.

## 🛠️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/takzen/llm-evaluation-framework.git
    cd llm-evaluation-framework
    ```

2.  **Set up your Google AI API Key:**
    *   Create a file named `.env` in the root of the project.
    *   Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Add your key to the `.env` file: `GOOGLE_API_KEY="YOUR_API_KEY_HERE"`

3.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create and activate the virtual environment
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install the required packages
    uv pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```

## 🖼️ Application Showcase

The application provides a seamless, step-by-step workflow for comprehensive LLM evaluation.

| **Step 1: Initial State** | **Step 2: Test Set Generation** |
| :---: | :---: |
| ![Initial State](images/01_initial_state.png) | ![Test Set Generation](images/02_test_set_generation.png) |
| *The user is presented with a clean interface, ready for configuration in the sidebar.* | *After providing a context and selecting parameters, a high-quality test set is automatically generated.* |

| **Step 3: High-Level Evaluation Dashboard** | **Step 4: Detailed Results & Export** |
| :---: | :---: |
| ![Evaluation Dashboard](images/03_evaluation_dashboard.png) | ![Detailed Results and Export](images/04_detailed_results_and_export.png) |
| *Upon running the evaluation, the app displays a dashboard with summary metrics and interactive visualizations like a radar chart for a quick overview of model performance.* | *The user can drill down into a detailed table with per-question scores and the AI judge's reasoning, and export the full results as a CSV or JSON file.* |