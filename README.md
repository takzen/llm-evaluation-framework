# Advanced LLM Evaluation Framework

### An interactive, enterprise-grade framework for evaluating, red-teaming, and benchmarking Large Language Models (LLMs), featuring automated test set generation, multi-metric AI-powered judging, and advanced data visualization.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-orange?logo=streamlit) ![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.0-blue?logo=google-gemini) ![Plotly](https://img.shields.io/badge/Plotly-6.3.1-blue?logo=plotly)

## 🚀 Overview

This project moves beyond simple LLM application development into the critical domain of **AI Quality Assurance and MLOps**. It provides a robust, interactive tool for systematically evaluating and comparing the performance of Large Language Models on specific, context-aware tasks.

The framework automates the entire evaluation pipeline, from generating high-quality test data to scoring model responses using an advanced AI judge. It is designed to be a practical tool for prompt engineers, AI developers, and data scientists to make data-driven decisions about which model or system prompt is best suited for their specific use case.

## ✨ Key Features & Techniques

*   **Meta-AI (AI-Powered Judging):** Implements a state-of-the-art evaluation strategy where a powerful "judge" LLM (`Gemini 2.0 Flash`) quantitatively scores the responses of other models. This demonstrates a deep understanding of modern LLM evaluation techniques.
*   **Multi-Metric Evaluation:** The AI judge assesses responses across four critical dimensions: **Factual Consistency, Helpfulness, Relevance, and Completeness**, providing a nuanced and comprehensive view of model quality.
*   **Automated Test Set Generation:** Users can provide any text-based context, and the application uses a powerful LLM to automatically generate a relevant and diverse set of questions and ground-truth answers for testing.
*   **Model & Prompt Benchmarking:** The interface allows users to select from multiple available models (e.g., `Gemini 1.5 Pro`, `Gemini 1.0 Pro`) and customize the system prompt, enabling direct, A/B testing of different configurations.
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

| 1. Configuration and Test Set Generation | 2. Comprehensive Evaluation Dashboard |
| :---: | :---: |
| *(Image of the full Streamlit app, showing the sidebar with model selection and context input, and the main area with the generated test set table)* | *(Image of the final results dashboard, highlighting the radar chart, the summary metrics, and the detailed results table below)* |
| *The user selects a model to test, provides a context, and generates a custom evaluation set.* | *The app displays a rich dashboard with average scores, a radar chart for a visual summary, and a detailed breakdown of the AI judge's scores for each question.* |