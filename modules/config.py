"""
Configuration constants for LLM Evaluation System.
"""

from pathlib import Path

# Available models for testing
AVAILABLE_MODELS = {
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.0 Flash (Latest)": "gemini-2.0-flash-exp",
    "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
    "Gemini 1.5 Flash": "gemini-1.5-flash-latest",
    "Gemini 1.0 Pro": "gemini-1.0-pro"
}

# Judge model for evaluation
JUDGE_MODEL = "gemini-2.5-pro"

# Pricing per 1M tokens (in USD)
MODEL_PRICING = {
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50}
}

# Database configuration
DB_PATH = Path("evaluation_history.db")

# API rate limiting (seconds)
API_RATE_LIMIT = 10  # Increased from 7 to 10 for better stability
MAX_RETRIES = 3