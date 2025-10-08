"""
LLM evaluation functions including test generation and AI judging.
"""

import json
import streamlit as st
import google.generativeai as genai
from modules.api_calls import safe_api_call
from modules.cost_tracker import update_cost_tracker
from utils.token_estimator import estimate_tokens


@st.cache_data(show_spinner=False)
def generate_test_set(context: str, num_questions: int, _model_name: str):
    """
    Generates a test set using a powerful LLM.
    
    Args:
        context: Text context to generate questions from
        num_questions: Number of questions to generate
        _model_name: Name of the model to use (underscore prevents caching issues)
        
    Returns:
        List of dictionaries with 'question' and 'answer' keys, or None if failed
    """
    model = genai.GenerativeModel(_model_name)
    prompt = f"""
    Based on the following context, generate {num_questions} diverse questions.
    For each question, provide a concise, factual "ground truth" answer based SOLELY on the context.
    
    Output MUST be a valid JSON array of objects with keys "question" and "answer".
    Provide ONLY the JSON array, no markdown formatting, no explanations.
    
    CONTEXT:
    ---
    {context}
    ---
    
    Example output format:
    [
        {{"question": "What is...", "answer": "The answer is..."}},
        {{"question": "How does...", "answer": "It works by..."}}
    ]
    """
    
    def _generate():
        response = model.generate_content(prompt)
        
        # Track tokens and cost
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(response.text)
        update_cost_tracker(input_tokens, output_tokens, _model_name)
        
        return response.text.strip()
    
    response_text = safe_api_call(_generate)
    if not response_text:
        return None
    
    try:
        cleaned = response_text.replace("```json", "").replace("```", "").strip()
        test_set = json.loads(cleaned)
        
        # Validate structure
        if isinstance(test_set, list) and all(
            isinstance(d, dict) and 'question' in d and 'answer' in d 
            for d in test_set
        ):
            return test_set
        else:
            st.error("❌ Generated test set has invalid format.")
            return None
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON parsing error: {e}")
        st.code(response_text[:500])
        return None


def get_model_response(context: str, question: str, system_prompt: str, model_name: str):
    """
    Gets a response from a specified Gemini model.
    
    Args:
        context: Background context for the question
        question: Question to answer
        system_prompt: System instruction for the model
        model_name: Name of the model to use
        
    Returns:
        Model's response text or error message
    """
    def _get_response():
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
        full_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        response = model.generate_content(full_prompt)
        
        # Track tokens and cost
        input_tokens = estimate_tokens(system_prompt + full_prompt)
        output_tokens = estimate_tokens(response.text)
        update_cost_tracker(input_tokens, output_tokens, model_name)
        
        return response.text.strip()
    
    result = safe_api_call(_get_response)
    return result if result else "Error: Failed to get response"


@st.cache_data(show_spinner=False)
def evaluate_response_with_judge(
    question: str, 
    ground_truth: str, 
    model_response: str, 
    _judge_model: str
):
    """
    Uses a powerful LLM as a 'judge' to evaluate a response.
    
    Args:
        question: The original question
        ground_truth: Expected correct answer
        model_response: Model's actual response
        _judge_model: Name of the judge model (underscore prevents caching issues)
        
    Returns:
        Dictionary with evaluation scores and reasoning
    """
    judge = genai.GenerativeModel(_judge_model)
    
    judge_prompt = f"""
    You are an impartial AI evaluator. Assess the quality of this language model response.
    
    Evaluate on these metrics (1-5 scale, where 5 is excellent):
    1. **Factual Consistency**: How well does the response align with the ground truth?
    2. **Helpfulness**: How helpful and direct is the response?
    3. **Relevance**: Does the response actually answer the question?
    4. **Completeness**: Is the response sufficiently detailed?
    
    Output MUST be valid JSON with these exact keys: "factual_consistency", "helpfulness", 
    "relevance", "completeness", "reasoning".
    Provide ONLY the JSON object, no markdown, no explanations.
    
    ---
    QUESTION: {question}
    ---
    GROUND TRUTH: {ground_truth}
    ---
    MODEL RESPONSE: {model_response}
    ---
    
    Example output:
    {{"factual_consistency": 4, "helpfulness": 5, "relevance": 5, "completeness": 4, "reasoning": "The response..."}}
    """
    
    def _evaluate():
        response = judge.generate_content(judge_prompt)
        
        # Track tokens and cost
        input_tokens = estimate_tokens(judge_prompt)
        output_tokens = estimate_tokens(response.text)
        update_cost_tracker(input_tokens, output_tokens, _judge_model)
        
        return response.text.strip()
    
    response_text = safe_api_call(_evaluate)
    if not response_text:
        return {
            "factual_consistency": 0, "helpfulness": 0, 
            "relevance": 0, "completeness": 0,
            "reasoning": "Evaluation failed"
        }
    
    try:
        cleaned = response_text.replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(cleaned)
        
        # Validate all required keys exist
        required_keys = [
            "factual_consistency", "helpfulness", 
            "relevance", "completeness", "reasoning"
        ]
        if not all(key in evaluation for key in required_keys):
            raise ValueError("Missing required keys in evaluation")
        
        return evaluation
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "factual_consistency": 0, "helpfulness": 0,
            "relevance": 0, "completeness": 0,
            "reasoning": f"Parse error: {str(e)}"
        }