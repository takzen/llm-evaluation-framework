"""
Cost tracking functionality for API usage monitoring.
"""

import streamlit as st
from datetime import datetime
from modules.config import MODEL_PRICING


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate cost based on token usage and model pricing.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_name: Name of the model used
        
    Returns:
        Total cost in USD
    """
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing['input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']
    
    return input_cost + output_cost


def init_cost_tracker():
    """Initialize cost tracker in session state if not exists."""
    if 'cost_tracker' not in st.session_state:
        st.session_state.cost_tracker = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0,
            'api_calls': 0,
            'session_start': datetime.now()
        }


def update_cost_tracker(input_tokens: int, output_tokens: int, model_name: str):
    """
    Update the session cost tracker with new API call data.
    
    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        model_name: Name of the model used
    """
    init_cost_tracker()
    
    cost = calculate_cost(input_tokens, output_tokens, model_name)
    
    st.session_state.cost_tracker['total_input_tokens'] += input_tokens
    st.session_state.cost_tracker['total_output_tokens'] += output_tokens
    st.session_state.cost_tracker['total_cost'] += cost
    st.session_state.cost_tracker['api_calls'] += 1


def reset_cost_tracker():
    """Reset the cost tracker to initial state."""
    st.session_state.cost_tracker = {
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_cost': 0.0,
        'api_calls': 0,
        'session_start': datetime.now()
    }


def get_cost_tracker():
    """
    Get current cost tracker data.
    
    Returns:
        Dictionary with cost tracking information
    """
    init_cost_tracker()
    return st.session_state.cost_tracker


def display_cost_metrics():
    """Display cost tracking metrics in sidebar."""
    tracker = get_cost_tracker()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Cost Tracking")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Cost", f"${tracker['total_cost']:.4f}")
    with col2:
        st.metric("API Calls", tracker['api_calls'])
    
    with st.sidebar.expander("📊 Token Usage Details"):
        st.write(f"**Input Tokens:** {tracker['total_input_tokens']:,}")
        st.write(f"**Output Tokens:** {tracker['total_output_tokens']:,}")
        st.write(f"**Total Tokens:** {tracker['total_input_tokens'] + tracker['total_output_tokens']:,}")
        
        session_duration = (datetime.now() - tracker['session_start']).seconds // 60
        st.write(f"**Session Duration:** {session_duration} min")
        
        if st.button("🔄 Reset Tracker", width='stretch'):
            reset_cost_tracker()
            st.rerun()