import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import json
import time
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Evaluation System", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="⚖️"
)

# --- Load API Key and Configure Gemini ---
load_dotenv()
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
        st.info("Please create a .env file with your GOOGLE_API_KEY.")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Google AI: {e}")
    st.stop()

# --- Constants ---
AVAILABLE_MODELS = {
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.0 Flash (Latest)": "gemini-2.0-flash-exp",
    "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
    "Gemini 1.5 Flash": "gemini-1.5-flash-latest",
    "Gemini 1.0 Pro": "gemini-1.0-pro"
}

JUDGE_MODEL = "gemini-2.5-pro"

# Pricing per 1M tokens (in USD)
MODEL_PRICING = {
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50}
}

# Initialize session state for cost tracking
if 'cost_tracker' not in st.session_state:
    st.session_state.cost_tracker = {
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_cost': 0.0,
        'api_calls': 0,
        'session_start': datetime.now()
    }

# --- Cost Tracking Functions ---
def estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 characters per token for English text."""
    return len(text) // 4

def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """Calculate cost based on token usage and model pricing."""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing['input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']
    
    return input_cost + output_cost

def update_cost_tracker(input_tokens: int, output_tokens: int, model_name: str):
    """Update the session cost tracker."""
    cost = calculate_cost(input_tokens, output_tokens, model_name)
    
    st.session_state.cost_tracker['total_input_tokens'] += input_tokens
    st.session_state.cost_tracker['total_output_tokens'] += output_tokens
    st.session_state.cost_tracker['total_cost'] += cost
    st.session_state.cost_tracker['api_calls'] += 1

def display_cost_metrics():
    """Display cost tracking metrics in sidebar."""
    tracker = st.session_state.cost_tracker
    
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
        
        if st.button("🔄 Reset Tracker", use_container_width=True):
            st.session_state.cost_tracker = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_cost': 0.0,
                'api_calls': 0,
                'session_start': datetime.now()
            }
            st.rerun()

# --- Helper Functions ---
def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with rate limiting and error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(7)  # Rate limiting
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                st.warning(f"API error, retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                st.error(f"Failed after {max_retries} attempts: {e}")
                return None

@st.cache_data(show_spinner=False)
def generate_test_set(context: str, num_questions: int, _model_name: str):
    """Generates a test set using a powerful LLM."""
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
    """Gets a response from a specified Gemini model."""
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
def evaluate_response_with_judge(question: str, ground_truth: str, model_response: str, _judge_model: str):
    """Uses a powerful LLM as a 'judge' to evaluate a response."""
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
        required_keys = ["factual_consistency", "helpfulness", "relevance", "completeness", "reasoning"]
        if not all(key in evaluation for key in required_keys):
            raise ValueError("Missing required keys in evaluation")
        
        return evaluation
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "factual_consistency": 0, "helpfulness": 0,
            "relevance": 0, "completeness": 0,
            "reasoning": f"Parse error: {str(e)}"
        }

def create_score_visualization(df_results):
    """Creates a radar chart of average scores."""
    metrics = ['factual_consistency', 'helpfulness', 'relevance', 'completeness']
    avg_scores = [df_results[metric].mean() for metric in metrics]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_scores,
        theta=['Factual Consistency', 'Helpfulness', 'Relevance', 'Completeness'],
        fill='toself',
        name='Average Scores'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        height=400
    )
    return fig

# --- Streamlit App ---
st.title("⚖️ LLM Evaluation System")
st.markdown("*Advanced AI-powered evaluation framework with automated judging*")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
st.sidebar.subheader("1. Select Model to Test")
selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    options=list(AVAILABLE_MODELS.keys()),
    index=0
)
model_to_test = AVAILABLE_MODELS[selected_model_name]

# Show pricing info
with st.sidebar.expander("💵 Pricing Information"):
    pricing = MODEL_PRICING.get(model_to_test, {"input": 0, "output": 0})
    st.write(f"**{selected_model_name}**")
    st.write(f"• Input: ${pricing['input']:.3f} per 1M tokens")
    st.write(f"• Output: ${pricing['output']:.3f} per 1M tokens")

# Context input
st.sidebar.subheader("2. Provide Context")
context_input = st.sidebar.text_area(
    "Context for evaluation",
    height=200,
    placeholder="Paste documentation, article, or any text...",
    help="This context will be used to generate questions and evaluate responses"
)

# Number of questions
num_questions_input = st.sidebar.slider(
    "3. Number of Questions",
    min_value=3,
    max_value=20,
    value=5,
    help="More questions = better evaluation but slower"
)

# Estimate cost for generation
if context_input:
    est_tokens = estimate_tokens(context_input) * num_questions_input * 2
    est_cost = calculate_cost(est_tokens, est_tokens, JUDGE_MODEL)
    st.sidebar.info(f"💡 Estimated generation cost: ~${est_cost:.4f}")

# Generate button
if st.sidebar.button("🎯 Generate Test Set", type="primary", use_container_width=True):
    if context_input:
        with st.spinner("🤖 Generating evaluation questions..."):
            st.session_state.eval_set = generate_test_set(
                context_input, 
                num_questions_input,
                JUDGE_MODEL
            )
            # Clear previous results
            if 'eval_results' in st.session_state:
                del st.session_state.eval_results
        
        if st.session_state.get('eval_set'):
            st.sidebar.success(f"✅ Generated {len(st.session_state.eval_set)} questions!")
    else:
        st.sidebar.warning("⚠️ Please provide context first")

# Display cost tracking
display_cost_metrics()

# Main content
if 'eval_set' in st.session_state and st.session_state.eval_set:
    st.header("📋 Generated Evaluation Set")
    df_eval = pd.DataFrame(st.session_state.eval_set)
    st.dataframe(df_eval, use_container_width=True, height=300)
    
    st.header("🧪 Run Evaluation")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        system_prompt_input = st.text_area(
            "System Prompt for Model Under Test",
            value="You are a helpful assistant. Answer questions accurately based on the provided context. Be concise and factual.",
            height=120,
            help="This prompt defines how the model should behave"
        )
    
    with col2:
        st.info(f"**Testing:** {selected_model_name}\n\n**Judge:** Gemini 2.5 Pro")
        
        # Cost estimation for evaluation
        if st.session_state.eval_set:
            num_q = len(st.session_state.eval_set)
            est_eval_tokens = estimate_tokens(context_input) * num_q * 4
            est_eval_cost = calculate_cost(est_eval_tokens, est_eval_tokens, model_to_test)
            st.metric("Est. Eval Cost", f"${est_eval_cost:.4f}")
    
    if st.button("🚀 Run Evaluation & Get AI Judgment", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Starting evaluation...")
        results_with_judgment = []
        eval_set = st.session_state.eval_set
        
        for i, item in enumerate(eval_set):
            question = item['question']
            ground_truth_answer = item['answer']
            
            # Get response from model under test
            progress_bar.progress(
                (i + 0.3) / len(eval_set), 
                text=f"Getting response {i+1}/{len(eval_set)}..."
            )
            model_response = get_model_response(
                context=context_input,
                question=question,
                system_prompt=system_prompt_input,
                model_name=model_to_test
            )
            
            # Get evaluation from judge
            progress_bar.progress(
                (i + 0.7) / len(eval_set),
                text=f"Evaluating response {i+1}/{len(eval_set)}..."
            )
            judgment = evaluate_response_with_judge(
                question, 
                ground_truth_answer, 
                model_response,
                JUDGE_MODEL
            )
            
            # Combine results
            item_result = {
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "model_response": model_response,
                "factual_consistency": judgment.get("factual_consistency", 0),
                "helpfulness": judgment.get("helpfulness", 0),
                "relevance": judgment.get("relevance", 0),
                "completeness": judgment.get("completeness", 0),
                "judge_reasoning": judgment.get("reasoning", "N/A")
            }
            results_with_judgment.append(item_result)
            
            progress_bar.progress(
                (i + 1) / len(eval_set),
                text=f"Completed {i+1}/{len(eval_set)}"
            )
        
        st.session_state.eval_results = results_with_judgment
        progress_bar.empty()
        st.success("✅ Evaluation complete!")

# Results section
if 'eval_results' in st.session_state:
    st.header("📊 Evaluation Results")
    df_results = pd.DataFrame(st.session_state.eval_results)
    
    # Convert scores to numeric
    score_cols = ['factual_consistency', 'helpfulness', 'relevance', 'completeness']
    for col in score_cols:
        df_results[col] = pd.to_numeric(df_results[col], errors='coerce')
    
    # Calculate metrics
    avg_consistency = df_results['factual_consistency'].mean()
    avg_helpfulness = df_results['helpfulness'].mean()
    avg_relevance = df_results['relevance'].mean()
    avg_completeness = df_results['completeness'].mean()
    overall_score = (avg_consistency + avg_helpfulness + avg_relevance + avg_completeness) / 4
    
    # Display metrics with cost
    st.subheader("🎯 Overall Performance")
    cols = st.columns(6)
    cols[0].metric("Overall Score", f"{overall_score:.2f}/5", help="Average of all metrics")
    cols[1].metric("Factual Consistency", f"{avg_consistency:.2f}/5")
    cols[2].metric("Helpfulness", f"{avg_helpfulness:.2f}/5")
    cols[3].metric("Relevance", f"{avg_relevance:.2f}/5")
    cols[4].metric("Completeness", f"{avg_completeness:.2f}/5")
    cols[5].metric("Total Cost", f"${st.session_state.cost_tracker['total_cost']:.4f}", 
                   help="Session total cost")
    
    # Visualization
    st.subheader("📈 Score Distribution")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_radar = create_score_visualization(df_results)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Bar chart of individual scores
        fig_bar = px.bar(
            df_results.reset_index(),
            x='index',
            y=score_cols,
            title="Scores per Question",
            labels={'index': 'Question #', 'value': 'Score', 'variable': 'Metric'},
            barmode='group'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed results
    st.subheader("📝 Detailed Results")
    st.dataframe(
        df_results,
        use_container_width=True,
        height=400,
        column_config={
            "factual_consistency": st.column_config.NumberColumn(
                "Consistency",
                format="%.1f ⭐"
            ),
            "helpfulness": st.column_config.NumberColumn(
                "Helpfulness",
                format="%.1f ⭐"
            ),
            "relevance": st.column_config.NumberColumn(
                "Relevance",
                format="%.1f ⭐"
            ),
            "completeness": st.column_config.NumberColumn(
                "Completeness",
                format="%.1f ⭐"
            )
        }
    )
    
    # Export functionality
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    # Add cost data to export
    export_data = df_results.copy()
    export_data['session_cost'] = st.session_state.cost_tracker['total_cost']
    export_data['total_api_calls'] = st.session_state.cost_tracker['api_calls']
    export_data['model_tested'] = selected_model_name
    export_data['timestamp'] = datetime.now().isoformat()
    
    with col1:
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"llm_evaluation_{selected_model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        json_str = export_data.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"llm_evaluation_{selected_model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>⚖️ LLM Evaluation System | Powered by Google Gemini 2.5 Pro</p>
    <p>💰 Session Cost: ${st.session_state.cost_tracker['total_cost']:.4f} | API Calls: {st.session_state.cost_tracker['api_calls']}</p>
    </div>
    """,
    unsafe_allow_html=True
)