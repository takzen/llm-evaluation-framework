"""
LLM Evaluation System - Main Application
Advanced AI-powered evaluation framework with automated judging and historical tracking.
"""

import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from datetime import datetime

# Import custom modules
from modules.config import AVAILABLE_MODELS, JUDGE_MODEL, MODEL_PRICING
from modules.cost_tracker import (
    init_cost_tracker, display_cost_metrics, 
    calculate_cost, get_cost_tracker
)
from modules.database import (
    init_database, save_evaluation, get_evaluation_history,
    get_evaluation_details, delete_evaluation, get_trend_data
)
from modules.evaluation import (
    generate_test_set, get_model_response, evaluate_response_with_judge
)
from modules.visualization import create_score_visualization
from utils.token_estimator import estimate_tokens

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Evaluation System", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="⚖️"
)

# --- Initialize ---
load_dotenv()

# Configure Gemini API
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

# Initialize database and cost tracker
init_database()
init_cost_tracker()

# --- Main Application ---
st.title("⚖️ LLM Evaluation System")
st.markdown("*Advanced AI-powered evaluation framework with automated judging*")

# Create tabs for main sections
tab1, tab2 = st.tabs(["🧪 New Evaluation", "📊 History & Analytics"])

# ===== TAB 1: NEW EVALUATION =====
with tab1:
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
    if st.sidebar.button("🎯 Generate Test Set", type="primary", width='stretch'):
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

    # Main content - Test Set Display
    if 'eval_set' in st.session_state and st.session_state.eval_set:
        st.header("📋 Generated Evaluation Set")
        df_eval = pd.DataFrame(st.session_state.eval_set)
        st.dataframe(df_eval, width='stretch', height=300)
        
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
        
        if st.button("🚀 Run Evaluation & Get AI Judgment", type="primary", width='stretch'):
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
        cols[5].metric("Total Cost", f"${get_cost_tracker()['total_cost']:.4f}", 
                       help="Session total cost")
        
        # Save to history button
        if st.button("💾 Save to History", type="secondary"):
            try:
                eval_id = save_evaluation(
                    selected_model_name,
                    system_prompt_input,
                    df_results,
                    context_input,
                    get_cost_tracker()
                )
                st.success(f"✅ Evaluation saved to history! (ID: {eval_id})")
            except Exception as e:
                st.error(f"Failed to save: {e}")
        
        # Visualization
        st.subheader("📈 Score Distribution")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_radar = create_score_visualization(df_results)
            st.plotly_chart(fig_radar, width='stretch')
        
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
            st.plotly_chart(fig_bar, width='stretch')
        
        # Detailed results
        st.subheader("📝 Detailed Results")
        st.dataframe(
            df_results,
            width='stretch',
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
        export_data['session_cost'] = get_cost_tracker()['total_cost']
        export_data['total_api_calls'] = get_cost_tracker()['api_calls']
        export_data['model_tested'] = selected_model_name
        export_data['timestamp'] = datetime.now().isoformat()
        
        with col1:
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"llm_evaluation_{selected_model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col2:
            json_str = export_data.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"llm_evaluation_{selected_model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width='stretch'
            )

# ===== TAB 2: HISTORY & ANALYTICS =====
with tab2:
    st.header("📊 Evaluation History & Analytics")
    
    # Get history
    history_df = get_evaluation_history()
    
    if history_df.empty:
        st.info("📭 No evaluations in history yet. Run an evaluation and save it to see it here!")
    else:
        # Summary statistics
        st.subheader("📈 Historical Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Evaluations", len(history_df))
        with col2:
            st.metric("Avg Overall Score", f"{history_df['overall_score'].mean():.2f}/5")
        with col3:
            st.metric("Total Cost", f"${history_df['total_cost'].sum():.2f}")
        with col4:
            unique_models = history_df['model_name'].nunique()
            st.metric("Models Tested", unique_models)
        
        # Trend visualization
        st.subheader("📊 Performance Trends Over Time")
        trend_data = get_trend_data()
        
        if not trend_data.empty:
            fig_trend = px.line(
                trend_data,
                x='timestamp',
                y='overall_score',
                color='model_name',
                title="Overall Score Trend by Model",
                labels={'overall_score': 'Overall Score', 'timestamp': 'Date'},
                markers=True
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, width='stretch')
            
            # Cost trend
            fig_cost = px.bar(
                trend_data,
                x='timestamp',
                y='total_cost',
                color='model_name',
                title="Cost per Evaluation",
                labels={'total_cost': 'Cost ($)', 'timestamp': 'Date'}
            )
            fig_cost.update_layout(height=400)
            st.plotly_chart(fig_cost, width='stretch')
        
        # History table
        st.subheader("📋 Evaluation History")
        
        # Format timestamp for display
        history_display = history_df.copy()
        history_display['timestamp'] = pd.to_datetime(history_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Display with selection
        selected_row = st.dataframe(
            history_display,
            width='stretch',
            height=400,
            column_config={
                "overall_score": st.column_config.NumberColumn(
                    "Overall Score",
                    format="%.2f ⭐"
                ),
                "avg_consistency": st.column_config.NumberColumn(
                    "Consistency",
                    format="%.2f"
                ),
                "avg_helpfulness": st.column_config.NumberColumn(
                    "Helpfulness",
                    format="%.2f"
                ),
                "avg_relevance": st.column_config.NumberColumn(
                    "Relevance",
                    format="%.2f"
                ),
                "avg_completeness": st.column_config.NumberColumn(
                    "Completeness",
                    format="%.2f"
                ),
                "total_cost": st.column_config.NumberColumn(
                    "Cost",
                    format="$%.4f"
                )
            },
            hide_index=True
        )
        
        # View details and delete
        st.subheader("🔍 View Evaluation Details")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            eval_id = st.selectbox(
                "Select Evaluation ID",
                options=history_df['id'].tolist(),
                format_func=lambda x: f"ID {x} - {history_df[history_df['id']==x]['model_name'].values[0]} ({history_df[history_df['id']==x]['timestamp'].values[0][:10]})"
            )
        
        with col2:
            if st.button("🗑️ Delete", type="secondary", width='stretch'):
                try:
                    delete_evaluation(eval_id)
                    st.success(f"✅ Evaluation {eval_id} deleted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
        
        if eval_id:
            eval_info, eval_details = get_evaluation_details(eval_id)
            
            if not eval_info.empty:
                st.markdown("---")
                st.markdown(f"### Evaluation Details - ID {eval_id}")
                
                # Show main info
                info_cols = st.columns(4)
                info_cols[0].metric("Model", eval_info['model_name'].values[0])
                info_cols[1].metric("Questions", eval_info['num_questions'].values[0])
                info_cols[2].metric("Overall Score", f"{eval_info['overall_score'].values[0]:.2f}/5")
                info_cols[3].metric("Cost", f"${eval_info['total_cost'].values[0]:.4f}")
                
                # Show system prompt
                with st.expander("📝 System Prompt Used"):
                    st.code(eval_info['system_prompt'].values[0])
                
                # Show context preview
                with st.expander("📄 Context Preview"):
                    st.write(eval_info['context_preview'].values[0])
                
                # Show detailed results
                st.markdown("#### Detailed Question Results")
                st.dataframe(
                    eval_details[['question', 'model_response', 'factual_consistency', 
                                 'helpfulness', 'relevance', 'completeness', 'judge_reasoning']],
                    width='stretch',
                    height=400
                )

# Footer
st.markdown("---")
tracker = get_cost_tracker()
st.markdown(
    f"""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>⚖️ LLM Evaluation System | Powered by Google Gemini 2.5 Pro</p>
    <p>💰 Session Cost: ${tracker['total_cost']:.4f} | API Calls: {tracker['api_calls']}</p>
    </div>
    """,
    unsafe_allow_html=True
)