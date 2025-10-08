"""
Database operations for storing and retrieving evaluation history.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from modules.config import DB_PATH


def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create evaluations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            system_prompt TEXT,
            num_questions INTEGER,
            overall_score REAL,
            avg_consistency REAL,
            avg_helpfulness REAL,
            avg_relevance REAL,
            avg_completeness REAL,
            total_cost REAL,
            api_calls INTEGER,
            context_preview TEXT
        )
    """)
    
    # Create evaluation_details table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id INTEGER,
            question TEXT,
            ground_truth TEXT,
            model_response TEXT,
            factual_consistency REAL,
            helpfulness REAL,
            relevance REAL,
            completeness REAL,
            judge_reasoning TEXT,
            FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
        )
    """)
    
    conn.commit()
    conn.close()


def save_evaluation(
    model_name: str, 
    system_prompt: str, 
    df_results: pd.DataFrame, 
    context: str, 
    cost_tracker: dict
) -> int:
    """
    Save evaluation results to database.
    
    Args:
        model_name: Name of the model that was evaluated
        system_prompt: System prompt used for the model
        df_results: DataFrame with evaluation results
        context: Original context text
        cost_tracker: Dictionary with cost tracking information
        
    Returns:
        ID of the saved evaluation
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Calculate metrics
    overall_score = (
        df_results['factual_consistency'].mean() +
        df_results['helpfulness'].mean() +
        df_results['relevance'].mean() +
        df_results['completeness'].mean()
    ) / 4
    
    context_preview = context[:200] + "..." if len(context) > 200 else context
    
    # Insert main evaluation record
    cursor.execute("""
        INSERT INTO evaluations (
            timestamp, model_name, system_prompt, num_questions,
            overall_score, avg_consistency, avg_helpfulness, 
            avg_relevance, avg_completeness, total_cost, api_calls, context_preview
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        model_name,
        system_prompt,
        len(df_results),
        overall_score,
        df_results['factual_consistency'].mean(),
        df_results['helpfulness'].mean(),
        df_results['relevance'].mean(),
        df_results['completeness'].mean(),
        cost_tracker['total_cost'],
        cost_tracker['api_calls'],
        context_preview
    ))
    
    evaluation_id = cursor.lastrowid
    
    # Insert detailed results
    for _, row in df_results.iterrows():
        cursor.execute("""
            INSERT INTO evaluation_details (
                evaluation_id, question, ground_truth, model_response,
                factual_consistency, helpfulness, relevance, completeness, judge_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation_id,
            row['question'],
            row['ground_truth_answer'],
            row['model_response'],
            row['factual_consistency'],
            row['helpfulness'],
            row['relevance'],
            row['completeness'],
            row['judge_reasoning']
        ))
    
    conn.commit()
    conn.close()
    
    return evaluation_id


def get_evaluation_history() -> pd.DataFrame:
    """
    Retrieve all evaluations from database.
    
    Returns:
        DataFrame with evaluation summary information
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT 
            id,
            timestamp,
            model_name,
            num_questions,
            overall_score,
            avg_consistency,
            avg_helpfulness,
            avg_relevance,
            avg_completeness,
            total_cost,
            context_preview
        FROM evaluations
        ORDER BY timestamp DESC
    """, conn)
    conn.close()
    return df


def get_evaluation_details(evaluation_id: int) -> tuple:
    """
    Retrieve detailed results for a specific evaluation.
    
    Args:
        evaluation_id: ID of the evaluation to retrieve
        
    Returns:
        Tuple of (eval_info DataFrame, details DataFrame)
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Get main evaluation info
    eval_info = pd.read_sql_query("""
        SELECT * FROM evaluations WHERE id = ?
    """, conn, params=(evaluation_id,))
    
    # Get detailed results
    details = pd.read_sql_query("""
        SELECT * FROM evaluation_details WHERE evaluation_id = ?
    """, conn, params=(evaluation_id,))
    
    conn.close()
    return eval_info, details


def delete_evaluation(evaluation_id: int):
    """
    Delete an evaluation and its details from database.
    
    Args:
        evaluation_id: ID of the evaluation to delete
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM evaluation_details WHERE evaluation_id = ?", (evaluation_id,))
    cursor.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))
    
    conn.commit()
    conn.close()


def get_trend_data() -> pd.DataFrame:
    """
    Get historical trend data for visualization.
    
    Returns:
        DataFrame with time-series evaluation data
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT 
            timestamp,
            model_name,
            overall_score,
            avg_consistency,
            avg_helpfulness,
            avg_relevance,
            avg_completeness,
            total_cost
        FROM evaluations
        ORDER BY timestamp ASC
    """, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df