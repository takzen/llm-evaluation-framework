"""
Data visualization functions for evaluation results.
"""

import plotly.graph_objects as go
import pandas as pd


def create_score_visualization(df_results: pd.DataFrame) -> go.Figure:
    """
    Creates a radar chart of average scores.
    
    Args:
        df_results: DataFrame containing evaluation results with score columns
        
    Returns:
        Plotly Figure object with radar chart
    """
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