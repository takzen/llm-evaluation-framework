"""
Token estimation utilities.
"""

def estimate_tokens(text: str) -> int:
    """
    Rough estimation: ~4 characters per token for English text.
    
    Args:
        text: Input text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0
    return len(text) // 4