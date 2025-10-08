import time
import streamlit as st
from modules.config import API_RATE_LIMIT, MAX_RETRIES


def safe_api_call(func, *args, **kwargs):
    """
    Wrapper for API calls with rate limiting and error handling.
    
    Args:
        func: Function to call (should return a result)
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result from the function or None if all retries failed
    """
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(API_RATE_LIMIT)  # Rate limiting
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 3  # Progressive backoff: 3s, 6s, 9s
                # Use info instead of warning for first retry
                if attempt == 0:
                    st.info(
                        f"⏳ API rate limit - waiting {wait_time}s before retry... "
                        f"(Attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                else:
                    st.warning(
                        f"⚠️ API error, retrying in {wait_time}s... "
                        f"(Attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                time.sleep(wait_time)
            else:
                st.error(f"❌ Failed after {MAX_RETRIES} attempts: {e}")
                return None