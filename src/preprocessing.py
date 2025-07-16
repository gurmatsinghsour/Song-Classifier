import re
import pandas as pd


def preprocess_lyrics(text):
    """
    Advanced preprocessing specifically designed for song lyrics
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove repetitive words (common in songs like "yeah yeah yeah")
    text = re.sub(r'\b(\w+)\s+\1\s+\1+\b', r'\1', text)
    
    # Remove common song artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove [Chorus], [Verse] etc.
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical expressions
    text = re.sub(r'embed$', '', text)  # Remove "embed" at end
    text = re.sub(r'\d+embed$', '', text)  # Remove numbers+embed at end
    
    # Keep apostrophes for contractions but remove other punctuation
    text = re.sub(r"[^\w\s']", ' ', text)
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short words and very long words (likely artifacts)
    words = text.split()
    words = [word for word in words if 2 <= len(word) <= 15]
    
    return ' '.join(words).strip()


def get_sentiment_label(score):
    """
    Convert sentiment score to label
    """
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
