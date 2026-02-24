"""
This file loads the pre-trained sentiment analysis model.
We use HuggingFace's transformers library for easy access to AI models.
"""

from transformers import pipeline

# Load a pre-trained sentiment analysis model
# This model is already trained on millions of text examples
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """
    Simple function to analyze text sentiment
    
    How it works:
    1. Model breaks text into smaller pieces (tokens)
    2. Compares with patterns learned during training
    3. Returns probability for positive/negative
    
    Args:
        text (str): User's input text
    
    Returns:
        str: "Positive" or "Negative"
    """
    # Get prediction from model
    result = sentiment_analyzer(text)[0]
    
    # Return simplified result
    return "Positive" if result['label'] == 'POSITIVE' else "Negative"

# Example (for testing)
if __name__ == "__main__":
    test_text = "I love this product! It's amazing."
    print(f"Text: {test_text}")
    print(f"Sentiment: {analyze_sentiment(test_text)}")