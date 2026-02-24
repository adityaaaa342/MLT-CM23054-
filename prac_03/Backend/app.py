"""
FastAPI backend server.
Handles requests from frontend and returns sentiment analysis results.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import analyze_sentiment

# Create FastAPI app instance
app = FastAPI(title="Sentiment Analysis API")

# Enable CORS (Cross-Origin Resource Sharing)
# This allows frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Define request structure using Pydantic
class TextRequest(BaseModel):
    text: str

# Define response structure
class SentimentResponse(BaseModel):
    sentiment: str
    input_text: str

@app.get("/")
def home():
    """Root endpoint - shows API is working"""
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/analyze", response_model=SentimentResponse)
def analyze_text(request: TextRequest):
    """
    Main endpoint for sentiment analysis
    
    Workflow:
    1. Frontend sends text to this endpoint
    2. We process text using AI model
    3. Return sentiment result to frontend
    
    Example request:
    {
        "text": "I love programming!"
    }
    
    Example response:
    {
        "sentiment": "Positive",
        "input_text": "I love programming!"
    }
    """
    # Get sentiment from model
    sentiment = analyze_sentiment(request.text)
    
    # Return response
    return SentimentResponse(
        sentiment=sentiment,
        input_text=request.text
    )

# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)