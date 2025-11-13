from pydantic import BaseModel, Field

# Input
class Tweet(BaseModel):
    text: str = Field(..., min_length=1, max_length=128)
    
# Output
class PredictedResult(BaseModel):
    text: str
    sentiment: str
    probability: float
    confidence: float

class UserFeedback(BaseModel):
    text: str
    predicted_sentiment: str
    correct_sentiment: str
    confidence: float