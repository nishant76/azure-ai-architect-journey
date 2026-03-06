from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import logging

#----Logging Setup --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Learning API",
    description="Week 1 Day 4 - FastAPI Practice",
    version="1.0.0"
)

#---Pydantic models -------
class QuestionRequest(BaseModel):
    question: str = Field(min_length=5, description="Question to ask")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=500, ge=1, le=4000)

class QuestionResponse(BaseModel):
    question: str
    answer: str
    tokens_used: int

class HealthResponse(BaseModel):
    status: str
    version: str

#----Routes --------------------------

# GET- health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check called")
    return HealthResponse(status="healthy", version="1.0.0")

# Get with path parameter
@app.get("/greet/{name}")
async def greet(name: str):
    logger.info(f"Greeting {name}")
    return {"message": f"Hello {name}, welcome to Azure AI Learning!"}

# -- POST- maine endpoint
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    logger.info(f"Questions received: {request.question}")

    #simulate AI response for now
    #tomorrow we this to Azure OpenAI
    mock_answer = f"this is a mock answer for: {request.question}"

    return QuestionResponse(
        question=request.question,
        answer=mock_answer,
        tokens_used=100
    )

#POST --- with eerror handling
@app.post("/ask-strict")
async def ask_strict(request: QuestionRequest):
    if "test" in request.question.lower():
        raise HTTPException(
            status_code=400,
            detail="Questions containing 'test' are not allowed"
        )
    
    return {"answer": f"Processing: {request.question}"}

if __name__ == "__main__":
    uvicorn.run("day4_trackA:app", host="0.0.0.0", port=8000, reload=True)