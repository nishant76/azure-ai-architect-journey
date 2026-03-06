from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import os
import asyncio
import uvicorn

#-Setup----------
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s- %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

#---AZure OpenAI client ----
client = AsyncOpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

#--- Pydantic models ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=5, description="Question to ask AI")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=500, ge=1, le=4000)
    system_prompt: Optional[str] = "You are helpful Azure AI assistant."

class AskResponse(BaseModel):
    question: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class HealthResponse(BaseModel):
    status: str
    model: str

#--FASTAPI App =---------------
app = FastAPI(
    title="Azure AI Assistant",
    description="Week 1 integration project FastAPI + OpenAI",
    version="1.0.0"
)

#Routes --------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model=DEPLOYMENT)

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    logger.info(f"Question: {request.question}")

    try:
        response = await client.chat.completions.create(
            model=DEPLOYMENT, 
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content":  request.question}
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        logger.info(f"Tokens used: {response.usage.total_tokens}")

        return AskResponse(
            question=request.question,
            answer=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
    
    except Exception as e:
        logger.error(f"Azure OpenAI error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask-multi")
async def ask_multi(questions: list[str]):
    """Ask multiple questions concurrently - uses async from day 3 """
    logger.info(f"Processing {len(questions)} questions concurrently")

    async def single_ask(q: str) -> dict:
        response = await client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                { "role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": q}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return {
            "question": q,
            "answer": response.choices[0].message.content
        }
    
    #asyncio.gather - concurrent calls, like day 3!
    results = await asyncio.gather(*[single_ask(q) for q in questions])
    return {"results": results, "total_questions": len(questions)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
