from openai import AsyncOpenAI
from config import (
    AZURE_OPENAI_DEPLOYMENT, 
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY
)

import logging

logger = logging.getLogger(__name__)

#Async client - use this with FastAPI
client = AsyncOpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY
)

async def ask_azure(
        question: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500
) -> dict:
    logger.info(f"Calling Azure OpenAI: {question[:50]}...")

    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            { "role": 'system', "content": system_prompt},
            { "role": 'user', "content": question}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return {
        "answer": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens
    }

#Test it directly
async def main():
    result = await ask_azure("What is FastAPI in one sentence?")
    print(f"Answer: {result['answer']}")
    print(f"Tokens: {result['tokens_used']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())