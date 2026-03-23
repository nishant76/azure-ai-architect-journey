from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import asyncio
import time

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-10-21",
    temperature=0.7
)

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Be concise"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# --- Streaming -----------------
print(" === Streaming Response ===")
print("Response: ", end="", flush=True)

for chunk in chain.stream({"question": "Explaing RAG in 3 sentences."}):
    print(chunk, end="", flush=True)
print("\n")


# ── Streaming with timing ─────────────────────────────────
print("=== Streaming with Token Timing ===")
start = time.time()
token_count =0

for chunk in chain.stream({"question": "What is LangGraph in 2 sentences?"}):
    print(chunk, end="", flush=True)
    token_count += 1


print(f"\n\nTokens streamed: {token_count}")
print(f"Time: {time.time() - start:.2f}s\n")

# ── Async single call ─────────────────────────────────────
print("=== Async Single Call ===")

async def ask_async(question: str) -> str:
    return await chain.ainvoke({"question": question})

result = asyncio.run(ask_async("What is FastAPI in one sentence?"))
print(f"Async result: {result}\n")


# ── Async batch — multiple questions simultaneously ───────
print("=== Async Batch (Concurrent) ===")

async def ask_batch(questions: list[str]) -> list[str]:
    results = []
    tasks = [chain.ainvoke({"question": q}) for q in questions]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    for r in gathered:
        if isinstance(r, Exception):
            results.append(f"Error: {str(r)}")
        else:
            results.append(r)
    return results

async def ask_batch(questions: list[str]) -> list[str]:
    tasks = [chain.ainvoke({"question": q}) for q in questions]

questions = [
    "What is LangChain in one sentence?",
    "What is LangGraph in one sentence?",
    "What is Azure OpenAI in one sentence?",
    "What is RAG in one sentence?",
]

# Sequential timing
print("Sequential: ")
start = time.time()
sequential_results = [chain.invoke({"question": q}) for q in questions]
seq_time = time.time() - start
print(f"Time: {seq_time:.2f}s")

# Concurrent timing
print("\nConcurrent (async):")
start = time.time()
concurrent_results = chain.batch([{"question": q} for q in questions])
print(f"Got {len(concurrent_results)} results")  # debug line
con_time = time.time() - start
print(f"Time: {con_time:.2f}s")

print(f"\nSpeedup: {seq_time/con_time:.1f}x faster with async!")

# Print results
print("\nAnswers:")
for q, a in zip(questions, concurrent_results):
    print(f"Q: {q}")
    print(f"A: {a}\n")

# ── Async streaming ───────────────────────────────────────
print("=== Async Streaming ===")

async def stream_async(question: str):
    print(f"Streaming answer to: '{question}'")
    print("Response: ", end="", flush=True)
    async for chunk in chain.astream({"question": question}):
        print(chunk, end="", flush=True)
    print("\n")

asyncio.run(stream_async("What are the top 3 benefits of using LangChain?"))



print("""
=== Summary ===
stream()   → sync streaming, tokens print as they arrive
ainvoke()  → async single call
astream()  → async streaming
gather()   → run multiple async calls simultaneously

In FastAPI:
  Use ainvoke() for single requests
  Use astream() for streaming endpoints
  Use gather() for batch processing
""")