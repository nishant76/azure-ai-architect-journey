from openai import AsyncOpenAI
import faiss
import numpy as np
from dotenv import load_dotenv
import asyncio
import os
import time

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

#---Knowledge base -------------------------
documents = [
    "LangChain is a framework for building LLM applications with Azure OpenAI.",
    "LangGraph extends LangChain with stateful multi-agent graph workflows.",
    "RAG combines retrieval with generation to ground LLM answers in real data.",
    "Azure OpenAI provides GPT-4o and embedding models on Azure infrastructure.",
    "Azure AI Search supports vector, keyword and hybrid search at enterprise scale.",
    "FastAPI is a modern async Python framework for building production REST APIs.",
    "Multi-agent systems coordinate multiple AI agents to solve complex tasks.",
    "Azure Container Apps deploys containerized AI apps with automatic scaling.",
    "Pydantic provides data validation using Python type hints for API models.",
    "Chunking splits documents into smaller pieces for better RAG retrieval.",
]

#---Async embedding -----------------
async def get_embedding(text: str) -> list[float]:
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )

    return response.data[0].embedding

#--- Sync vs Async comparison ----------------
async def embed_sequential(texts: list[str]) -> list:
    """Embed texts one by one - slow"""
    embeddings = []
    for text in texts:
        emb = await get_embedding(text)
        embeddings.append(emb)
    return embeddings

async def embed_concurrent(texts: list[str]) -> list:
    """Embed all at once - fast"""
    embeddings = await asyncio.gather(
        *[get_embedding(text) for text in texts]
    )

    return list(embeddings)


#-- Build async FAISS index ----------------
async def build_index(docs: list[str]):
    print("Building index concurrently...")
    start = time.time()

    embeddings = await embed_concurrent(docs)
    embeddings_array = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    print(f"Index built in {time.time() - start:.2f}s with {index.ntotal} vectors")
    return index

# --- Async retrieve ---------------
async def retrieve(query: str, index, docs: list[str], top_k: int = 3) -> list[dict]:
    query_emb = await get_embedding(query)
    query_array = np.array([query_emb], dtype=np.float32)

    distances, indices = index.search(query_array, top_k)
    
    return [
        { "chunk": docs[idx], "distance": float(dist)}
        for dist, idx in zip(distances[0], indices[0])
    ]

# --- Async generate ----------------
async def generate(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(c["chunk"] for c in chunks)

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": """Answer based ONLY on the provided context.
If answer not in context say: 'I don't have enough information.'"""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# --- Full async RAG pipeline ---------------
async def rag(question: str, index, docs: list[str]) -> dict:
    chunks = await retrieve(question, index, docs)
    answer = await generate(question, chunks)
    return {
        "question": question,
        "answer": answer
    }

# -- Process multiple questions concurrently ---------------
async def batch_rag(questions: list[str], index, docs: list[str]) -> list[dict]:
    """Answer multiple questions simultaneously"""
    print(f"\nProcessing {len(questions)} questions concurrently...")

    start = time.time()

    results = await asyncio.gather(
        *[rag(q, index, docs) for q in questions]
    )
    print(f"All questions processed in {time.time() - start:.2f}s")

    return list(results)

# === MAIN ----------------
async def main():
    # Build index
    index = await build_index(documents)

    #Single question
    print("\n--- Single question RAG ---")
    result = await rag("What is RAG and why is it useful?", index, documents)

    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")

    #Batch questions
    print("\n--- Batch question RAG (Concurrent) ---")
    questions = [
        "What is LangGraph used for?",
        "How do I deploy AI apps on Azure?",
        "What is the difference between LangChain and LangGraph?",
    ]

    results = await batch_rag(questions, index, documents)
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer']}")

asyncio.run(main())