from openai import OpenAI
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import re

load_dotenv()

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

#=== Step 1: Knowledge base --------
# Simulating a real document-- later replace with PDF
knowledge_base = """
Azure OpenAI Service provides REST API access to OpenAI's powerful language models 
including GPT-4o, GPT-4o-mini, and embedding models. These models can be used for 
content generation, summarization, and semantic search.

LangChain is a framework for developing applications powered by language models. 
It provides tools for chaining prompts, managing memory, and integrating with 
external data sources. LangChain supports Azure OpenAI as a model provider.

LangGraph extends LangChain by enabling stateful, multi-actor applications. 
It uses a graph-based approach where nodes represent processing steps and edges 
define the flow between them. LangGraph is ideal for complex multi-agent workflows.

RAG stands for Retrieval Augmented Generation. It combines the power of large 
language models with the ability to search and retrieve relevant information 
from a knowledge base. RAG helps reduce hallucinations by grounding answers in real data.

Azure AI Search is a cloud search service that supports full-text search, 
vector search, and hybrid search. It can store document embeddings and perform 
semantic similarity search at scale. Azure AI Search integrates natively with Azure OpenAI.

FastAPI is a modern Python web framework for building REST APIs. It uses Python 
type hints and Pydantic models for automatic validation and documentation. 
FastAPI supports async operations natively.

Multi-agent systems use multiple AI agents working together to solve complex tasks. 
Each agent has a specific role such as planning, research, or writing. 
Agents communicate through shared state or message passing.

Azure Container Apps is a serverless platform for deploying containerized 
applications. It supports automatic scaling, managed identity, and integrates 
with Azure Monitor for observability. It is ideal for deploying FastAPI and LangChain apps.
"""


#--STep 2 Chunking -----------
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            #split large paragraphs further
            start = 0
            while start < len(para):
                chunks.append(para[start: start+chunk_size])
                start += chunk_size - overlap
    return chunks

# Step 3: Embedding --------------------
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )

    return response.data[0].embedding

#---Step 4 --- Build index------------
def build_index(chunks: list[str]) -> tuple:
    print(f"Building index for {len(chunks)} chunks .....")
    embeddings = [get_embedding(chunk) for chunk in chunks]
    embeddings_array = np.array(embeddings, dtype=np.float32)

    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    print(f"Index built with {index.ntotal} vectors")
    return index, embeddings_array

# Step 5: Retrieve relevant chunks =========
def retrieve(query: str, index, chunks: list[str], top_k: int = 3) -> list[str]:
    query_embedding = get_embedding(query)
    query_array = np.array([query_embedding], dtype=np.float32)

    distances, indices = index.search(query_array, top_k)

    retrieved = []
    for dist, idx in zip(distances[0], indices[0]):
        retrieved.append({
            "chunk": chunks[idx],
            "distance": float(dist)
        })

    return retrieved

# Step 6: Generate Answer ------
def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    # Build context from relevant chunks
    context = "\n\n".join(r["chunk"] for r in retrieved_chunks)

    system_prompt = """You are a helpful AI assistant.
    Answer the question based ONLY on the provided context.
    If the answer is not in the context, say 'I don't have enough information to answer this.'
    Do not make up information."""

    user_prompt = f"""Context: 
    {context}
    Question: {question}
    Answer: """

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# Step 7: Full RAG pipeline --------
def rag_pipeline(question: str, index, chunks: list[str]) -> dict:
    print(f"\n{'='*50}")
    print(f"Question: {question}")

    #Retrieve
    retrieved = retrieve(question, index, chunks, top_k=3)
    print(f"\nTop {len(retrieved)} retrieved chunks:")
    for i, r in enumerate(retrieved):
        print(f"    Chunk {i+1} (distance: {r['distance']:.4f}): {r['chunk'][:80]}...")

    # Generate 
    answer = generate_answer(question, retrieved)
    print(f"\nAnswer: {answer}")

    return {
        "question": question,
        "retrieved_chunks": len(retrieved),
        "answer": answer
    }

#--Run it------------------
print("==== BUilding RAG pipeline===")
chunks = chunk_text(knowledge_base)
print(f"Created {len(chunks)} chunks")

index, _ = build_index(chunks)

#Test with different questions
questions = [
    "What is RAG and why is it helpful?",
    "How does LangGraph differ from LangChain?",
    "What Azure service should I use for deploying FastAPI apps?",
    "What is the capital of France?",  # out of scope — tests grounding
]

results =[]
for q in questions:
    result = rag_pipeline(q, index, chunks)
    results.append(result)


# --Architect observation -----
print(f"\n{'='*50}")
print("=== Architect observations ===")
print(f"Total chunks in index: {len(chunks)}")
print(f"Questions answered: {len(results)}")
print("""
Notice:
- Questions 1-3: answered from context (grounded)
- Question 4: 'I don't have enough information' (hallucination prevented)
- This is why RAG is powerful — it stays within known facts
- Lower temperature (0.3) = more factual, less creative answers
""")