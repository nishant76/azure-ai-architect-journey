import faiss 
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

# --- Embedding helper -------
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text 
    )
    return response.data[0].embedding

#---Sample knowledge base -----------
# simulating chunks from a real document
documents = [
    "LangChain is a framework for building applications powered by language models.",
    "LangGraph extends LangChain with stateful multi-agent workflows using graphs.",
    "RAG stands for Retrieval Augmented Generation — it grounds LLM answers in real data.",
    "Azure OpenAI Service provides access to GPT-4o and other OpenAI models on Azure.",
    "FastAPI is a modern Python web framework for building REST APIs quickly.",
    "Pydantic provides data validation using Python type hints.",
    "Vector databases store embeddings and enable semantic search.",
    "Azure AI Search supports vector, keyword and hybrid search modes.",
    "Chunking is the process of splitting documents into smaller pieces for RAG.",
    "Cosine similarity measures the angle between two vectors to find similar content.",
    "Multi-agent systems use multiple AI agents working together to solve complex tasks.",
    "Azure Container Apps is a serverless platform for deploying containerized apps.",
]

print("=== Building FAISS index ====")
print(f"Generating embeddings for {len(documents)} documents.....")

#Generate embeddings for all documents
embeddings = [get_embedding(doc) for doc in documents]
embeddings_array = np.array(embeddings, dtype=np.float32)

print(f"Embedding dimension: {embeddings_array.shape[1]}")
print(f"Total vectors: {embeddings_array.shape[0]}")

#--- Build FAISS Index ----------
dimension = embeddings_array.shape[1]

#IndexFlatL2 = exact search using L2 distance
#For production use IndexIVFlat for faster approximate search
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

print(f"FAISS index built with {index.ntotal} vectors")

#---Search function----------------------
def search(query: str, top_k: int = 3)-> list[dict]:
    """Search for most relevant documents given a query"""
    query_embedding = get_embedding(query)
    query_array = np.array([query_embedding], dtype=np.float32)

    #Search FAISS index
    distances, indices = index.search(query_array, top_k)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            "rank": i + 1,
            "document": documents[idx],
            "distance": float(dist),
            "index": int(idx)
        })
    return results

#--- Test searches ------------
queries = [
    "How do I build multi-agent AI systems?",
    "What is vector search and how does it work?",
    "How do I deploy Python apps on Azure?",
]

for query in queries:
    print(f"\n=== Query: '{query}' ====")
    results = search(query, top_k=3)
    for r in results:
        print(f"Rank {r['rank']} (distance: {r['distance']:.4f}): {r['document']}")


#--- Save and load index ------------
print("\n === Saving FAISS index ====")
faiss.write_index(index, "knowledge_base.index")

#Save documents mapping
with open("documents.json", "w") as f:
    json.dump(documents, f)

print("Index saved to knowledge_base.index")
print("Documents saved to documents.json")

# Reload and verify
print("\n==== Loading FAISS index =====")
loaded_index = faiss.read_index("knowledge_base.index")
with open("documents.json") as f:
    loaded_docs = json.load(f)


print(f"Loaded index with {loaded_index.ntotal} vectors")

#Test loaded index
results = search("What is RAG?", top_k=2)
print("\nTest search on loaded index:")
for r in results:
    print(f"Rank {r['rank']}: {r['document']}")