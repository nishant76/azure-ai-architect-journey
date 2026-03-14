from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

#Use same client setup that worked in Week 1

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

#--- Generate embeddings -----------
def get_embedding(text: str) -> list[float]:
    """Convert text to vector using Azure OpenAI embeddings"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )

    return response.data[0].embedding

#--- Cosine similarity -----------------
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Measure similarity between two vectors.
    1.0 = identical, 0.0 = completely different
    This is how vector search works under the hood.
    """
    dot_product = sum(a * b for a,b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)

# --- Test: semantic similarity -------------
print("==== Generating embeddings ===")

texts = [
    "LangChain is a framework for building LLM applications",
    "LangChain helps developers create AI powered apps",
    "Python is a programming language",
    "Azure is Microsoft's cloud platform",
    "RAG combines retrieval with language model generation",
]

print("Generating embeddings for 5 texts...")
embeddings = [get_embedding(text) for text in texts]
print(f"Each embedding has {len(embeddings[0])} dimensions")

# --- Compare similarities--------
print("\n === Similarity scores ===")
query = "How do I build apps with LangChain?"
print(f"Query: '{query}")
query_embedding = get_embedding(query)

similarities = []
for i, (text, emb) in enumerate(zip(texts, embeddings)):
    score = cosine_similarity(query_embedding, emb)
    similarities.append((score, text))
    print(f"Score {score:.4f}: {text[:60]}")

#---- Find most similar -----------
print("\n ==== Most relevant result ======")
best = max(similarities, key= lambda x: x[0])
print(f"Best match (score: {best[0]:.4f}):")
print(f"'{best[1]}'")


print("""
Key insight:
- Texts about LangChain scored HIGH for the LangChain query
- Python and Azure scored LOW — different topic
- This is semantic search — finds meaning, not just keywords
- This is the core of how RAG retrieval works
""")