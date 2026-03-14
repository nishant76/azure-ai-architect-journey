from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile, 
    SearchField as VectorField
)
from azure.core.credentials import AzureKeyCredential
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# --- Clients  -------------------------
openai_client = AsyncOpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
index_name = "rag-knowledge-base"


# --- Embedding -------------------------
async def get_embedding(text: str) -> list[float]:
    response = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )

    return response.data[0].embedding

# --- Create Search index -------------------------
async def create_search_index():
    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_key)
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        VectorField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="vector_profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
        profiles=[VectorSearchProfile(name="vector_profile", algorithm_configuration_name="hnsw-config")]
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"Index '{index_name}' ready.")


# --- Index documents -------------------------
async def index_documents(docs: list[dict]):
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key)
    )

    print(f"Indexing {len(docs)} documents...")

    # Generate embeddings concurrently
    embeddings = await asyncio.gather(
        *[get_embedding(doc["content"]) for doc in docs]
    )

    # Add embeddings to documents
    for doc, emb in zip(docs, embeddings):
        doc["content_vector"] = emb


    search_client.upload_documents(docs)
    print("Documents indexed.")
    return search_client

#--- Hybrid search -------------------------
async def hybrid_search(client: SearchClient, query: str, top_k: int = 3, source_filter: str = None) -> list[str]:
    query_embedding = await get_embedding(query)

    search_params = {
        "search_text": query,
        "vector_queries": [{
            "kind": "vector",
            "vector": query_embedding,
            "fields": "content_vector",
            "k": top_k
        }],
        "top": top_k
    }

    if source_filter:
        search_params["filter"] = f"source eq '{source_filter}'"

    results = client.search(**search_params)
    return [r["content"] for r in results]

# --- Generate answer -------------------------
async def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)

    response = await openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": """You are a helpful AI assistant.
Answer based ONLY on the provided context.
If the answer is not in context say: 'I don't have enough information.'
Be concise and precise."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# -- Full RAG pipeline -------------------------
async def rag_pipeline(question: str, client: SearchClient, source_filter: str = None) -> dict:
    # Retrieve
    chunks = await hybrid_search(
        client, question,
        top_k=3,
        source_filter=source_filter,
    )

    # Generate 
    answer = await generate_answer(question, chunks)

    return {
        "question": question,
        "retrieved_chunks": len(chunks),
        "answer": answer
    }

# -- Main -------------------------
async def main():
    # Create index and index documents
    await create_search_index()

    # Sample documents from two sources
    documents = [
        {"id": "1", "content": "LangChain is a framework for building LLM applications.", "source": "langchain-docs"},
        {"id": "2", "content": "LangGraph enables stateful multi-agent workflows using graphs.", "source": "langchain-docs"},
        {"id": "3", "content": "RAG grounds LLM answers in real data to reduce hallucinations.", "source": "azure-docs"},
        {"id": "4", "content": "Azure OpenAI provides GPT-4o and embedding models on Azure.", "source": "azure-docs"},
        {"id": "5", "content": "Azure AI Search supports vector and hybrid search at scale.", "source": "azure-docs"},
        {"id": "6", "content": "FastAPI is a modern async Python framework for REST APIs.", "source": "python-docs"},
        {"id": "7", "content": "Multi-agent systems coordinate multiple AI agents for complex tasks.", "source": "langchain-docs"},
        {"id": "8", "content": "Azure Container Apps deploys containerized apps serverlessly.", "source": "azure-docs"},
    ]

    search_client = await index_documents(documents)

    # wait for indexing to complete
    import time
    time.sleep(2)

    # Test 1: General RAG
    print("\n--- Test 1: General RAG ---")

    result = await rag_pipeline("What is RAG and why is it useful?", search_client)
    print(f"Q: {result['question']}")
    print(f"Retrieved Chunks: {result['retrieved_chunks']}")
    print(f"Answer: {result['answer']}")

    #Test 2: Source-filtered RAG
    print("\n--- Test 2: Source-filtered RAG (azure-docs) ---")
    result = await rag_pipeline("What does Azure AI Search do?", search_client, source_filter="azure-docs")
    print(f"Q: {result['question']}")
    print(f"Retrieved Chunks: {result['retrieved_chunks']}")
    print(f"Answer: {result['answer']}")

    #Test 3: Out of scope
    print("\n--- Test 3: Out of scope question ---")
    result = await rag_pipeline("What is the capital of France?", search_client)
    print(f"Q: {result['question']}")
    print(f"Retrieved Chunks: {result['retrieved_chunks']}")
    print(f"Answer: {result['answer']}")

    #Test 3: Concurrent RAG queries
    print("\n--- Test 4: Concurrent RAG queries ---")
    questions = [
        "How do I use LangChain?",
        "What is Azure OpenAI?",
        "How do I deploy Python apps?"
    ]

    results = await asyncio.gather(
        *[rag_pipeline(q, search_client) for q in questions]
    )

    for r in results: 
        print(f"\nQ: {r['question']}")
        print(f"Retrieved Chunks: {r['retrieved_chunks']}")
        print(f"Answer: {r['answer']}")

asyncio.run(main())