from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile,
    SearchField as VectorField
)
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio
import uuid
import uvicorn
import logging
import time

#---Setup--------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#---Clients----------------------
openai_client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
index_name = "document-qa"

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_key)
)

print(f"Endpoint: {search_endpoint}")
print(f"Index name: {index_name}")
# ---- Pydantic models ----------------
class IngestRequest(BaseModel):
    content: str = Field(min_length=50, description="Document content to ingest")
    source: Optional[str] = Field(default=None, description="Source label for filtering")

class IngestResponse(BaseModel):
    chunks_created: int
    source: str
    message: str

class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Question to ask the system")
    source_filter: Optional[str] = Field(default=None, description="Filter by source")
    top_k: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)

class AskResponse(BaseModel):
    question: str
    retrieved_chunks: int
    answer: str
    source_filter: Optional[str]

class BatchAskRequest(BaseModel):
    questions: list[str] = Field(min_items=1, max_length=10)
    source_filter: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    index: str

#--- Helpers----------------------
async def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )

    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                chunks.append(para[start: start + chunk_size])
                start += chunk_size - overlap
    return chunks
    
async def hybrid_search(
        query: str,
        top_k: int = 3,
        source_filter: str = None
) -> list[str]:
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

    results = search_client.search(**search_params)
    return [r["content"] for r in results]

async def generate_answer(question: str, chunks: list[str], temperature: float = 0.3) -> str:
    if not chunks:
        return "No relevant information found to answer the question."
    
    context = "\n\n".join(chunks)

    response = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": """You are a helpful AI assistant.
Answer based ONLY on the provided context.
If the answer is not in context, say: 'I don't have enough information to answer this.'
Be concise and precise."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

# -- App startup - Create index --------------------------
def setup_index():
    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_key)
    )

    try:
        index_client.get_index(index_name)
        logger.info(f"Index '{index_name}' already exists.")
    except:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String,
                       filterable=True, retrievable=True),
            VectorField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="vector-profile"
            )
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")]
        )

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

        index_client.create_or_update_index(index)

        logger.info(f"Index '{index_name}' created.")

# --- FastAPI App ----------------
app = FastAPI(
    title="Document Q&A API", 
    version="2.0.0",
    description="Week 2 Integration - RAG powered by Azure OpenAI + Azure AI Search"
)

@app.on_event("startup")
async def startup():
    setup_index()
    logger.info("App startup complete.")

# ---Routes ----------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model=deployment,
        index=index_name
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    logger.info(f"Ingesting document from source: {request.source}")

    #Chunk
    chunks = chunk_text(request.content)
    logger.info(f"Created {len(chunks)} chunks from document.")

    #Embed concurently
    embeddings = await asyncio.gather(
        *[get_embedding(chunk) for chunk in chunks]
    )

    #Build documents
    documents = [
        {
            "id": str(uuid.uuid4()),
            "content": chunk,
            "source": request.source,
            "content_vector": emb
        }
        for chunk, emb in zip(chunks, embeddings)
    ]

    #Upload to Azure AI Search
    search_client.upload_documents(documents)
    logger.info(f"Uploaded {len(documents)} chunks to index.")

    #Wait for indexing
    await asyncio.sleep(1)

    return IngestResponse(
        chunks_created=len(chunks),
        source=request.source,
        message=f"Document ingested with {len(chunks)} chunks."
    )

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    logger.info(f"Question: {request.question}")

    chunks = await hybrid_search(
        request.question,
        top_k=request.top_k,
        source_filter=request.source_filter
    )

    answer = await generate_answer(
        request.question,
        chunks,
        temperature=request.temperature
    )

    return AskResponse(
        question=request.question,
        retrieved_chunks=len(chunks),
        answer=answer,
        source_filter=request.source_filter
    )

@app.post("/ask-batch")
async def ask_batch(request: BatchAskRequest):
    logger.info(f"Batch questions: {len(request.questions)}")
    start = time.time()

    async def single_rag(question: str) -> dict:
        chunks = await hybrid_search(
            question,
            source_filter=request.source_filter
        )
        answer = await generate_answer(
            question,
            chunks
        )
        return { "question": question, "answer": answer }
    
    results = await asyncio.gather(
        *[single_rag(q) for q in request.questions]
    )

    return {
        "results": list(results),
        "total_question": len(request.questions),
        "time_taken": f"{time.time() - start:.2f} seconds"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
