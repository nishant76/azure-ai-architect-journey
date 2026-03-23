from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import asyncio
import time
import uvicorn
import logging

# Setup ----------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

# ── Clients ──────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-10-21",
    temperature=0.3
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment="text-embedding-ada-002",
    api_version="2024-10-21"
)

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = "langchain-week3"

# Vector store --------------
vectorstore = AzureSearch(
    azure_search_endpoint=search_endpoint,
    azure_search_key=search_key, 
    index_name=index_name,
    embedding_function=embeddings.embed_query
)

# Text splitter ---------------
spliter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=50
)

# RAG Prompt --------------------
rag_prompt = ChatPromptTemplate([
    ("system", """You are a helpful AI assistant.
Answer based ONLY on the provided context.
If answer not in context say: 'I don't have enough information.'"""),
    ("human", """Context:
{context}

Question: {question}""")
])

# Agent tools ------------------
@tool 
def get_tech_info(technology: str) -> str:
    """Get information about a technology or framework."""
    text_db = {
        "langchain": "LangChain is a framework for building LLM applications.",
        "langgraph": "LangGraph extends LangChain with stateful multi-agent workflows.",
        "fastapi": "FastAPI is a modern async Python framework for REST APIs.",
        "azure openai": "Azure OpenAI provides GPT-4o and embedding models on Azure.",
        "rag": "RAG combines retrieval with generation to ground LLM answers in real data.",
    }

    return text_db.get(technology.lower(), f"No info found for {technology}")

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found."
    return "\n\n".join(doc.page_content for doc in docs)

# Agent ----------
agent = create_react_agent(
    model=llm,
    tools=[get_tech_info, calculate, search_knowledge_base],
    prompt=SystemMessage(content="You are a helpful AI assistant. Use the available tools to answer questions accurately. Always use tools when needed.")


)

# Pydantic models -----------
class IngestRequest(BaseModel):
    content: str = Field(min_length=50)
    source: str = Field(default ="manual")

class IngestResponse(BaseModel):
    chunks_created: int
    source: str
    message: str
    

class AskRequest(BaseModel):
    question: str = Field(min_length=5)
    stream: bool = Field(default=False)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class AgentRequest(BaseModel):
    question: str = Field(min_length=5)

class HealthResponse(BaseModel):
    status: str
    model: str
    index: str

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="Week 3 AI API",
    description="LangChain + LangGraph + Azure AI Search + Streaming",
    version="3.0.0"
)

# ── Routes ────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        index=index_name
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    logger.info(f"Ingesting from source: {request.source}")

    #Split into chunks
    chunks = spliter.split_text(request.content)

    # Create docs
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": request.source}
        )
        for chunk in chunks
    ]

    # Add to Azure AI Search
    vectorstore.add_documents(docs)
    await asyncio.sleep(1) # Wait for indexing

    logger.info(f"Ingested {len(chunks)} chunks")

    return IngestResponse(
        chunks_created=len(chunks),
        source=request.source,
        message=f"Successfully ingested {len(chunks)} chunks from '{request.source}'"
    )

@app.post("/ask")
async def ask(request: AskRequest):
    logger.info(f"Question: {request.question}")

    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        request.question, k=3
    )

    relevant_docs = [
        doc for doc, score in docs_with_scores
        if score >= request.threshold
    ]

    if not relevant_docs:
        return {"answer": "I don't have enough information to answer this question."}

    # ── Add this block here ───────────────────────────────
    # Deduplicate
    seen = set()
    unique_docs = []
    for doc in relevant_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    context = "\n\n".join(doc.page_content for doc in unique_docs)
    print(f"Unique docs: {len(unique_docs)}")
    print(f"Context: {context}")

    # Test LLM directly
    from langchain_core.messages import SystemMessage, HumanMessage
    direct_response = llm.invoke([
        SystemMessage(content="You are a helpful AI assistant. Answer based ONLY on the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {request.question}")
    ])
    print(f"Direct LLM response: {direct_response.content}")
    # ── End debug block ───────────────────────────────────

    if request.stream:
        async def generate():
            async for chunk in llm.astream([
                SystemMessage(content="You are a helpful AI assistant. Answer based ONLY on context."),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {request.question}")
            ]):
                yield chunk.content
        return StreamingResponse(generate(), media_type="text/plain")

    # Replace with this — direct LLM call that we know works
    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([
        SystemMessage(content="You are a helpful AI assistant. Answer based ONLY on the provided context. If answer not in context say: 'I don't have enough information.'"),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {request.question}")
    ])
    answer = response.content

    return {
        "question": request.question,
        "answer": answer,
        "chunks_used": len(unique_docs)
    }

@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    logger.info(f"Agent question: {request.question}")
    from langchain_core.messages import HumanMessage

    result = agent.invoke({
        "messages": [HumanMessage(content=request.question)]
    })

    # Debug — print all messages
    print(f"Total messages: {len(result['messages'])}")
    for msg in result['messages']:
        print(f"  {type(msg).__name__}: {msg.content[:50]}")

    return {
        "question": request.question,
        "answer": result["messages"][-1].content
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)