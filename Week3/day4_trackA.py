from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

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

# Azure AI Search Setup -----------
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = "langchain-week3"

print("=== Setting up Azure AI Search Vector Store ===")

#LangChain wraps Azure AI Search in one line!
vectorstore = AzureSearch(
    azure_search_endpoint=search_endpoint,
    azure_search_key=search_key,
    index_name=index_name,
    embedding_function=embeddings.embed_query
)

print(f"Connected to index: {index_name}")

# Add documents -----------
print("\n ========= Adding Documents =======")

documents = [
    Document(page_content="LangChain is a framework for building LLM applications with chains and agents.", metadata={"source": "langchain-docs", "topic": "frameworks"}),
    Document(page_content="LangGraph extends LangChain with stateful multi-agent graph workflows.", metadata={"source": "langchain-docs", "topic": "frameworks"}),
    Document(page_content="RAG combines retrieval with generation to ground LLM answers in real data.", metadata={"source": "ai-docs", "topic": "patterns"}),
    Document(page_content="Azure OpenAI provides GPT-4o and embedding models on Azure infrastructure.", metadata={"source": "azure-docs", "topic": "azure"}),
    Document(page_content="Azure AI Search supports vector, keyword and hybrid search at enterprise scale.", metadata={"source": "azure-docs", "topic": "azure"}),
    Document(page_content="FastAPI is a modern async Python framework for building production REST APIs.", metadata={"source": "python-docs", "topic": "frameworks"}),
    Document(page_content="Multi-agent systems coordinate multiple AI agents to solve complex tasks.", metadata={"source": "ai-docs", "topic": "patterns"}),
    Document(page_content="LangSmith provides observability and tracing for LangChain applications.", metadata={"source": "langchain-docs", "topic": "tooling"}),
]

vectorstore.add_documents(documents)
print(f"Added {len(documents)} documents")

import time
time.sleep(2) # Wait for indexing

# Different Search modes -----------
print("\n=== Search Modes ===")

query = "How do i build AI agents on Azure?"

# Similarity search (vector)
print(f"\nQuery: '{query}'")
print("\n-- Vector Search --")
vector_results = vectorstore.similarity_search(query, k=3)
for doc in vector_results:
    print(f"  [{doc.metadata.get('topic')}] {doc.page_content[:70]}...")

# Similarity search with score
print("\n-- Vector Search with Scores --")
scored_results= vectorstore.hybrid_search_with_relevance_scores(query, k=3)
for doc, score in scored_results:
    print(f"    Score {score:.4f}: {doc.page_content[:60]}")

# Metadata filtering -------------------
print("\n=== Metadata Filtering ===")

azure_docs = vectorstore.similarity_search(
    query="AI Services",
    k=3
)
print("Azure docs only:")
for doc in azure_docs:
    print(f"  {doc.page_content[:70]}...")

# ── RAG chain with Azure AI Search ───────────────────────
print("\n=== RAG Chain with Azure AI Search ===")

retriever = vectorstore.as_retriever(k=3)

rag_prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Answer based ONLY on context."),
    ("human", """Context:
{context}

Question: {question}""")
])

def rag(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = rag_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


questions = [
    "What is LangGraph used for?",
    "What Azure services are available for AI?",
    "What is the capital of France?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {rag(q)}")


print("""
=== FAISS vs Azure AI Search ===
FAISS:
  ✅ Fast local development
  ✅ No infrastructure needed
  ❌ Not persistent across restarts without saving
  ❌ Not production scalable

Azure AI Search:
  ✅ Production ready
  ✅ Persistent — data survives restarts
  ✅ Hybrid search built in
  ✅ Enterprise security and compliance
  ❌ Needs Azure subscription
  ❌ Slight latency vs local FAISS
""")