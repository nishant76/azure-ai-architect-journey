from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader, 
    WebBaseLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
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


# ── Text Splitters comparison ─────────────────────────────
print("=== Text Splitter Comparison ===")

sample_text = """
LangChain is a framework for developing applications powered by language models.
It provides tools for chaining prompts, managing memory, and integrating with external data.

LangGraph extends LangChain by enabling stateful multi-agent workflows.
It uses a graph-based approach where nodes represent processing steps.
LangGraph is ideal for complex workflows that require loops and branching.

RAG stands for Retrieval Augmented Generation.
It combines language models with the ability to search and retrieve information.
RAG helps reduce hallucinations by grounding answers in real data sources.

Azure OpenAI provides access to GPT-4o and embedding models on Azure infrastructure.
It integrates natively with Azure services like Azure AI Search and Azure Monitor.
Azure OpenAI supports enterprise security, compliance, and governance requirements.
"""


# Splitter 1: Character splitter
char_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)
char_chunks = char_splitter.split_text(sample_text)
print(f"Character splitter: {len(char_chunks)} chunks")
for i, chunk in enumerate(char_chunks[:2]):
    print(f"    Chunk {i+1}: {chunk[:80]}...")

# Splitter 2: Recursive character splitter (recommended)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ".", " "]
)

recursive_chunks = recursive_splitter.split_text(sample_text)
print(f"\nRecursive splitter: {len(recursive_chunks)} chunks")
for i, chunk in enumerate(recursive_chunks[:2]):
    print(f"  Chunk {i+1}: {chunk[:80]}...")

print("""
Key difference:
- CharacterTextSplitter: splits on ONE separator
- RecursiveCharacterTextSplitter: tries multiple separators in order
- Recursive is almost always better — use it by default
""")


# --- Document Loaders ------------
print("=== Document Loaders =====")

# Loader 1: Text file
print("---- Text loader ----")
#Create a sample text file
with open("sample.txt", "w") as f:
    f.write(sample_text)

text_loader = TextLoader("sample.txt")
text_docs = text_loader.load()
print(f"Loaded {len(text_docs)} document(s)")
print(f"Content preview: {text_docs[0].page_content[:100]}...")
print(f"Metadata: {text_docs[0].metadata}")

# Loader 2: Web loader
print("\n --- Web loader ----")
try:
    import urllib.request
    url = "https://python.langchain.com/docs/introduction/"
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')

    # Clean basic HTML tags
    import re
    clean = re.sub('<[^<]+?>', '', content)
    web_docs = [Document(page_content=clean[:3000], metadata={"source": url})]
    print(f"Loaded web content: {len(web_docs[0].page_content)} chars")
except Exception as e:
    print(f"Web loader error: {e}")
    web_docs = []


# ── Full RAG Pipeline with Document Loader ───────────────
print("\n=== Full RAG Pipeline with Real Documents ===")

# Use our text file as knowledge base
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

#Split Documents 
all_docs = splitter.split_documents(text_docs)
print(f"Split into {len(all_docs)} chunks")

#Build vector store
vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG chain
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
    "How does RAG reduce hallucinations?",
    "What Azure services does Azure OpenAI integrate with?",
]


for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {rag(q)}")

# Cleanup
os.remove("sample.txt")

print("""
=== Summary ===
Document Loaders: TextLoader, PyPDFLoader, WebBaseLoader
Text Splitters:   CharacterTextSplitter, RecursiveCharacterTextSplitter
Best practice:    Always use RecursiveCharacterTextSplitter
Pipeline:         Load → Split → Embed → Store → Retrieve → Generate
""")