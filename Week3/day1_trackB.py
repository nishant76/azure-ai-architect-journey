from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessageChunk
from dotenv import load_dotenv
import os

load_dotenv()

# --- LLM + Embeddings --------------------
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

# --- Knowledge Base -----------------
documents = [
    "LangChain is a framework for building LLM applications with chains and agents.",
    "LangGraph extends LangChain with stateful multi-agent graph workflows.",
    "RAG combines retrieval with generation to ground LLM answers in real data.",
    "Azure OpenAI provides GPT-4o and embedding models on Azure infrastructure.",
    "Azure AI Search supports vector, keyword and hybrid search at enterprise scale.",
    "FastAPI is a modern async Python framework for building production REST APIs.",
    "Multi-agent systems coordinate multiple AI agents to solve complex tasks.",
    "Azure Container Apps deploys containerized AI apps with automatic scaling.",
    "Pydantic provides data validation using Python type hints.",
    "FAISS is a library for efficient similarity search on dense vectors."
]

# -- Build FAISS Vector store -----------
print("=== Building Vector Store====")
vectorstore= FAISS.from_texts(
    texts=documents,
    embedding=embeddings
)
print(f"Vecotr store built with {len(documents)} documents")

#--Retriever -----------
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
)

# --- RAG Prompt -------
rag_prompt = ChatPromptTemplate([
    ("system", """You are a helpful AI assistant.
Answer based ONLY on the provided context.
If the answer is not in context say: 'I do not have enough information.'"""),
    ("human", """Context:
{context}

Question: {question}""")
])

#--- Format docs ------------
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# -- RAG Chain ------------
rag_chain = (
    {
        "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x))),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# -- Test ---------------------
questions = [
    "What is LangGraph used for?",
    "How does RAG work?",
    "What is the capital of France?",  # out of scope
]

print("\n ======Testing LangChain RAG ----")
print("\n=== Debug Chain ===")
test_input = "What is LangGraph used for?"

# Step 1: check retriever
docs = retriever.invoke(test_input)
print(f"Retriever works: {len(docs)} docs")

# Step 2: check format
context = format_docs(docs)
print(f"Context: {context[:100]}")

# Step 3: check prompt
prompt_result = rag_prompt.invoke({"context": context, "question": test_input})
print(f"Prompt works: {prompt_result}")

# Step 4: check LLM directly
llm_result = llm.invoke(prompt_result)
print(f"LLM result: {llm_result.content}")
for q in questions:
    print(f"\n Q: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")

# -- Save and reload -------
print("\n === Saving Vector Store ====")
vectorstore.save_local("faiss_index")
print("saved!")

print("\n==== Loading vector store ====")
loaded_vs = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)

loaded_retriever = loaded_vs.as_retriever(search_kwargs={"k": 3})
docs = loaded_retriever.invoke("What is Azure AI Search?")
print(f"Retrieved {len(docs)} docs from loaded index")
print(f"\nDebug — retrieved {len(docs)} docs:")
for doc in docs:
    print(f"    - {doc.page_content[:80]}")


print("""
    ==== Week 2 vs Week 3= = ===
    Week 2 manual RAG: --80 lines of code
    Week 3 LangChain RAG: ~40 lines of code
    Same result - LangChain just makes it cleaner!
      """)