from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# ── Vector store ──────────────────────────────────────────
documents = [
    "LangChain is a framework for building LLM applications with chains and agents.",
    "LangGraph extends LangChain with stateful multi-agent graph workflows.",
    "RAG combines retrieval with generation to ground LLM answers in real data.",
    "Azure OpenAI provides GPT-4o and embedding models on Azure infrastructure.",
    "Azure AI Search supports vector, keyword and hybrid search at enterprise scale.",
    "FastAPI is a modern async Python framework for building production REST APIs.",
    "Multi-agent systems coordinate multiple AI agents to solve complex tasks.",
    "Prompt engineering involves crafting inputs to get better outputs from LLMs.",
    "Vector databases store embeddings for fast semantic similarity search.",
    "LangSmith provides observability and tracing for LangChain applications.",
]

vectorstore=FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Technique 1: Query rewritting -----
print("==== Technique 1: Query Rewritting ===")

rewrite_prompt = ChatPromptTemplate([
    ("system", """Rewriet the question to improve document retrieval.
     Make it more specific and technical.
     Return ONLY the rewritten question, nothing else."""),
     ("human", "{question}")
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

original = "How do I build AI Agents?"
rewritten = rewrite_chain.invoke({"question": original})
print(f"Original: {original}")
print(f"Rewritten: {rewritten}")


# --Technique 2: Multi-Query Retrieval ----
print("\n=== Technique 2: Multi-Query Retrieval ===")

multi_query_prompt = ChatPromptTemplate([
    ("system", """Generate 3 different versions of this question
     to retrieve more relevant documents.
     Return ONLY the 3 questions, one per line."""),
     ("human", "{question}")
])

multi_query_chain = multi_query_prompt | llm | StrOutputParser()

question = "What tools help build production AI systems?"
queries = multi_query_chain.invoke({"question": question})
print(f"Original: {question}")
print(f"Generated queries:\n{queries}")

#Retrieve for all queries and merge
all_docs = []
for q in queries.strip().split('\n'):
    if q.strip():
        docs = retriever.invoke(q.strip())
        all_docs.extend(docs)

# Deduplicate

seen = set()
unique_docs = []
for doc in all_docs:
    if doc.page_content not in seen:
        seen.add(doc.page_content)
        unique_docs.append(doc)

print(f"\nUnique docs retrieved: {len(unique_docs)}")
for doc in unique_docs:
    print(f"    - {doc.page_content[:70]}...")


# Technique 3: Advanced RAG pipeline ---------
print("\n=== Technique 3: Advanced RAG Pipeline ===")
rag_prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Answer based ONLY on context provided."),
    ("human", """Context:
{context}

Question: {question}""")
])

def advanced_rag(question: str) -> str:
    # Step 1: Rewrite query
    better_query = rewrite_chain.invoke({"question": question})
    print(f"Rewritten: {better_query}")

    #Step 2: Retrieve with better query
    docs = retriever.invoke(better_query)
    context = format_docs(docs)

    #Step 3: Generate
    response = rag_prompt | llm | StrOutputParser()
    return response.invoke({"context": context, "question": question})

# ── Technique 4: RAG Evaluation ──────────────────────────
print("\n=== Technique 4: RAG Evaluation ===")

eval_prompt = ChatPromptTemplate([
    ("system", """Evaluate the RAG answer on 3 dimensions.
     Return ONLY valid JSON:
     {{"faithfulness": <0-10>, "relevance": <0-10>, "completeness": <0-10>, "feedback": "<one sentence>"}}"""),
    ("human", """Question: {question}
Context: {context}
Answer: {answer}

Evaluate:""")
])

def evaluate_rag(question: str, context: str, answer: str) -> dict:
    import json
    eval_chain = eval_prompt | llm | StrOutputParser()
    result = eval_chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    })
    try:
        return json.loads(result)
    except:
        return {"error": "Could not parse", "raw": result}
    
# Test advanced RAG + evaluation
questions = [
    "How do I make AI remember things?",
    "What helps search through lots of documents?",
]

print("\nAdvanced RAG + Evaluation:")
for q in questions:
    print(f"\nQ: {q}")
    better_query=rewrite_chain.invoke({"question": q})
    docs = retriever.invoke(better_query)
    context = format_docs(docs)
    answer = advanced_rag(q)
    print(f"A: {answer}")

    # Evaluate
    scores = evaluate_rag(q, context, answer)
    print(f"Faithfulness: {scores.get('faithfulness')}/10")
    print(f"Relevance:    {scores.get('relevance')}/10")
    print(f"Completeness: {scores.get('completeness')}/10")
    print(f"Feedback:     {scores.get('feedback')}")

print("""
=== Summary ===
1. Query Rewriting   → better queries = better retrieval
2. Multi-Query       → multiple angles = more coverage
3. Advanced Pipeline → rewrite + retrieve + generate
4. RAG Evaluation    → measure quality, find gaps
""")