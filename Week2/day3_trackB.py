from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

#- -- Technique 1: Query rewriting --------
def rewrite_query(original_query: str) -> str:
    """
    Generate multiple versions of the query.
    Different phrasings retrieve different relevant chunks.
    """

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": """Generate 3 different versions of the given question 
to improve document retrieval. Each version should:
- Use different words/phrases
- Approach the topic from a different angle
- Be concise and clear
Return ONLY the 3 questions, one per line, no numbering."""},
            {"role": "user", "content": original_query}
        ],
        temperature=0.7
    )

    rewrites = response.choices[0].message.content.strip().split("\n")
    rewrites = [q.strip() for q in rewrites if q.strip()]
    return rewrites

#-- Technique 2: RAG Evaluation --------
def evaluate_answer(question: str, context: str, answer: str) -> dict:
    """
    Evaluate RAG answer quality on 3 dimensions:
    - Faithfulness: is the answer grounded in context?
    - Relevance: does it answer the question?
    - Completeness: is it thorough enough?
    """

    responnse = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": """Evaluate the RAG answer on these dimensions.
Return ONLY valid JSON in this exact format:
{
  "faithfulness": <0-10>,
  "relevance": <0-10>,
  "completeness": <0-10>,
  "feedback": "<one sentence feedback>"
}"""},
            {"role": "user", "content": f"""Question: {question}

Context used: {context}

Answer given: {answer}

Evaluate:"""}
        ],
        temperature=0
    )
    
    import json
    try:
        return json.loads(responnse.choices[0].message.content)
    except:
        return { "error": "Could not parse evaluation" }
    
# -- Technique 3: HyDE ------
#Hypothetical document embeddings
#Generate a fake answer first, use it to search
def hyde_query(question: str) -> str:
    """
    Generate a hypothetical answer to use as search query.
    This often retrieves better chunks than the raw question.
    """
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": """Write a hypothetical answer to the question.
This will be used to search a knowledge base.
Be specific and technical. 2-3 sentences max."""},
            {"role": "user", "content": question}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content

# ---Test all techniques ------------
question = "How do I build a production RAG system on Azure?"

print("\n=== Query Rewriting ===")
print("Original Question:", question)
rewrites = rewrite_query(question)
print("\nRewritten Queries:")
for i, q in enumerate(rewrites):
    print(f"  {i+1}. {q}")

print("\n== RAG Evaluation ==")
sample_context = """Azure AI Search supports vector, keyword and hybrid search.
Azure OpenAI provides GPT-4o and embedding models.
RAG grounds LLM answers in real data to reduce hallucinations."""

sample_answer = """To build a production RAG system on Azure, use Azure AI Search for vector stroage and retrieval, 
Azure OpenAI for embeddings and generation, and combine them with hybrid search for best results."""

evaluation = evaluate_answer(question, sample_context, sample_answer)
print("\nEvaluation results")
print(f" Faithfulness: {evaluation.get('faithfulness')}/10")
print(f" Relevance: {evaluation.get('relevance')}/10")
print(f" Completeness: {evaluation.get('completeness')}/10")
print(f" Feedback: {evaluation.get('feedback')}/10")

print("\n=== HyDE ===")
print("Question:", question)
hypothetical = hyde_query(question)
print(f"\nHypothetical Answer (used for search query)")
print(f"  {hypothetical}")
print("""
Why HyDE works:
- Raw question: 'How do I build RAG on Azure?' (short, vague)
- HyDE answer: detailed technical text (matches document style better)
- Using the hypothetical answer as embedding query finds better chunks
      """)

print("=== Summary: RAG improvement techniques:===")
print("""
1. Query Rewriting  → multiple phrasings = better coverage
2. Evaluation       → measure quality, find gaps
3. HyDE             → hypothetical answer = better retrieval
4. These techniques go into Advanced RAG in Week 3 (LangChain)
""")