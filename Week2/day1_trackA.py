import re
from pathlib import Path

#---------Chunking Strategies -------------
# This is the foundation of RAG - how you split documents matters

def chunk_fixed(text: str, chunk_size: int = 500, overlap:int = 50) -> list[str]:
    """
    Split text into fixed size chunks with overlap.
    Overlap ensures context isn't lost at boundaries.
    Think of overlap like a sliding window.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap # overlap with previous chunk

    return chunks

# Strategy 2: Paragraph chunking
def chunk_by_paragraph(text: str, max_size: int = 1000) -> list[str]:
    """
    Split by paragraphs — more natural boundaries.
    Better for prose documents like policies, articles.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"

    if current:
        chunks.append(current.strip())

    return chunks

#Strategy 3: Sentence chunking
def chunk_by_sentences(text: str, sentences_per_chunk: int = 3) -> list[str]:
    """
    Split by sentences - good for Q&A over factual content.
    """
    sentences = re.split(r'(?<=[.!?])\s', text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks

# ── Test with sample text ────────────────────────────────
sample_text = """
Azure OpenAI Service provides REST API access to OpenAI's powerful language models.
These models can be easily adapted to your specific task including content generation,
summarization, semantic search, and natural language to code translation.

RAG stands for Retrieval Augmented Generation. It is a technique that combines
the power of large language models with the ability to search and retrieve
relevant information from a knowledge base.

LangChain is a framework for developing applications powered by language models.
It enables applications that are context-aware and can reason about their environment.
LangChain connects language models to other sources of data and allows them to
interact with their environment.

LangGraph is a library for building stateful, multi-actor applications with LLMs.
It extends LangChain with the ability to coordinate multiple chains or actors
across multiple steps of computation in a cyclic manner.
"""

print("======= Fixed size chunks =====")
fixed_chunks = chunk_fixed(sample_text, chunk_size=200, overlap=30)
for i, chunk in enumerate(fixed_chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk[:100] + "....")

print("\n==== Paragraph chunks =====")
para_chunks = chunk_by_paragraph(sample_text, max_size=300)
for i, chunk in enumerate(para_chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk[:100]+ ".....")

print("\n====== Sentence chunks====")
sent_chunks = chunk_by_sentences(sample_text, sentences_per_chunk=2)
for i, chunk in enumerate(sent_chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)


#--- Architect thinking -----------
print("\n==== Chunk Analysis========")
print(f"Fixed chunks: {len(fixed_chunks)}")
print(f"Paragraph chunks: {len(para_chunks)}")
print(f"Sentence chunks: {len(sent_chunks)}")
print("""
When to use which strategy:
- Fixed size: Large documents, consistent content, simple use cases
- Paragraph: Articles, policies, documentation
- Sentence: Q&A, factual content, precise retrieval needed
""")