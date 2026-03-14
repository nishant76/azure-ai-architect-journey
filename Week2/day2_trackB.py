from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField as VectorField
)
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

#--- Clients ---------
openai_client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key =os.getenv("AZURE_SEARCH_KEY")
index_name ="ai-knowledge-base"

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(search_key)
)

#--Create index --------------
def create_index():
    print("==== Creating Azure AI Search Index ===")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
        VectorField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")],
        profiles=[VectorSearchProfile(
            name="my-vector-profile",
            algorithm_configuration_name="my-hnsw"
        )]
    )

    index= SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )

    index_client.create_or_update_index(index)
    print(f"Index '{index_name}' created successfully")


#--- Generate embedding ------------
def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

#--- Upload documents -----------
def upload_documents():
    print("\n ==== Uploading documents =====")

    documents = [
        {"id": "1", "content": "LangChain is a framework for building LLM applications.", "category": "frameworks"},
        {"id": "2", "content": "LangGraph extends LangChain with stateful multi-agent workflows.", "category": "frameworks"},
        {"id": "3", "content": "RAG grounds LLM answers in real data by retrieving relevant context.", "category": "patterns"},
        {"id": "4", "content": "Azure OpenAI provides GPT-4o and embedding models on Azure.", "category": "azure"},
        {"id": "5", "content": "Azure AI Search supports vector, keyword and hybrid search.", "category": "azure"},
        {"id": "6", "content": "FastAPI is a modern Python framework for building REST APIs.", "category": "frameworks"},
        {"id": "7", "content": "Multi-agent systems use multiple AI agents for complex tasks.", "category": "patterns"},
        {"id": "8", "content": "Azure Container Apps deploys containerized apps serverlessly.", "category": "azure"},
    ]

    # Add embeddings to each document
    print("Generating embeddings for documents...")
    for doc in documents:
        doc["content_vector"]= get_embedding(doc["content"])

    # Upload to Azure AI Search
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key)
    )

    result = search_client.upload_documents(documents)
    print(f"Uploaded {len(documents)} docuements successfully")
    return search_client

# --- Search functions -----------
def keyword_search(client: SearchClient, query: str):
    print(f"\n --- Keyword search: '{query}' ----")
    results = client.search(search_text=query, top=3)
    for r in results:
        print(f" [{r['category']}]  {r['content']}")

def vector_search(client: SearchClient, query: str):
    print(f"\n--- Vector search: '{query}' ----")
    query_vector = get_embedding(query)
    results = client.search(
        search_text=None,
        vector_queries=[{
            "kind": "vector",
            "vector": query_vector,
            "fields": "content_vector",
            "k": 3
        }]
    )
    for r in results:
        print(f" '' [{r['category']}]  {r['content']}")


def hybrid_search(client: SearchClient, query: str):
    print(f"\n --- Hybrid search: '{query}' ----")
    query_vector = get_embedding(query)
    results = client.search(
        search_text=query,
        vector_queries=[{
            "kind": "vector",
            "vector": query_vector,
            "fields": "content_vector",
            "k": 3
        }],
        top=3
    )
    for r in results:
        print(f"    [{r['category']}]   {r['content']}")


#--- Main ----------------
create_index()
search_client = upload_documents()


print("\n === Testing search modes ====")
query = "How do I build AI agents on Azure?"

keyword_search(search_client, query)
vector_search(search_client, query)
hybrid_search(search_client, query)

print("\n ======Filter by category ====")
results = search_client.search(
    search_text="deployment",
    filter="category eq 'asd'",
    top=3
)
for r in results:
    print(f"   [{r['category']}]    {r['content']}")