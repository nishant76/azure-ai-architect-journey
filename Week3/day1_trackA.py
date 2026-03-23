from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

#-- LLM Setup ---------------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-10-21",
    temperature=0.7
)

#-- Chain 1: Basic chain ----------
print("==== Chain1 : Basic LCEL Chain ===")

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Answer Concisely."),
    HumanMessagePromptTemplate.from_template("{question}")
])

basic_chain = prompt | llm | StrOutputParser()

result = basic_chain.invoke({"question": "What is LangChain in one sentence?"})
print(f"Result: {result}")

# Chain 2: Multi-step Chain ---------
print(f"\n=== Chain 2: Multi step chain")

explain_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Explain the concept clearly in 2 sentences."),
    HumanMessagePromptTemplate.from_template("{concept}")
])

simplify_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Simplify this for a 10 year old in one sentence"),
    HumanMessagePromptTemplate.from_template("{explanation}")
])

explain_chain = explain_prompt | llm | StrOutputParser()
simplify_chain = simplify_prompt | llm | StrOutputParser()

full_chain = (
    explain_chain
    | (lambda explanation: {"explanation": explanation})
    | simplify_chain
)

result = full_chain.invoke({"concept": "RAG (Retrieval Augmented Generation)"})
print(f"Simplified: {result}")

# Chain 3: Parallel chain --------------
print("\n === Chain 3: Parallel Chain ===")

pros_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("List 3 pros of the technology. Be concise."),
    HumanMessagePromptTemplate.from_template("{technology}")
])

cons_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("List 3 cons of the technology. Be concise."),
    HumanMessagePromptTemplate.from_template("{technology}")
])

parallel_chain = RunnableParallel(
    pros=(pros_prompt | llm | StrOutputParser()),
    cons=(cons_prompt | llm | StrOutputParser())
)

result = parallel_chain.invoke({"technology": "LangChain"})
print(f"Pros: \n{result['pros']}")
print(f"\nCons: \n{result['cons']}")

# --- Chain 4: Streaming ----------
print("\n=== CHain 4: Streaming ====")
print("Streaming: ", end="", flush=True)
for chunk in basic_chain.stream({ "question": "What is Azure OpenAI in 3 sentences?"}):
    print(chunk, end="", flush=True)
print()