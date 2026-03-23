from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-10-21",
    temperature=0
)

# -- Tools --------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city"""
    weather_data = {
        "dubai": "Sunny, 35°C, humidity 60%",
        "london": "Cloudy, 12°C, light rain",
        "bangalore": "Partly cloudy, 28°C, humidity 70%",
        "mumbai": "Hot and humid, 32°C, humidity 85%",
    }

    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Example: '2 + 2' or '10 * 5' """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def get_tech_info(technology: str)-> str:
    """Get information about a technology or framework."""
    tech_db = {
        "langchain": "LangChain is a framework for building LLM applications.",
        "langgraph": "LangGraph extends LangChain with stateful multi-agent workflows.",
        "fastapi": "FastAPI is a modern async Python framework for REST APIs.",
        "azure openai": "Azure OpenAI provides GPT-4o and embedding models on Azure.",
        "rag": "RAG combines retrieval with generation to ground LLM answers in real data.",
    }
    return tech_db.get(technology.lower(), f"No information found for {technology}")


@tool 
def search_jobs(role:str, location: str = "remote") -> str:
    """Search for job openings for a given role and location"""
    jobs =[
        {"title": f"Senior {role}", "company": "Microsoft", "location": location, "salary": "$150K"},
        {"title": f"{role} Architect", "company": "Google", "location": location, "salary": "$180K"},
        {"title": f"Lead {role}", "company": "Amazon", "location": location, "salary": "$160K"},
    ]
    return json.dumps(jobs, indent=2)

#--- Agent setup -------
tools = [get_weather, calculate, get_tech_info, search_jobs]

# create react agent is the modern LangGraph way
agent = create_react_agent(
    model=llm,
    tools = tools,
    prompt="You are a helpful AI assistant. Use tools when needed to answer accurately."
)

def ask(question: str) -> str:
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content

# --- Tests -------------------------
print("=== Test 1: Single tool ===")
print(ask("What is the weather in Dubai?"))

print("\n=== Test 2: Multiple tools ===")
print(ask("What is LangChain and what is 25 * 4?"))

print("\n=== Test 3: No tool needed ===")
print(ask("What is the capital of France?"))

print("\n=== Test 4: Multi-tool task ===")
print(ask("Find AI Engineer jobs in Dubai and tell me the weather there."))
