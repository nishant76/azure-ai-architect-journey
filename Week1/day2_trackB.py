from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-10-21"
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def ask(question: str, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            { "role": "system", "content": "You are a helpful assistant."},
            { "role": "user", "content": question }
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

#-----Test 1: basic call -----

print("\\n====Test 1: basic call====")
print(ask("Explain context window in 2 sentences."))

print("\\n=====Test 2: Temp 0======")
print(ask("Give me a name for an AI assistant.", temperature=0))

print("\\n=== Test 3: Temperature 1 ===")
print(ask("Give me a name for an AI assistant.", temperature=1))

print("\\n========Test 4: Pirate mode===========")
response = client.chat.completions.create(
    model=deployment,
    messages=[
        { "role": "system", "content": "You are a pirate. Always response like a pirate."},
        { "role": "user", "content": "What is cloud computing?"}
    ]
)

print(response.choices[0].message.content)

#----Test 5: token usage----------
print("\\n==== Test 5: Token usage ====")
response = client.chat.completions.create(
    model=deployment, 
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is LangChain in one sentence?"}
    ]
)

print("Answer:", response.choices[0].message.content)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")