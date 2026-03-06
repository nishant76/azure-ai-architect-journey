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


# ----Test 1" Multi-turn conversation -----
# This is how chatbots maintain context

print("=== Multi turn conversation =====")

messages= [
    {"role": "system", "content": "You are a helpful Azure AI assistant."}
]

#Turn 1
messages.append({"role": "user", "content": "My name is Nishant."})
response = client.chat.completions.create(model=deployment, messages=messages)
reply = response.choices[0].message.content
messages.append({"role": "assistant", "content": reply})
print(f"Turn 1: {reply}")

#Turn 2 - does it remember the name?
messages.append({"role": "user", "content": "What is my name?"})
response = client.chat.completions.create(model=deployment, messages=messages)
reply = response.choices[0].message.content
messages.append({ "role": "assistant", "content": reply})
print(f"Turn 2: {reply}")

# Turn 3
messages.append({ "role": "user", "content": "What is LangChain?"})
response = client.chat.completions.create(model=deployment, messages=messages)
reply = response.choices[0].message.content
print(f"Turn 3: {reply}")

# Test 3: Zero-shot vs Few-shot
print("\\n === Zero-shot prompting ===")
response = client.chat.completions.create(
    model=deployment, 
    messages=[
        {"role": "system", "content": "Classify the sentiment of the text as postive, negative or neutral."},
        { "role": 'user', "content": "The product quality is terrible and I want a refund."}
    ]
)
print(response.choices[0].message.content)

print("\\n === Few-shot prompting===")
response = client.chat.completions.create(
    model=deployment, 
    messages=[{
        "role": "system", "content": """Classify sentiment. Examples: 
        Text: I love this product -> Positive
        Text: This is okay -> Neutral
        Text: Worst experience ever -> Negative"""},
        {"role": "user", "content": "The product quality is terrible and I want a refund."}
    ]
)
print(response.choices[0].message.content)

#Test 3: Structured JSON output -----
response = client.chat.completions.create(
    model=deployment,
    messages=[
        { "role": "system", "content": """Extract information and return ONLY valid JSON. 
         Format: {"name": "", "role": "", "skills": []}"""},
         {"role": "user", "content": "Nishant is a Lead software engineer in C#, Azure and .NET"}
    ]
)
print(response.choices[0].message.content)

#----Test 4: Chain of thought -------------
print("\\n ----Chain of thought------")
response = client.chat.completions.create(
    model=deployment,
    messages = [
        { "role": "system", "content": "Think step by step before giving your answer." },
        { "role": "user", "content": "Should a company with 10,000 employees use RAG or fine-tuning for their internal Q&A bot?" }
    ]
)

print(response.choices[0].message.content)