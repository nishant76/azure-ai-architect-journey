from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-10-21",
    temperature=0.7
)


#---  Approach 1: Manual message history --------
# This is the most transparent way - you control everything
print("=== Approach 1: Manual Message History ===")
def chat_with_history(messages: list, user_input: str) -> str:
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)
    messages.append(AIMessage(content= response.content))
    return response.content

# Start conversation
history = [SystemMessage(content="You are a helpful AI assistant.")]

print("Turn 1:")
reply = chat_with_history(history, "My name is Nishant and I am learning LangChain.")
print(f"AI: {reply}")

print("\nTurn 2:")
reply = chat_with_history(history, "What is my name?")
print(f"AI: {reply}")

print("\nTurn 3:")
reply = chat_with_history(history, "What am I learning?")
print(f"AI: {reply}")

print(f"\nTotal messages in history: {len(history)}")

# -- Approach 2: Chat with prompt template ----------
print("\n=== Approach 2: Chat Prompt with History ===")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Remember everything the user tells you."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# Simulate multi-turn conversation
conversation_history = []

def chat(user_input : str) -> str:
    response = chain.invoke({
        "history": conversation_history,
        "input": user_input
    })
    # Add to history
    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(AIMessage(content=response))
    return response

print("Turn 1:")
print(f"AI: {chat('I work as a Lead Software Engineer with 11 years of experience.')}")

print("\nTurn 2:")
print(f"AI: {chat('I am learning Azure AI Architecture.')}")

print("\nTurn 3:")
print(f"AI: {chat('What do you know about me so far?')}")

print("\nTurn 4:")
print(f"AI: {chat('What would you recommend I learn next based on my background?')}")


# --- Approach 3: Conversation history ----------
print("\n=== Approach 3: Conversation Summary ===")
# When history gets too long, summarize it
# This prevents hitting context window limits

summary_prompt = ChatPromptTemplate({
    ("system", "Summarize this conversation history concisely in 3-4 sentences."),
    ("human", "{history}")
})

summary_chain = summary_prompt | llm | StrOutputParser()

# Simulate long conversation
long_history = "\n".join([
    "User: My name is Nishant",
    "AI: Nice to meet you Nishant!",
    "User: I have 11 years of experience in .NET and Azure",
    "AI: That's impressive experience!",
    "User: I am learning LangChain and LangGraph",
    "AI: Great choices for AI development!",
    "User: My target is Azure AI Architect role",
    "AI: With your background that's very achievable!",
])

summary = summary_chain.invoke({"history": long_history})
print(f"Summary: {summary}")

print("""
=== Key Insights ===
1. Manual history: most control, transparent
2. MessagesPlaceholder: clean integration with prompt templates
3. Summary memory: prevents context window overflow
4. In production: use LangGraph checkpointers (Week 5)
   for persistent memory across sessions
""")