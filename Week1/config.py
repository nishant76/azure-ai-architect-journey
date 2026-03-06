from dotenv import load_dotenv
import os

load_dotenv()

#Centralized config - like appsettings.json in .net
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY=os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT= os.getenv("AZURE_OPENAI_DEPLOYMENT")

#Validate on startup - fail fast if config missing
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")

    