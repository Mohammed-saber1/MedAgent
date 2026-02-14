import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

# Constants
MAX_ITERATIONS = 3
BYPASS_REFLECTION = True  # Set to True to bypass reflection LLM calls

# Initialize LLMs
def get_finetuned_model():
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "medllama2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,  # Keeping it deterministic for medical facts
        max_retries=3
    )

def get_groq_model():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )

# Singleton instances (lazy initialization pattern commonly used, but direct instantiation is fine here too)
# We use functions to allow for easier testing/mocking if needed, 
# but providing direct instances for simplicity as per original TS structure.

FINETUNED_MODEL = get_finetuned_model()
LLM = get_groq_model()
