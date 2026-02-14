import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Constants
MAX_ITERATIONS = 3
BYPASS_REFLECTION = True  # Set to True to bypass reflection LLM calls

# Initialize LLMs — all using Groq (no local Ollama needed)
def get_groq_model():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )

# Both use Groq — FINETUNED_MODEL kept for backward compatibility with agent imports
FINETUNED_MODEL = get_groq_model()
LLM = get_groq_model()

