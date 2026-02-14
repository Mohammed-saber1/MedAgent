import os
import logging
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MedAgent")

class Settings(BaseSettings):
    """
    Application settings and environment variables.
    """
    # API Keys
    GROQ_API_KEY: str
    TAVILY_API_KEY: str = "" # Optional if not using search? No, usually required.
    
    # Database
    MONGODB_URI: str
    MONGODB_DB_NAME: str = "medagent_db"
    
    # Models
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Agent Settings
    MAX_SEARCH_RESULTS: int = 5
    BYPASS_REFLECTION: bool = True
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

# ── Initialize Resources ─────────────────────────────────

def get_llm():
    """Get the configured ChatGroq instance."""
    return ChatGroq(
        model=settings.LLM_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY
    )

def get_embeddings():
    """Get the configured Embeddings instance."""
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

# Export instances for backward compatibility
# Deprecated: Import functions or settings instead in new code
LLM = get_llm()
FINETUNED_MODEL = LLM # Alias as per old config
embeddings = get_embeddings()
