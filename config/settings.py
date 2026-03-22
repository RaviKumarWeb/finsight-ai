from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # LLM
    GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
    
    # Search
    TAVILY_API_KEY: str = os.environ["TAVILY_API_KEY"]
    
    # Reranking
    COHERE_API_KEY: str = os.environ["COHERE_API_KEY"]
    
    # Vector DB
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///finsight.db"
    )
    
    # Models
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()

if __name__ == "__main__":
    from config.settings import settings
    print("✅ Settings loaded")
    print(f"Model: {settings.LLM_MODEL}")
    print(f"Groq key exists: {bool(settings.GROQ_API_KEY)}")
    print(f"Tavily key exists: {bool(settings.TAVILY_API_KEY)}")