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

    # Simple call counter
    call_count = 0

    @staticmethod
    def increment_calls(agent_name: str):
        global call_count
        Settings.call_count += 1
        print(f"LLM CALL #{Settings.call_count} from {agent_name}")


settings = Settings()
