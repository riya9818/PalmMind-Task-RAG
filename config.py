from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenRouter API key 
    OPENROUTER_API_KEY: str 
    OPENROUTER_MODEL: str = "meta-llama/llama-3.2-3b-instruct:free"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "documents"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Embedding model 
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Database
    DATABASE_URL: str = "sqlite:///./rag.db"

    # Uploads
    UPLOAD_DIR: str = "uploads"

    class Config:
        env_file = ".env"


settings = Settings()