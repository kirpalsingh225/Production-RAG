from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pinecone_api_key: str
    pinecone_index_host: str
    pinecone_namespace: str = "default"

    embedding_model: str = "llama-text-embed-v2"
    embedding_dimension: int = 1024


settings = Settings()