from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_default_model: str = "qwen3.5:4b"
    ollama_deep_model: str = "qwen3.5:9b"
    ollama_num_predict: int = 8192
    ollama_context_length: int = 4096

    tavily_api_key: str | None = None

    chroma_persist_dir: str = "./chroma_db"
    eval_file_path: str = "./eval_results.json"
    sessions_dir: str = "./sessions"

    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    return Settings()