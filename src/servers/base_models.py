from pydantic import validator
from pydantic import BaseModel
from typing import List, Dict, Any
from pydantic_settings import BaseSettings


class TelegramArgs(BaseSettings):
    token: str
    retriever_host: str = "http://retriever:8000"
    generator_host: str
    model_name: str
    generator_type: str

    @validator("generator_type")
    def check_generator_type(cls, v: str) -> str:
        if v not in ["ollama", "openai"]:
            raise ValueError(f"Invalid generator type: {v}")
        return v

    class Config:
        env_prefix = ""
        env_file = ".env"


class RetrieverArgs(BaseSettings):
    data_folder: str = "/data"
    target_field: str = "text"
    chunk_size: int = 300
    chunk_overlap: int = 30
    model_name: str = "ai-forever/ruBert-base"
    retriever_host: str = "0.0.0.0"
    retriever_port: int = 8000
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    class Config:
        env_prefix = ""
        env_file = ".env"


class RerankRequest(BaseModel):
    query: str
    docs: List[Dict[str, Any]]


class RerankResponse(BaseModel):
    reranked_docs: List[Dict[str, Any]]


# Example usage
if __name__ == "__main__":
    telegram_args = TelegramArgs()
    print(f"Telegram args: {telegram_args.dict()}")

    retriever_args = RetrieverArgs()
    print(f"Retriever args: {retriever_args.dict()}")
