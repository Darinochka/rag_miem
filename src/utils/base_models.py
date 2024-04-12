from pydantic import validator, BaseModel
from pydantic_settings import BaseSettings
from typing import Optional


class TelegramArgs(BaseSettings):
    token: str
    retriever_host: str
    generator_host: str
    llm_name: str
    generator_type: str

    @validator("generator_type")
    def check_generator_type(cls, v: str) -> str:
        if v not in ["ollama", "openai"]:
            raise ValueError(f"Invalid generator type: {v}")
        return v


class RetrieverArgs(BaseModel):
    embedding_model: str
    rerank_model: Optional[str]
    normalize_embeddings: bool
