from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional


class TelegramArgs(BaseSettings):
    token: str
    retriever_host: str
    retriever_person_host: str
    study_classifier_host: str
    generator_host: str
    llm_name: str


class RetrieverArgs(BaseModel):
    embedding_model: str
    rerank_model: Optional[str]
    normalize_embeddings: bool
    ensemble: bool = False
