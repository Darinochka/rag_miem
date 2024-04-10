from pydantic import validator
from pydantic_settings import BaseSettings


class TelegramArgs(BaseSettings):
    token: str
    retriever_host: str = "http://retriever:8000"
    generator_host: str
    llm: str
    generator_type: str

    @validator("generator_type")
    def check_generator_type(cls, v: str) -> str:
        if v not in ["ollama", "openai"]:
            raise ValueError(f"Invalid generator type: {v}")
        return v


class RetrieverArgs(BaseSettings):
    data_folder: str = "/data"
    target_field: str = "text"
    chunk_size: int = 700
    chunk_overlap: int = 200
    embedding_model: str = "BAAI/bge-m3"
    retriever_host: str = "0.0.0.0"
    retriever_port: int = 8000
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


# Example usage
if __name__ == "__main__":
    telegram_args = TelegramArgs()
    print(f"Telegram args: {telegram_args}")

    retriever_args = RetrieverArgs()
    print(f"Retriever args: {retriever_args}")
