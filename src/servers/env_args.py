import os


class TelegramArgs:
    def __init__(self) -> None:
        self.token_env = os.getenv("TOKEN")
        self.retriever_host = os.getenv("RETRIEVER_HOST_PORT", "http://retriever:8000")
        self.generator_host_env = os.getenv("GENERATOR_HOST")
        self.model_name_env = os.getenv("LLM")
        self.generator_type_env = os.getenv("GENERATOR_TYPE")

        self.check_variables()
        print(f"Retriever args: {self.__dict__}")

    def check_variables(self) -> None:
        if self.generator_type_env in ["ollama", "openai"]:
            self.generator_type = self.generator_type_env
        else:
            raise ValueError(f"Invalid generator type: {self.generator_type}")

        if self.token_env is not None:
            self.token = self.token_env
        else:
            raise ValueError("Telegram token is not set")

        if self.generator_host_env is not None:
            self.generator_host = self.generator_host_env
        else:
            raise ValueError("Generator host is not set")

        if self.model_name_env is not None:
            self.model_name = self.model_name_env
        else:
            raise ValueError("Model name is not set")


class RetrieverArgs:
    def __init__(self) -> None:
        self.data_folder = os.getenv("DATA_FOLDER", "/data")
        self.target_field = os.getenv("TARGET_FIELD", "text")
        self.chunk_size_str = os.getenv("CHUNK_SIZE", 300)
        self.chunk_overlap_str = os.getenv("CHUNK_OVERLAP", 30)
        self.model_name = os.getenv("EMBEDDING_MODEL", "ai-forever/ruBert-base")
        self.retriever_host = os.getenv("RETRIEVER_HOST", "0.0.0.0")
        self.retriever_port = os.getenv("RETRIEVER_PORT", 8000)

        self.check_variables()
        print(f"Retriever args: {self.__dict__}")

    def check_variables(self) -> None:
        try:
            self.chunk_size = int(self.chunk_size_str)
            self.chunk_overlap = int(self.chunk_overlap_str)
            self.retriever_port = int(self.retriever_port)
        except ValueError:
            raise ValueError(
                f"Invalid chunk size or overlap: {self.chunk_size_str}, {self.chunk_overlap_str}"
            )
