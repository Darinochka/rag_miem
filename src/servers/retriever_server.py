from fastapi import FastAPI
from langserve import add_routes
import src.utils.retriever as retriever_utils
from src.servers.base_models import RetrieverArgs
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


args = RetrieverArgs()
documents = retriever_utils.create_documents(
    folder=args.data_folder,
    target_column=args.target_field,
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap,
)
retriever = retriever_utils.create_db(
    documents, embeddings_model=args.embedding_model, normalize_embeddings=False
).as_retriever()

model = HuggingFaceCrossEncoder(model_name=args.reranker_model)
compressor = CrossEncoderReranker(model=model, top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# retriever = compression_retriever.as_retriever()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, compression_retriever)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.retriever_host, port=args.retriever_port)
