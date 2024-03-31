#!/usr/bin/env python
"""Example of using here https://github.com/langchain-ai/langserve/blob/main/examples/retrieval/client.ipynb"""
from fastapi import FastAPI
from langserve import add_routes

import src.retriever as retriever_utils
from src.servers.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_FOLDER,
    MODEL_NAME,
    RETRIEVER_HOST,
    RETRIEVER_PORT,
    TARGET_FIELD,
)

documents = retriever_utils.create_documents(
    folder=DATA_FOLDER,
    target_column=TARGET_FIELD,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
vectorstore = retriever_utils.create_db(documents, embeddings_model=MODEL_NAME)
retriever = vectorstore.as_retriever()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, retriever)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=RETRIEVER_HOST, port=int(RETRIEVER_PORT))
