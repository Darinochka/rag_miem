from fastapi import FastAPI
from langserve import add_routes

import src.retriever as retriever_utils
from src.servers.env_args import RetrieverArgs

args = RetrieverArgs()
documents = retriever_utils.create_documents(
    folder=args.data_folder,
    target_column=args.target_field,
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap,
)
vectorstore = retriever_utils.create_db(documents, embeddings_model=args.model_name)
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

    uvicorn.run(app, host=args.retriever_host, port=args.retriever_port)
