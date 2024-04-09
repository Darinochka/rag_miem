from fastapi import FastAPI, HTTPException
from langserve import add_routes

import src.retriever as retriever_utils
from src.retriever import get_reranked_docs, create_reranker
from src.servers.base_models import RetrieverArgs, RerankRequest, RerankResponse

args = RetrieverArgs()
documents = retriever_utils.create_documents(
    folder=args.data_folder,
    target_column=args.target_field,
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap,
)
vectorstore = retriever_utils.create_db(
    documents, embeddings_model=args.model_name, normalize_embeddings=True
)
retriever = vectorstore.as_retriever()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

tokenizer, model = create_reranker(args.reranker_model_name)


@app.post("/rerank", response_model=RerankResponse)
async def rerank_docs(request: RerankRequest) -> RerankResponse:
    try:
        reranked_docs = get_reranked_docs(
            query=request.query, docs=request.docs, tokenizer=tokenizer, model=model
        )
        return RerankResponse(reranked_docs=reranked_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, retriever)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.retriever_host, port=args.retriever_port)
