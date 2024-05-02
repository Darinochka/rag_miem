import pandas as pd
import argparse
import logging
from src.utils.retriever import create_documents, Retriever
from src.utils.base_models import RetrieverArgs
from typing import List, Any, Callable
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_relevant_doc_retriever(question: str, retriever: Retriever) -> List[Any]:
    return retriever.retriever.get_relevant_documents(question)


def get_relevant_doc_host(question: str, host: str) -> List[Any]:
    inputs = {"input": question}
    response = requests.post(host, json=inputs)
    logging.info(f"Request to {host} returned status code {response.status_code}")
    return response.json()["output"]


def process_csv(
    input_csv: str,
    output_csv: str,
    retriever_func: Callable[[str, Any], List[Any]],
    *args: Any,
) -> None:
    logging.info("Starting to process CSV file.")
    df = pd.read_csv(input_csv)
    results = []
    for question in df.iloc[:, 0]:
        logging.info(f"Retrieving documents for question: {question}")
        documents = retriever_func(question, *args)
        documents_contents = [doc["page_content"] for doc in documents]
        results.append([question] + documents_contents)
    results_df = pd.DataFrame(
        results, columns=["Вопрос", "Документ1", "Документ2", "Документ3", "Документ4"]
    )
    results_df.to_csv(output_csv, index=False)
    logging.info("CSV processing completed and results saved.")


def main() -> None:
    logging.info("Script started.")
    parser = argparse.ArgumentParser(
        description="Processing questions and retrieving documents."
    )
    parser.add_argument(
        "--input_csv", required=True, help="Path to the input CSV file with questions"
    )
    parser.add_argument(
        "--output_csv", required=True, help="Path to save the results in a CSV file"
    )
    parser.add_argument("--embedding_model", help="Model for embedding")
    parser.add_argument(
        "--rerank_model", type=str, default=None, help="Model for reranking"
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=bool,
        default=False,
        help="Whether to normalize embeddings",
    )
    parser.add_argument(
        "--folder", type=str, default="/data", help="Folder with documents"
    )
    parser.add_argument(
        "--target_column", type=str, default="text", help="Target column of documents"
    )
    parser.add_argument("--chunk_size", type=int, default=700, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host with retriever. For example, http://localhost:8080/invoke",
    )

    args = parser.parse_args()

    logging.info(
        f"Arguments parsed: {args} Creating documents and initializing Retriever."
    )
    if not args.host:
        retriever_args = RetrieverArgs(
            embedding_model=args.embedding_model,
            rerank_model=args.rerank_model,
            normalize_embeddings=args.normalize_embeddings,
        )
        documents = create_documents(
            folder=args.folder,
            target_column=args.target_column,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        retriever = Retriever(documents=documents, args=retriever_args)
        process_csv(
            args.input_csv, args.output_csv, get_relevant_doc_retriever, retriever
        )
    else:
        process_csv(args.input_csv, args.output_csv, get_relevant_doc_host, args.host)
    logging.info("Script completed.")


if __name__ == "__main__":
    main()
