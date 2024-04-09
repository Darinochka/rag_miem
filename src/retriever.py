#!/usr/bin/env python
import logging
import os
from typing import List, Any, Tuple, Dict

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_documents(
    folder: str,
    target_column: str = "text",
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> Any:
    raw_docs = []
    for filename in os.listdir(folder):
        csv_path = os.path.join(folder, filename)
        if not os.path.isfile(csv_path):
            continue

        logger.info(f"Process file {filename}...")
        data = pd.read_csv(
            csv_path,
            sep=",",
            quotechar='"',
            # names=["title", "date", "id", "raw_text", "text", "type", "url"],
        )
        loader = DataFrameLoader(data, page_content_column=target_column)
        raw_docs.extend(loader.load())

    logger.info(f"Len of the documents before splitting {len(raw_docs)}")
    text_splitter = RecursiveCharacterTextSplitter(
        length_function=len,
        is_separator_regex=False,
        chunk_size=chunk_size,  # min: 50, max: 2000
        chunk_overlap=chunk_overlap,  # min: 0, max: 500,
    )
    documents = text_splitter.split_documents(raw_docs)
    logger.info(f"Len of the documents after splitting {len(documents)}")

    return documents


def create_db(
    documents: List[Document],
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    normalize_embeddings: bool = False,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model,
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )
    logger.info(f"Embedding model {embeddings}")

    db = Chroma.from_documents(
        documents,
        embeddings,
        # persist_directory="chroma",
        # url=host,
    )
    logger.info(f"Db created f{db}")

    return db


def get_similar_docs(query: str, db: Chroma) -> Any:
    docs = db.similarity_search(query)
    return docs


def create_reranker(
    reranker_model: str,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(reranker_model)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_model)
    model.eval()

    return tokenizer, model


def get_reranked_docs(
    query: str,
    docs: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    max_length: int = 512,
) -> List[Dict[str, Any]]:
    pairs = [[query, doc["page_content"]] for doc in docs]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        scores = model(**inputs, return_dict=True).logits.squeeze().tolist()

    for doc, score in zip(docs, scores):
        doc["score"] = score

    # Sort documents by their scores in descending order
    sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)

    return sorted_docs
