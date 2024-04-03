#!/usr/bin/env python
import logging
import os
from typing import List, Any

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document

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


def create_hf_embeddings_model(
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def create_db(
    documents: List[Document],
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
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
