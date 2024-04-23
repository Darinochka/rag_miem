#!/usr/bin/env python
import logging
import os
from typing import List, Any, Optional

import pandas as pd
from langchain_community.document_loaders import (
    DataFrameLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
from src.utils.base_models import RetrieverArgs
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOADER_MAPPING = {
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


class Retriever:
    def __init__(
        self,
        documents: Optional[List[Document]],
        args: RetrieverArgs,
        db_load_folder: Optional[str] = None,
    ):
        self.args = args
        self._create_embedding_model(
            model_name=self.args.embedding_model,
            normalize_embeddings=self.args.normalize_embeddings,
        )

        if db_load_folder is None:
            self.create_db(documents)
        else:
            self.load_db(db_load_folder)

        self.retriever = self.db.as_retriever()

        if self.args.rerank_model is not None:
            self.base_retriever = self.retriever
            self.retriever = self._add_rerank(self.args.rerank_model)
            logger.info(f"Reranking added {self.retriever}")

    def _create_embedding_model(
        self, model_name: str, normalize_embeddings: bool = False
    ) -> HuggingFaceEmbeddings:
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )
        logger.info(f"Embedding model {self.embedding_model}")

    def load_db(self, folder: str) -> None:
        self.db = FAISS.load_local(
            folder, self.embedding_model, allow_dangerous_deserialization=True
        )
        logger.info(f"Db loaded {self.db}")

    def create_db(self, documents: Optional[List[Document]]) -> None:
        self.db = FAISS.from_documents(
            documents,
            self.embedding_model,
        )
        logger.info(f"Db created {self.db}")

    def save_db(self, folder: str) -> None:
        self.db.save_local(folder)
        logger.info(f"Db saved to {folder}")

    def _add_rerank(
        self, model_name: str, top_n: int = 4
    ) -> ContextualCompressionRetriever:
        model = HuggingFaceCrossEncoder(model_name=model_name)
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.base_retriever
        )


def create_documents(
    folder: str,
    target_column: str = "text",
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> Any:
    raw_docs = []
    logger.info(
        f"folder: {folder} target_column: {target_column} chunk_size: {chunk_size} chunk_overlap: {chunk_overlap}"
    )
    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        if not os.path.isfile(full_path):
            continue

        logger.info(f"Process file {filename}...")
        ext = "." + filename.rsplit(".", 1)[-1]

        if ext == ".csv":
            data = pd.read_csv(
                full_path,
                sep=",",
                quotechar='"',
            )
            loader = DataFrameLoader(data, page_content_column=target_column)
        elif ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(full_path, **loader_args)
        else:
            logger.warning(f"Unsupported file type {ext}, skipping...")
            continue
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
