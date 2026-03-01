"""
Legacy helper module — DEPRECATED.

Document loading, chunking, and embedding functions have been moved to
the retrieval package for better modularity:

  - retrieval/indexer.py    → DocumentIndexer (load, chunk, index)
  - retrieval/embeddings.py → EmbeddingManager (lazy-loaded singleton)
  - retrieval/vectorstore.py → PineconeManager (index lifecycle)

This module is retained for backward compatibility.
"""

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


def load_pdf_file(data):
    """Load PDF files from a directory. See retrieval.indexer.DocumentIndexer."""
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Strip metadata to essentials. See retrieval.indexer.DocumentIndexer."""
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source")},
        )
        for doc in docs
    ]


def text_split(extracted_data):
    """Split documents into chunks. See retrieval.indexer.DocumentIndexer."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings():
    """Load embedding model. See retrieval.embeddings.EmbeddingManager."""
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')