"""
Document indexing pipeline — ingests PDFs, chunks, and upserts to Pinecone.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.logging_config import get_logger
from config.settings import get_settings
from retrieval.vectorstore import PineconeManager

logger = get_logger("indexer")


class DocumentIndexer:
    """End-to-end document ingestion pipeline: load → chunk → embed → index."""

    def __init__(
        self,
        data_dir: str = "data",
        pinecone_manager: Optional[PineconeManager] = None,
    ) -> None:
        self._settings = get_settings()
        self._data_dir = Path(data_dir)
        self._pinecone = pinecone_manager or PineconeManager()

    def load_pdfs(self) -> List[Document]:
        """Load all PDF files from the data directory."""
        logger.info(f"Loading PDFs from {self._data_dir}")
        loader = DirectoryLoader(
            str(self._data_dir),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDFs")
        return documents

    def filter_metadata(self, docs: List[Document]) -> List[Document]:
        """Reduce metadata to essential fields only."""
        return [
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source", "unknown")},
            )
            for doc in docs
        ]

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into retrieval-optimized chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks (size={self._settings.chunk_size})")
        return chunks

    def run(self) -> int:
        """Execute the full indexing pipeline. Returns number of chunks indexed."""
        docs = self.load_pdfs()
        docs = self.filter_metadata(docs)
        chunks = self.chunk_documents(docs)
        self._pinecone.index_documents(chunks)
        logger.info(f"Indexing pipeline complete: {len(chunks)} chunks indexed")
        return len(chunks)

    def add_document(self, content: str, source: str = "manual") -> None:
        """Add a single document to the index."""
        doc = Document(page_content=content, metadata={"source": source})
        chunks = self.chunk_documents([doc])
        store = self._pinecone.get_store()
        store.add_documents(chunks)
        logger.info(f"Added 1 document ({len(chunks)} chunks) from '{source}'")
