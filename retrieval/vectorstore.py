"""
Pinecone vector store management with index lifecycle operations.
"""

from __future__ import annotations

from typing import Optional

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config.logging_config import get_logger
from config.settings import get_settings
from retrieval.embeddings import get_embedding_manager

logger = get_logger("vectorstore")


class PineconeManager:
    """Manages Pinecone index lifecycle and vector store connections."""

    def __init__(self, index_name: Optional[str] = None) -> None:
        self._settings = get_settings()
        self._index_name = index_name or self._settings.pinecone_index_name
        self._client: Optional[Pinecone] = None
        self._store: Optional[PineconeVectorStore] = None

    @property
    def client(self) -> Pinecone:
        if self._client is None:
            self._client = Pinecone(api_key=self._settings.pinecone_api_key)
        return self._client

    def ensure_index(self) -> None:
        """Create index if it doesn't exist."""
        if not self.client.has_index(self._index_name):
            logger.info(f"Creating Pinecone index: {self._index_name}")
            self.client.create_index(
                name=self._index_name,
                dimension=self._settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )
            logger.info("Pinecone index created successfully")

    def get_store(self) -> PineconeVectorStore:
        """Get or create vector store connection."""
        if self._store is None:
            embeddings = get_embedding_manager().model
            self._store = PineconeVectorStore.from_existing_index(
                index_name=self._index_name,
                embedding=embeddings,
            )
        return self._store

    def get_index(self):
        """Get raw Pinecone index handle."""
        return self.client.Index(self._index_name)

    def index_documents(self, documents: list) -> PineconeVectorStore:
        """Index a list of LangChain Document objects."""
        self.ensure_index()
        embeddings = get_embedding_manager().model
        logger.info(f"Indexing {len(documents)} documents into '{self._index_name}'")
        self._store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=self._index_name,
        )
        logger.info("Document indexing complete")
        return self._store

    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        if self.client.has_index(self._index_name):
            self.client.delete_index(self._index_name)
            logger.info(f"Deleted Pinecone index: {self._index_name}")
