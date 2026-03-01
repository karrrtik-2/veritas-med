"""
Embedding model management with caching and lazy initialization.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain.embeddings import HuggingFaceEmbeddings

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger("embeddings")


class EmbeddingManager:
    """Manages embedding model lifecycle with lazy loading and caching."""

    _instance: Optional["EmbeddingManager"] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None

    def __new__(cls) -> "EmbeddingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            settings = get_settings()
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model
            )
            logger.info("Embedding model loaded successfully")
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    @property
    def dimension(self) -> int:
        return get_settings().embedding_dimension


@lru_cache(maxsize=1)
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()
