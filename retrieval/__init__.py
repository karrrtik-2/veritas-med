"""Retrieval augmentation layer for DSPy-native medical knowledge retrieval."""

from retrieval.embeddings import EmbeddingManager
from retrieval.vectorstore import PineconeManager
from retrieval.indexer import DocumentIndexer
from retrieval.retriever import MedicalRetriever

__all__ = [
    "EmbeddingManager",
    "PineconeManager",
    "DocumentIndexer",
    "MedicalRetriever",
]
