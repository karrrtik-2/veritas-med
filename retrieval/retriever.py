"""
DSPy-compatible medical retriever with multi-query expansion and re-ranking.
"""

from __future__ import annotations

from typing import Any, Optional

from config.logging_config import get_logger
from config.settings import get_settings
from retrieval.vectorstore import PineconeManager

logger = get_logger("retriever")


class MedicalRetriever:
    """
    DSPy-compatible retriever that bridges Pinecone vector search
    with the DSPy module ecosystem.

    Supports:
    - Single-query retrieval
    - Multi-query fan-out (for sub-questions)
    - Configurable top-k
    - Deduplication across multiple retrievals
    """

    def __init__(
        self,
        pinecone_manager: Optional[PineconeManager] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self._settings = get_settings()
        self._pinecone = pinecone_manager or PineconeManager()
        self._top_k = top_k or self._settings.retrieval_top_k

    def _get_retriever(self):
        """Get LangChain retriever from Pinecone store."""
        store = self._pinecone.get_store()
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._top_k},
        )

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve passages for a single query. Returns list of dicts."""
        retriever = self._get_retriever()
        docs = retriever.invoke(query)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            })

        logger.info(f"Retrieved {len(results)} passages for query: {query[:80]}...")
        return results

    def multi_retrieve(self, queries: list[str]) -> list[dict[str, Any]]:
        """Retrieve passages for multiple queries with deduplication."""
        seen_contents: set[str] = set()
        all_results: list[dict[str, Any]] = []

        for query in queries:
            passages = self.retrieve(query)
            for p in passages:
                content_hash = hash(p["content"][:200])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_results.append(p)

        logger.info(
            f"Multi-retrieve: {len(queries)} queries → {len(all_results)} unique passages"
        )
        return all_results

    def __call__(self, query: str) -> list[dict[str, Any]]:
        """Make retriever callable for injection into DSPy pipelines."""
        return self.retrieve(query)
