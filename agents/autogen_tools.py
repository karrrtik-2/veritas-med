"""
AutoGen Tool Wrappers — Exposes the existing LangChain/Pinecone/DSPy
RAG pipeline as callable tools for AutoGen agents.
"""

from __future__ import annotations

from typing import Annotated

from config.logging_config import get_logger
from config.settings import get_settings
from retrieval.retriever import MedicalRetriever
from retrieval.vectorstore import PineconeManager

logger = get_logger("autogen_tools")

# ── Singleton retriever (lazy-initialized) ────────────────────────────────────
_retriever: MedicalRetriever | None = None


def _get_retriever() -> MedicalRetriever:
    """Return (or create) a shared MedicalRetriever instance."""
    global _retriever
    if _retriever is None:
        pinecone_mgr = PineconeManager()
        _retriever = MedicalRetriever(pinecone_manager=pinecone_mgr)
        logger.info("Initialized shared MedicalRetriever for AutoGen tools")
    return _retriever


def set_retriever(retriever: MedicalRetriever) -> None:
    """Allow the server lifespan to inject an already-initialized retriever."""
    global _retriever
    _retriever = retriever
    logger.info("Injected existing MedicalRetriever into AutoGen tools")


# ─── Tool Functions ───────────────────────────────────────────────────────────


def search_medical_database(
    query: Annotated[str, "The medical query or symptoms to search for in the knowledge base"],
) -> str:
    """
    Search the Pinecone medical knowledge base using the existing
    LangChain/DSPy RAG retrieval pipeline.

    Returns relevant medical passages formatted as numbered results.
    """
    retriever = _get_retriever()
    results = retriever.retrieve(query)

    if not results:
        return "No relevant medical information found for the given query."

    formatted_parts: list[str] = []
    for idx, passage in enumerate(results, 1):
        source = passage.get("source", "unknown")
        content = passage.get("content", "")
        formatted_parts.append(
            f"[Result {idx}] (source: {source})\n{content}"
        )

    output = "\n\n".join(formatted_parts)
    logger.info(f"search_medical_database returned {len(results)} results for: {query[:80]}")
    return output


def search_medical_database_multi(
    queries: Annotated[
        list[str],
        "A list of related medical queries to search in parallel with deduplication",
    ],
) -> str:
    """
    Run multiple queries against the Pinecone medical knowledge base
    with automatic deduplication of overlapping passages.
    """
    retriever = _get_retriever()
    results = retriever.multi_retrieve(queries)

    if not results:
        return "No relevant medical information found for the given queries."

    formatted_parts: list[str] = []
    for idx, passage in enumerate(results, 1):
        source = passage.get("source", "unknown")
        content = passage.get("content", "")
        formatted_parts.append(
            f"[Result {idx}] (source: {source})\n{content}"
        )

    output = "\n\n".join(formatted_parts)
    logger.info(
        f"search_medical_database_multi returned {len(results)} results "
        f"for {len(queries)} queries"
    )
    return output
