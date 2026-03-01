"""
Retrieval Agent — specialized agent for adaptive knowledge retrieval.
"""

from __future__ import annotations

from typing import Any, Optional

from config.logging_config import get_logger
from core.modules import QueryAnalyzer
from retrieval.retriever import MedicalRetriever

logger = get_logger("retrieval_agent")


class RetrievalAgent:
    """
    Specialized agent for adaptive multi-strategy retrieval.

    Strategies:
    1. Direct query retrieval
    2. Reformulated query retrieval
    3. Sub-question fan-out retrieval
    4. Entity-based retrieval
    """

    def __init__(
        self,
        retriever: Optional[MedicalRetriever] = None,
    ) -> None:
        self.retriever = retriever or MedicalRetriever()
        self.query_analyzer = QueryAnalyzer()

    def retrieve_adaptive(self, query: str) -> list[dict[str, Any]]:
        """Run adaptive retrieval — analyze query then choose strategy."""
        analysis = self.query_analyzer(query=query)

        all_passages: list[dict[str, Any]] = []
        seen: set[int] = set()

        def _dedupe_add(passages: list[dict[str, Any]]) -> None:
            for p in passages:
                h = hash(p.get("content", "")[:200])
                if h not in seen:
                    seen.add(h)
                    all_passages.append(p)

        # Strategy 1: Reformulated query
        _dedupe_add(self.retriever.retrieve(analysis.reformulated_query))

        # Strategy 2: Original query (catches direct matches)
        if analysis.reformulated_query != query:
            _dedupe_add(self.retriever.retrieve(query))

        # Strategy 3: Sub-question fan-out
        if analysis.requires_multi_hop and analysis.sub_questions:
            for sub_q in analysis.sub_questions[:3]:
                _dedupe_add(self.retriever.retrieve(sub_q))

        # Strategy 4: Entity-based retrieval
        for entity in analysis.clinical_entities[:2]:
            _dedupe_add(self.retriever.retrieve(entity.name))

        logger.info(
            f"Adaptive retrieval: {len(all_passages)} unique passages "
            f"(strategies: reformulated + original"
            f"{' + sub-questions' if analysis.requires_multi_hop else ''}"
            f"{' + entities' if analysis.clinical_entities else ''})"
        )

        return all_passages
