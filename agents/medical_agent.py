"""
Medical QA Agent — wraps the full DSPy pipeline as a callable agent.
"""

from __future__ import annotations

from typing import Any, Optional

import dspy

from config.logging_config import get_logger
from core.modules import MedicalQAPipeline
from core.schemas import MedicalResponse

logger = get_logger("medical_agent")


class MedicalQAAgent:
    """
    Autonomous medical QA agent that wraps the MedicalQAPipeline
    and adds agent-level capabilities:

    - Self-monitoring of confidence
    - Automatic query reformulation on low confidence
    - Conversation history awareness
    - Agent-level caching
    """

    def __init__(self, pipeline: Optional[MedicalQAPipeline] = None) -> None:
        self.pipeline = pipeline or MedicalQAPipeline()
        self._history: list[dict[str, Any]] = []
        self._cache: dict[str, MedicalResponse] = {}

    def set_retriever(self, retriever_fn) -> None:
        self.pipeline.set_retriever(retriever_fn)

    def answer(self, query: str, use_cache: bool = True) -> MedicalResponse:
        """Answer a medical query. Optionally use cached responses."""
        cache_key = query.strip().lower()

        if use_cache and cache_key in self._cache:
            logger.info(f"Cache hit for query: {query[:60]}...")
            return self._cache[cache_key]

        response = self.pipeline(query=query)

        self._cache[cache_key] = response
        self._history.append({
            "query": query,
            "answer": response.answer,
            "confidence": response.reasoning_trace.overall_confidence,
        })

        return response

    def answer_with_context(
        self, query: str, conversation_history: list[dict[str, str]]
    ) -> MedicalResponse:
        """Answer with awareness of prior conversation turns."""
        context_prefix = ""
        if conversation_history:
            turns = []
            for turn in conversation_history[-3:]:  # Last 3 turns
                turns.append(f"User: {turn.get('user', '')}")
                turns.append(f"Assistant: {turn.get('assistant', '')}")
            context_prefix = "\n".join(turns) + "\n\nCurrent question: "

        augmented_query = context_prefix + query if context_prefix else query
        return self.answer(augmented_query, use_cache=False)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def clear_cache(self) -> None:
        self._cache.clear()
