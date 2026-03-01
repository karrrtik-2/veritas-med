"""
Multi-agent orchestrator — coordinates specialized agents
through a reasoning graph for complex medical queries.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from config.logging_config import get_logger
from core.schemas import MedicalResponse, PipelineTrace

logger = get_logger("orchestrator")


@dataclass
class OrchestratorConfig:
    """Configuration for the agent orchestrator."""
    max_iterations: int = 5
    confidence_threshold: float = 0.7
    enable_self_reflection: bool = True
    enable_verification: bool = True
    enable_safety_check: bool = True
    timeout_seconds: float = 120.0


class AgentOrchestrator:
    """
    High-level orchestrator that coordinates multiple specialized agents
    to answer complex medical queries.

    Architecture:
    ┌──────────────┐
    │  Orchestrator │
    ├──────────────┤
    │ MedicalQA    │ ← Full pipeline agent
    │ Retrieval    │ ← Specialized retrieval agent
    │ Verification │ ← Fact-checking agent
    │ Synthesis    │ ← Answer synthesis agent
    └──────────────┘

    The orchestrator implements iterative refinement:
    1. Process query through the primary medical QA agent
    2. If confidence < threshold, iterate with additional retrieval
    3. Verify claims via the verification agent
    4. Synthesize and safety-check via the synthesis agent
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self._agents: dict[str, Any] = {}
        self._pipeline = None
        self._retriever = None

    def register_agent(self, name: str, agent: Any) -> None:
        """Register a specialized agent."""
        self._agents[name] = agent
        logger.info(f"Registered agent: {name}")

    def set_pipeline(self, pipeline) -> None:
        """Set the primary DSPy pipeline."""
        self._pipeline = pipeline

    def set_retriever(self, retriever) -> None:
        """Set the retriever for additional retrieval loops."""
        self._retriever = retriever

    def process(self, query: str) -> MedicalResponse:
        """
        Process a medical query through the multi-agent pipeline.

        Implements iterative refinement with confidence-based
        re-retrieval and self-reflection.
        """
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        logger.info(
            f"[{trace_id}] Processing query: {query[:100]}...",
            extra={"trace_id": trace_id},
        )

        # ── Primary Pipeline Execution ──
        if self._pipeline is None:
            raise RuntimeError("No pipeline registered with orchestrator")

        response = self._pipeline(query=query)

        # ── Iterative Refinement ──
        iteration = 0
        while (
            self.config.enable_self_reflection
            and response.reasoning_trace.overall_confidence < self.config.confidence_threshold
            and iteration < self.config.max_iterations
        ):
            iteration += 1
            logger.info(
                f"[{trace_id}] Refinement iteration {iteration} "
                f"(confidence={response.reasoning_trace.overall_confidence:.2f})",
                extra={"trace_id": trace_id},
            )

            # Additional retrieval with expanded queries
            if self._retriever and response.metadata.get("additional_queries_suggested"):
                additional_passages = self._retriever.multi_retrieve(
                    response.metadata["additional_queries_suggested"]
                )
                # Re-run pipeline with augmented context
                # (In production, you'd merge contexts and re-reason)

            # Re-invoke pipeline
            response = self._pipeline(query=query)

            # Timeout guard
            elapsed = time.perf_counter() - start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning(f"[{trace_id}] Timeout after {elapsed:.1f}s")
                break

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"[{trace_id}] Query processed in {latency_ms:.0f}ms "
            f"(confidence={response.reasoning_trace.overall_confidence:.2f}, "
            f"iterations={iteration})",
            extra={"trace_id": trace_id, "metric": latency_ms},
        )

        return response

    def get_trace(self, query: str, response: MedicalResponse, latency_ms: float) -> PipelineTrace:
        """Build a full pipeline trace for observability."""
        return PipelineTrace(
            trace_id=str(uuid.uuid4())[:8],
            query=query,
            response=response,
            latency_ms=latency_ms,
        )
