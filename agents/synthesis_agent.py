"""
Synthesis Agent — responsible for producing final cohesive answers.
"""

from __future__ import annotations

from typing import Any

from config.logging_config import get_logger
from core.modules import AnswerSynthesizer, SafetyGuard
from core.schemas import (
    ReasoningTrace,
    SafetyAssessment,
    SafetyLevel,
    VerificationResult,
)

logger = get_logger("synthesis_agent")


class SynthesisAgent:
    """
    Specialized agent for answer synthesis and safety enforcement.

    Combines reasoning traces with verification results to produce
    safe, accurate, patient-friendly medical responses.
    """

    def __init__(self) -> None:
        self.synthesizer = AnswerSynthesizer()
        self.safety_guard = SafetyGuard()

    def synthesize(
        self,
        query: str,
        reasoning_trace: ReasoningTrace,
        verification: VerificationResult,
        context: str,
    ) -> dict[str, Any]:
        """Produce a final answer with safety checks."""
        answer, key_points, confidence_summary = self.synthesizer(
            query=query,
            reasoning_trace=reasoning_trace,
            verification=verification,
            context=context,
        )

        safety = self.safety_guard(query=query, response=answer)

        # Apply safety modifications
        final_answer = self._apply_safety_modifications(answer, safety)

        return {
            "answer": final_answer,
            "key_points": key_points,
            "confidence_summary": confidence_summary,
            "safety": safety,
        }

    def _apply_safety_modifications(
        self, answer: str, safety: SafetyAssessment
    ) -> str:
        """Modify answer based on safety assessment."""
        modifications = []

        if safety.disclaimers:
            modifications.append("⚠️ " + " | ".join(safety.disclaimers))

        if safety.requires_professional_review:
            modifications.append(
                "📋 Please consult a healthcare professional for personalized medical advice."
            )

        if safety.level == SafetyLevel.UNSAFE:
            return (
                "⛔ I cannot provide specific advice on this topic. "
                "Please consult a healthcare professional immediately."
            )

        if modifications:
            return "\n".join(modifications) + "\n\n" + answer
        return answer
