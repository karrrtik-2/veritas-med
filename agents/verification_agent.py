"""
Verification Agent — specialized agent for medical fact-checking.
"""

from __future__ import annotations

from typing import Any

from config.logging_config import get_logger
from core.modules import FactVerifier
from core.schemas import VerificationResult, VerificationStatus

logger = get_logger("verification_agent")


class VerificationAgent:
    """
    Specialized agent for verifying medical claims.

    Performs multi-claim verification by splitting composite claims
    and verifying each independently against evidence.
    """

    def __init__(self) -> None:
        self.verifier = FactVerifier()

    def verify_response(
        self, response_text: str, evidence: str
    ) -> list[VerificationResult]:
        """Verify all claims in a response against evidence.

        Splits response into sentences and verifies each as a claim.
        """
        # Simple sentence splitting for claim extraction
        sentences = [
            s.strip()
            for s in response_text.replace("\n", " ").split(".")
            if len(s.strip()) > 20
        ]

        results: list[VerificationResult] = []
        for sentence in sentences:
            result = self.verifier(claim=sentence, evidence=evidence)
            results.append(result)

        self._log_summary(results)
        return results

    def verify_single_claim(self, claim: str, evidence: str) -> VerificationResult:
        """Verify a single medical claim."""
        return self.verifier(claim=claim, evidence=evidence)

    def _log_summary(self, results: list[VerificationResult]) -> None:
        counts = {}
        for r in results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        logger.info(f"Verification summary: {counts}")

    @staticmethod
    def aggregate_confidence(results: list[VerificationResult]) -> float:
        """Compute aggregate verification confidence."""
        if not results:
            return 0.0
        return sum(r.confidence for r in results) / len(results)

    @staticmethod
    def has_contradictions(results: list[VerificationResult]) -> bool:
        """Check if any claims were contradicted."""
        return any(r.status == VerificationStatus.CONTRADICTED for r in results)
