"""
Trainable DSPy Modules for the Medical AI System.

Each module wraps a DSPy Signature with a forward() method,
enabling automatic prompt optimization via DSPy compilers.
Modules are composable and can be chained into pipelines.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import dspy

from config.logging_config import get_logger
from core.signatures import (
    ClinicalEntityExtraction,
    MedicalContextRetrieval,
    MedicalQueryAnalysis,
    MedicalReasoning,
    MedicalSynthesis,
    MedicalVerification,
    SafetyCheck,
    StructuredDiagnosisSignature,
)
from core.schemas import (
    ClinicalEntity,
    MedicalResponse,
    QueryAnalysisOutput,
    QueryIntent,
    ReasoningStep,
    ReasoningTrace,
    RetrievedContext,
    SafetyAssessment,
    SafetyLevel,
    StructuredDiagnosis,
    VerificationResult,
    VerificationStatus,
)

logger = get_logger("modules")


# ─── Utility ─────────────────────────────────────────────────────────────────

def _safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON robustly, stripping markdown fences if present."""
    if default is None:
        default = []
    if not text:
        return default
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return 0.5


# ─── Query Analyzer ─────────────────────────────────────────────────────────

class QueryAnalyzer(dspy.Module):
    """Analyzes incoming medical queries: intent classification,
    entity extraction, complexity scoring, and sub-question decomposition."""

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(MedicalQueryAnalysis)

    def forward(self, query: str) -> QueryAnalysisOutput:
        result = self.analyze(query=query)

        entities_raw = _safe_json_loads(result.clinical_entities, [])
        entities = []
        for e in entities_raw:
            if isinstance(e, dict):
                entities.append(
                    ClinicalEntity(
                        name=e.get("name", "unknown"),
                        entity_type=e.get("entity_type", "unknown"),
                        confidence=_clamp(e.get("confidence", 0.5)),
                        span=e.get("span"),
                    )
                )

        sub_questions = _safe_json_loads(result.sub_questions, [])

        intent_str = (result.intent or "general").strip().lower()
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.GENERAL

        return QueryAnalysisOutput(
            original_query=query,
            reformulated_query=result.reformulated_query or query,
            intent=intent,
            clinical_entities=entities,
            requires_multi_hop=bool(result.requires_multi_hop),
            complexity_score=_clamp(result.complexity_score),
            sub_questions=sub_questions if isinstance(sub_questions, list) else [],
        )


# ─── Context Ranker ──────────────────────────────────────────────────────────

class ContextRanker(dspy.Module):
    """Re-ranks retrieved passages using LLM-based relevance assessment."""

    def __init__(self) -> None:
        super().__init__()
        self.rank = dspy.ChainOfThought(MedicalContextRetrieval)

    def forward(
        self, query: str, passages: list[dict[str, str]]
    ) -> tuple[list[RetrievedContext], str, list[str]]:
        result = self.rank(
            query=query,
            passages=json.dumps(passages, ensure_ascii=False),
        )

        ranked_raw = _safe_json_loads(result.ranked_passages, [])
        ranked = []
        for p in ranked_raw:
            if isinstance(p, dict):
                ranked.append(
                    RetrievedContext(
                        content=p.get("content", ""),
                        source=p.get("source"),
                        relevance_score=_clamp(p.get("relevance_score", 0.5)),
                    )
                )

        quality = (result.retrieval_quality or "partial").strip().lower()
        additional = _safe_json_loads(result.suggested_additional_queries, [])

        return ranked, quality, additional


# ─── Medical Reasoner ────────────────────────────────────────────────────────

class MedicalReasoner(dspy.Module):
    """Performs multi-step evidence-based medical reasoning with
    chain-of-thought traces and confidence calibration."""

    def __init__(self) -> None:
        super().__init__()
        self.reason = dspy.ChainOfThought(MedicalReasoning)

    def forward(
        self,
        query: str,
        context: str,
        query_analysis: QueryAnalysisOutput,
    ) -> ReasoningTrace:
        result = self.reason(
            query=query,
            context=context,
            query_analysis=query_analysis.model_dump_json(),
        )

        steps_raw = _safe_json_loads(result.reasoning_steps, [])
        steps = []
        for i, s in enumerate(steps_raw):
            if isinstance(s, dict):
                steps.append(
                    ReasoningStep(
                        step_number=s.get("step_number", i + 1),
                        thought=s.get("thought", ""),
                        evidence=s.get("evidence"),
                        confidence=_clamp(s.get("confidence", 0.5)),
                    )
                )

        return ReasoningTrace(
            query=query,
            steps=steps,
            conclusion=result.conclusion or "",
            overall_confidence=_clamp(result.overall_confidence),
            reasoning_type=result.reasoning_type or "chain_of_thought",
        )


# ─── Fact Verifier ───────────────────────────────────────────────────────────

class FactVerifier(dspy.Module):
    """Verifies medical claims against retrieved evidence using
    rigorous evidence appraisal methodology."""

    def __init__(self) -> None:
        super().__init__()
        self.verify = dspy.ChainOfThought(MedicalVerification)

    def forward(self, claim: str, evidence: str) -> VerificationResult:
        result = self.verify(claim=claim, evidence=evidence)

        status_str = (result.status or "unverified").strip().lower()
        try:
            status = VerificationStatus(status_str)
        except ValueError:
            status = VerificationStatus.UNVERIFIED

        return VerificationResult(
            claim=claim,
            status=status,
            supporting_evidence=_safe_json_loads(result.supporting_evidence, []),
            contradicting_evidence=_safe_json_loads(result.contradicting_evidence, []),
            confidence=_clamp(result.confidence),
            reasoning=result.reasoning or "",
        )


# ─── Answer Synthesizer ─────────────────────────────────────────────────────

class AnswerSynthesizer(dspy.Module):
    """Synthesizes a final patient-friendly answer from reasoning traces,
    verification results, and retrieved context."""

    def __init__(self) -> None:
        super().__init__()
        self.synthesize = dspy.ChainOfThought(MedicalSynthesis)

    def forward(
        self,
        query: str,
        reasoning_trace: ReasoningTrace,
        verification: VerificationResult,
        context: str,
    ) -> tuple[str, list[str], str]:
        result = self.synthesize(
            query=query,
            reasoning_trace=reasoning_trace.model_dump_json(),
            verification=verification.model_dump_json(),
            context=context,
        )
        key_points = _safe_json_loads(result.key_points, [])
        return result.answer or "", key_points, result.confidence_summary or ""


# ─── Clinical Entity Extractor ───────────────────────────────────────────────

class ClinicalEntityExtractor(dspy.Module):
    """Extracts clinical entities from free text."""

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.Predict(ClinicalEntityExtraction)

    def forward(self, text: str) -> list[ClinicalEntity]:
        result = self.extract(text=text)
        entities_raw = _safe_json_loads(result.entities, [])
        entities = []
        for e in entities_raw:
            if isinstance(e, dict):
                entities.append(
                    ClinicalEntity(
                        name=e.get("name", ""),
                        entity_type=e.get("entity_type", "unknown"),
                        confidence=_clamp(e.get("confidence", 0.5)),
                        span=e.get("span"),
                    )
                )
        return entities


# ─── Safety Guard ────────────────────────────────────────────────────────────

class SafetyGuard(dspy.Module):
    """Evaluates safety of generated medical content and produces
    necessary disclaimers and warnings."""

    def __init__(self) -> None:
        super().__init__()
        self.check = dspy.ChainOfThought(SafetyCheck)

    def forward(self, query: str, response: str) -> SafetyAssessment:
        result = self.check(query=query, response=response)

        level_str = (result.safety_level or "caution").strip().lower()
        try:
            level = SafetyLevel(level_str)
        except ValueError:
            level = SafetyLevel.CAUTION

        return SafetyAssessment(
            level=level,
            flags=_safe_json_loads(result.flags, []),
            disclaimers=_safe_json_loads(result.disclaimers, []),
            requires_professional_review=bool(result.requires_professional_review),
            reasoning=result.reasoning or "",
        )


# ─── Structured Diagnosis Module ────────────────────────────────────────────

class DiagnosisModule(dspy.Module):
    """Generates structured differential diagnosis for diagnostic queries."""

    def __init__(self) -> None:
        super().__init__()
        self.diagnose = dspy.ChainOfThought(StructuredDiagnosisSignature)

    def forward(
        self,
        query: str,
        reasoning_trace: ReasoningTrace,
        context: str,
    ) -> StructuredDiagnosis:
        result = self.diagnose(
            query=query,
            reasoning_trace=reasoning_trace.model_dump_json(),
            context=context,
        )
        return StructuredDiagnosis(
            primary_condition=result.primary_condition or "Unknown",
            confidence=_clamp(result.confidence),
            differential_diagnoses=_safe_json_loads(result.differential_diagnoses, []),
            recommended_tests=_safe_json_loads(result.recommended_tests, []),
            red_flags=_safe_json_loads(result.red_flags, []),
        )


# ─── Full Medical QA Pipeline (Composite Module) ────────────────────────────

class MedicalQAPipeline(dspy.Module):
    """
    End-to-end medical QA pipeline composing all sub-modules into a
    single trainable, optimizable DSPy Module.

    Pipeline graph:
        query → QueryAnalyzer → ContextRanker → MedicalReasoner
              → FactVerifier → AnswerSynthesizer → SafetyGuard
              → (optional) DiagnosisModule → MedicalResponse
    """

    def __init__(self, retriever_fn=None) -> None:
        super().__init__()
        self.query_analyzer = QueryAnalyzer()
        self.context_ranker = ContextRanker()
        self.reasoner = MedicalReasoner()
        self.verifier = FactVerifier()
        self.synthesizer = AnswerSynthesizer()
        self.safety_guard = SafetyGuard()
        self.diagnosis_module = DiagnosisModule()
        self._retriever_fn = retriever_fn

    def set_retriever(self, retriever_fn) -> None:
        """Inject retriever function post-init (for deferred initialization)."""
        self._retriever_fn = retriever_fn

    def forward(self, query: str) -> MedicalResponse:
        logger.info("Pipeline invoked", extra={"pipeline": "medical_qa", "query": query[:100]})

        # ── Step 1: Query Analysis ──
        query_analysis = self.query_analyzer(query=query)
        search_query = query_analysis.reformulated_query

        # ── Step 2: Retrieval ──
        raw_passages = []
        if self._retriever_fn is not None:
            raw_passages = self._retriever_fn(search_query)

        # ── Step 3: Context Ranking ──
        passages_for_ranking = [
            {"content": p.get("content", ""), "source": p.get("source", "")}
            for p in raw_passages
        ]
        if passages_for_ranking:
            ranked_contexts, quality, additional_queries = self.context_ranker(
                query=search_query, passages=passages_for_ranking
            )
        else:
            ranked_contexts = []
            quality = "insufficient"
            additional_queries = []

        # Prepare flat context string for downstream modules
        context_str = "\n\n---\n\n".join(
            rc.content for rc in ranked_contexts if rc.content
        )

        # ── Step 4: Medical Reasoning ──
        reasoning_trace = self.reasoner(
            query=query,
            context=context_str,
            query_analysis=query_analysis,
        )

        # ── Step 5: Fact Verification ──
        verification = self.verifier(
            claim=reasoning_trace.conclusion,
            evidence=context_str,
        )

        # ── Step 6: Answer Synthesis ──
        answer, key_points, confidence_summary = self.synthesizer(
            query=query,
            reasoning_trace=reasoning_trace,
            verification=verification,
            context=context_str,
        )

        # ── Step 7: Safety Check ──
        safety = self.safety_guard(query=query, response=answer)

        # Prepend disclaimers if needed
        if safety.disclaimers:
            disclaimer_text = " | ".join(safety.disclaimers)
            answer = f"⚠️ {disclaimer_text}\n\n{answer}"

        # ── Step 8: Structured Diagnosis (conditional) ──
        structured_diagnosis = None
        if query_analysis.intent in (QueryIntent.DIAGNOSIS, QueryIntent.SYMPTOMS):
            structured_diagnosis = self.diagnosis_module(
                query=query,
                reasoning_trace=reasoning_trace,
                context=context_str,
            )

        # ── Assemble response ──
        return MedicalResponse(
            answer=answer,
            query_analysis=query_analysis,
            reasoning_trace=reasoning_trace,
            verification=verification,
            safety=safety,
            structured_diagnosis=structured_diagnosis,
            retrieved_contexts=ranked_contexts,
            metadata={
                "retrieval_quality": quality,
                "additional_queries_suggested": additional_queries,
                "key_points": key_points,
                "confidence_summary": confidence_summary,
            },
        )
