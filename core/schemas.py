"""
Pydantic schemas for structured JSON outputs with schema enforcement.

Every DSPy module output is validated against these schemas to ensure
type safety, completeness, and downstream compatibility.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    SYMPTOMS = "symptoms"
    DRUG_INFO = "drug_info"
    PREVENTION = "prevention"
    GENERAL = "general"
    EMERGENCY = "emergency"
    LIFESTYLE = "lifestyle"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


class SafetyLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    REQUIRES_PROFESSIONAL = "requires_professional"


# ─── Core Output Schemas ─────────────────────────────────────────────────────

class ClinicalEntity(BaseModel):
    """Extracted clinical entity from text."""
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type (condition, drug, symptom, procedure)")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    span: Optional[str] = Field(None, description="Original text span")

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


class QueryAnalysisOutput(BaseModel):
    """Structured output from query analysis."""
    original_query: str
    reformulated_query: str = Field(..., description="Optimized query for retrieval")
    intent: QueryIntent
    clinical_entities: list[ClinicalEntity] = Field(default_factory=list)
    requires_multi_hop: bool = Field(False, description="Whether multi-hop reasoning is needed")
    complexity_score: float = Field(ge=0.0, le=1.0)
    sub_questions: list[str] = Field(default_factory=list, description="Decomposed sub-questions")


class RetrievedContext(BaseModel):
    """A single retrieved context chunk with metadata."""
    content: str
    source: Optional[str] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    chunk_id: Optional[str] = None


class ReasoningStep(BaseModel):
    """A single step in a chain-of-thought reasoning trace."""
    step_number: int
    thought: str
    evidence: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class ReasoningTrace(BaseModel):
    """Complete reasoning trace for a medical query."""
    query: str
    steps: list[ReasoningStep]
    conclusion: str
    overall_confidence: float = Field(ge=0.0, le=1.0)
    reasoning_type: str = Field("chain_of_thought", description="Type of reasoning applied")


class VerificationResult(BaseModel):
    """Result of fact verification against evidence."""
    claim: str
    status: VerificationStatus
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class SafetyAssessment(BaseModel):
    """Safety evaluation of generated medical content."""
    level: SafetyLevel
    flags: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
    requires_professional_review: bool = False
    reasoning: str


class StructuredDiagnosis(BaseModel):
    """Structured diagnosis output with differential diagnoses."""
    primary_condition: str
    confidence: float = Field(ge=0.0, le=1.0)
    differential_diagnoses: list[dict[str, Any]] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    icd_codes: list[str] = Field(default_factory=list)


class MedicalResponse(BaseModel):
    """Final structured response from the medical AI pipeline."""
    answer: str
    query_analysis: QueryAnalysisOutput
    reasoning_trace: ReasoningTrace
    verification: VerificationResult
    safety: SafetyAssessment
    structured_diagnosis: Optional[StructuredDiagnosis] = None
    retrieved_contexts: list[RetrievedContext] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.safety.level in (SafetyLevel.SAFE, SafetyLevel.CAUTION)


class PipelineTrace(BaseModel):
    """Full observability trace for a single pipeline execution."""
    trace_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    query: str
    response: MedicalResponse
    latency_ms: float
    tokens_used: int = 0
    optimizer_version: Optional[str] = None
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
