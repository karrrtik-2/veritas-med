"""
Core DSPy components for the Medical AI System.

Provides declarative signatures, trainable modules,
reasoning graphs, and structured output schemas.
"""

from core.signatures import (
    MedicalQueryAnalysis,
    MedicalContextRetrieval,
    MedicalReasoning,
    MedicalVerification,
    MedicalSynthesis,
    ClinicalEntityExtraction,
    SafetyCheck,
)
from core.schemas import (
    QueryAnalysisOutput,
    RetrievedContext,
    ReasoningTrace,
    VerificationResult,
    MedicalResponse,
    ClinicalEntity,
    SafetyAssessment,
    StructuredDiagnosis,
    PipelineTrace,
)
from core.modules import (
    QueryAnalyzer,
    MedicalReasoner,
    FactVerifier,
    AnswerSynthesizer,
    ClinicalEntityExtractor,
    SafetyGuard,
    MedicalQAPipeline,
)
from core.reasoning import (
    ReasoningGraph,
    GraphNode,
    ReasoningEdge,
    MedicalReasoningGraph,
)

__all__ = [
    # Signatures
    "MedicalQueryAnalysis",
    "MedicalContextRetrieval",
    "MedicalReasoning",
    "MedicalVerification",
    "MedicalSynthesis",
    "ClinicalEntityExtraction",
    "SafetyCheck",
    # Schemas
    "QueryAnalysisOutput",
    "RetrievedContext",
    "ReasoningTrace",
    "VerificationResult",
    "MedicalResponse",
    "ClinicalEntity",
    "SafetyAssessment",
    "StructuredDiagnosis",
    "PipelineTrace",
    # Modules
    "QueryAnalyzer",
    "MedicalReasoner",
    "FactVerifier",
    "AnswerSynthesizer",
    "ClinicalEntityExtractor",
    "SafetyGuard",
    "MedicalQAPipeline",
    # Reasoning
    "ReasoningGraph",
    "GraphNode",
    "ReasoningEdge",
    "MedicalReasoningGraph",
]
