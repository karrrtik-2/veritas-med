"""
Declarative DSPy Signatures for the Medical AI System.

Signatures define the *what* — the input/output contract for each
reasoning step — while modules (core/modules.py) define the *how*.

Each signature is a typed, self-documenting declaration that DSPy
compilers and optimizers use to generate, refine, and cache prompts
automatically.
"""

from __future__ import annotations

import dspy


# ─── Query Understanding ────────────────────────────────────────────────────

class MedicalQueryAnalysis(dspy.Signature):
    """Analyze a medical query to determine intent, extract clinical entities,
    assess complexity, and decompose into sub-questions if needed.
    Output a reformulated query optimized for medical knowledge retrieval."""

    query: str = dspy.InputField(desc="Raw user medical query")

    reformulated_query: str = dspy.OutputField(
        desc="Semantically enriched query optimized for medical retrieval"
    )
    intent: str = dspy.OutputField(
        desc="Query intent: diagnosis|treatment|symptoms|drug_info|prevention|general|emergency|lifestyle"
    )
    clinical_entities: str = dspy.OutputField(
        desc="JSON array of extracted clinical entities [{name, entity_type, confidence}]"
    )
    requires_multi_hop: bool = dspy.OutputField(
        desc="Whether multi-hop reasoning across multiple evidence sources is needed"
    )
    complexity_score: float = dspy.OutputField(
        desc="Complexity score from 0.0 (simple factoid) to 1.0 (complex multi-step reasoning)"
    )
    sub_questions: str = dspy.OutputField(
        desc="JSON array of decomposed sub-questions for multi-hop reasoning, or empty array"
    )


# ─── Context Retrieval Refinement ───────────────────────────────────────────

class MedicalContextRetrieval(dspy.Signature):
    """Given a medical query and retrieved context passages, assess relevance
    of each passage and produce a ranked, filtered set of contexts with
    relevance scores. Discard irrelevant or redundant passages."""

    query: str = dspy.InputField(desc="Medical query being answered")
    passages: str = dspy.InputField(
        desc="JSON array of retrieved passages [{content, source}]"
    )

    ranked_passages: str = dspy.OutputField(
        desc="JSON array of passages ranked by relevance [{content, source, relevance_score}]"
    )
    retrieval_quality: str = dspy.OutputField(
        desc="Assessment: sufficient|partial|insufficient"
    )
    suggested_additional_queries: str = dspy.OutputField(
        desc="JSON array of additional queries to fill knowledge gaps, or empty array"
    )


# ─── Medical Reasoning ──────────────────────────────────────────────────────

class MedicalReasoning(dspy.Signature):
    """Perform step-by-step medical reasoning over the query and retrieved
    context. Produce a chain-of-thought trace with evidence citations,
    confidence scores per step, and a final conclusion.

    Follow evidence-based medicine principles:
    1. Identify the clinical question
    2. Gather and appraise evidence
    3. Apply evidence to the specific case
    4. Evaluate the outcome"""

    query: str = dspy.InputField(desc="Medical query")
    context: str = dspy.InputField(desc="Retrieved and ranked context passages")
    query_analysis: str = dspy.InputField(desc="JSON query analysis from upstream")

    reasoning_steps: str = dspy.OutputField(
        desc="JSON array of reasoning steps [{step_number, thought, evidence, confidence}]"
    )
    conclusion: str = dspy.OutputField(
        desc="Final medical conclusion based on evidence-based reasoning"
    )
    overall_confidence: float = dspy.OutputField(
        desc="Overall confidence in the conclusion (0.0 to 1.0)"
    )
    reasoning_type: str = dspy.OutputField(
        desc="Type of reasoning applied: chain_of_thought|differential_diagnosis|causal|comparative"
    )


# ─── Fact Verification ──────────────────────────────────────────────────────

class MedicalVerification(dspy.Signature):
    """Verify the factual accuracy of a medical claim against provided
    evidence. Identify supporting and contradicting evidence, and
    assess the overall verification status.

    Apply rigorous evidence appraisal — distinguish between strong
    evidence (systematic reviews, RCTs) and weaker evidence (case reports)."""

    claim: str = dspy.InputField(desc="Medical claim to verify")
    evidence: str = dspy.InputField(desc="Evidence passages to verify against")

    status: str = dspy.OutputField(
        desc="Verification status: verified|partially_verified|unverified|contradicted"
    )
    supporting_evidence: str = dspy.OutputField(
        desc="JSON array of evidence strings supporting the claim"
    )
    contradicting_evidence: str = dspy.OutputField(
        desc="JSON array of evidence strings contradicting the claim"
    )
    confidence: float = dspy.OutputField(
        desc="Verification confidence (0.0 to 1.0)"
    )
    reasoning: str = dspy.OutputField(
        desc="Detailed reasoning about the verification assessment"
    )


# ─── Answer Synthesis ────────────────────────────────────────────────────────

class MedicalSynthesis(dspy.Signature):
    """Synthesize a final, patient-friendly medical answer from the reasoning
    trace, verified claims, and retrieved evidence.

    The answer must be:
    - Accurate and evidence-based
    - Clear and understandable to non-medical audiences
    - Appropriately hedged when evidence is uncertain
    - Include relevant disclaimers when necessary"""

    query: str = dspy.InputField(desc="Original medical query")
    reasoning_trace: str = dspy.InputField(desc="JSON reasoning trace from medical reasoner")
    verification: str = dspy.InputField(desc="JSON verification results")
    context: str = dspy.InputField(desc="Relevant context passages")

    answer: str = dspy.OutputField(
        desc="Final synthesized medical answer, clear and evidence-based"
    )
    key_points: str = dspy.OutputField(
        desc="JSON array of key medical points for structured display"
    )
    confidence_summary: str = dspy.OutputField(
        desc="Brief summary of confidence level and evidence quality"
    )


# ─── Clinical Entity Extraction ─────────────────────────────────────────────

class ClinicalEntityExtraction(dspy.Signature):
    """Extract clinical entities (conditions, symptoms, drugs, procedures,
    anatomical locations) from medical text with type classification
    and confidence scoring."""

    text: str = dspy.InputField(desc="Medical text to extract entities from")

    entities: str = dspy.OutputField(
        desc="JSON array of entities [{name, entity_type, confidence, span}]"
    )


# ─── Safety Check ────────────────────────────────────────────────────────────

class SafetyCheck(dspy.Signature):
    """Evaluate the safety of a medical AI response. Check for:
    - Potentially harmful medical advice
    - Missing critical disclaimers
    - Statements that require professional medical review
    - Emergency situations that need immediate attention

    Err on the side of caution — flag anything uncertain."""

    query: str = dspy.InputField(desc="Original user query")
    response: str = dspy.InputField(desc="Generated medical response to evaluate")

    safety_level: str = dspy.OutputField(
        desc="Safety level: safe|caution|unsafe|requires_professional"
    )
    flags: str = dspy.OutputField(
        desc="JSON array of safety concern flags"
    )
    disclaimers: str = dspy.OutputField(
        desc="JSON array of disclaimers to prepend/append to the response"
    )
    requires_professional_review: bool = dspy.OutputField(
        desc="Whether the response should recommend consulting a healthcare professional"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of the safety assessment"
    )


# ─── Structured Diagnosis (for diagnostic queries) ──────────────────────────

class StructuredDiagnosisSignature(dspy.Signature):
    """For diagnostic queries, produce a structured differential diagnosis
    with confidence scores, recommended tests, and red flags.
    Follow clinical decision-making frameworks."""

    query: str = dspy.InputField(desc="Diagnostic query")
    reasoning_trace: str = dspy.InputField(desc="Medical reasoning trace")
    context: str = dspy.InputField(desc="Retrieved medical context")

    primary_condition: str = dspy.OutputField(
        desc="Most likely primary condition"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in primary diagnosis (0.0 to 1.0)"
    )
    differential_diagnoses: str = dspy.OutputField(
        desc="JSON array of differential diagnoses [{condition, probability, reasoning}]"
    )
    recommended_tests: str = dspy.OutputField(
        desc="JSON array of recommended diagnostic tests"
    )
    red_flags: str = dspy.OutputField(
        desc="JSON array of red flag symptoms to watch for"
    )
