"""
Programmatic Reasoning Graphs for the Medical AI System.

Implements directed acyclic graph (DAG) execution of DSPy modules
with conditional branching, parallel fan-out, and convergence.
This enables complex multi-agent reasoning topologies beyond
simple linear pipelines.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from config.logging_config import get_logger

logger = get_logger("reasoning_graph")


class NodeType(str, Enum):
    TRANSFORM = "transform"
    BRANCH = "branch"
    MERGE = "merge"
    PARALLEL = "parallel"
    TERMINAL = "terminal"


@dataclass
class GraphNode:
    """A single node in the reasoning graph."""
    name: str
    node_type: NodeType = NodeType.TRANSFORM
    module_fn: Optional[Callable[..., Any]] = None
    condition_fn: Optional[Callable[[dict], bool]] = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute this node's module function against the current state."""
        if self.module_fn is None:
            return state
        start = time.perf_counter()
        result = self.module_fn(state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Node '{self.name}' executed in {elapsed_ms:.1f}ms",
            extra={"pipeline": "reasoning_graph", "agent": self.name},
        )
        if isinstance(result, dict):
            state.update(result)
        return state


@dataclass
class ReasoningEdge:
    """Directed edge between two nodes in the reasoning graph."""
    source: str
    target: str
    condition_fn: Optional[Callable[[dict], bool]] = None
    label: str = ""


class ReasoningGraph:
    """
    A DAG-based reasoning graph that orchestrates DSPy module execution.

    Supports:
    - Sequential execution
    - Conditional branching
    - Parallel fan-out and merge
    - State passing between nodes
    - Execution tracing

    Usage:
        graph = ReasoningGraph("medical_qa")
        graph.add_node(GraphNode("analyze", module_fn=analyze_fn))
        graph.add_node(GraphNode("retrieve", module_fn=retrieve_fn))
        graph.add_edge(ReasoningEdge("analyze", "retrieve"))
        result = graph.execute({"query": "What is diabetes?"})
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[ReasoningEdge] = []
        self._adjacency: dict[str, list[ReasoningEdge]] = {}
        self._execution_trace: list[dict[str, Any]] = []

    def add_node(self, node: GraphNode) -> "ReasoningGraph":
        """Add a node to the graph. Returns self for chaining."""
        self._nodes[node.name] = node
        if node.name not in self._adjacency:
            self._adjacency[node.name] = []
        return self

    def add_edge(self, edge: ReasoningEdge) -> "ReasoningGraph":
        """Add a directed edge. Returns self for chaining."""
        self._edges.append(edge)
        if edge.source not in self._adjacency:
            self._adjacency[edge.source] = []
        self._adjacency[edge.source].append(edge)
        return self

    def _find_entry_nodes(self) -> list[str]:
        """Find nodes with no incoming edges."""
        targets = {e.target for e in self._edges}
        return [name for name in self._nodes if name not in targets]

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm for topological ordering."""
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        for edge in self._edges:
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for edge in self._adjacency.get(node, []):
                if edge.target in in_degree:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)

        if len(order) != len(self._nodes):
            raise ValueError(f"Cycle detected in reasoning graph '{self.name}'")

        return order

    def execute(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Execute the full reasoning graph synchronously."""
        self._execution_trace = []
        state = dict(initial_state)
        execution_order = self._topological_sort()

        logger.info(
            f"Executing reasoning graph '{self.name}' with {len(execution_order)} nodes",
            extra={"pipeline": "reasoning_graph"},
        )

        for node_name in execution_order:
            node = self._nodes[node_name]

            # Check incoming edge conditions
            incoming_edges = [e for e in self._edges if e.target == node_name]
            should_execute = True
            if incoming_edges:
                for edge in incoming_edges:
                    if edge.condition_fn is not None and not edge.condition_fn(state):
                        should_execute = False
                        break

            if not should_execute:
                logger.info(f"Skipping node '{node_name}' — condition not met")
                self._execution_trace.append({
                    "node": node_name,
                    "status": "skipped",
                    "reason": "condition_not_met",
                })
                continue

            start = time.perf_counter()
            try:
                state = node.execute(state)
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._execution_trace.append({
                    "node": node_name,
                    "status": "success",
                    "latency_ms": elapsed_ms,
                })
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"Node '{node_name}' failed: {exc}",
                    extra={"pipeline": "reasoning_graph"},
                )
                self._execution_trace.append({
                    "node": node_name,
                    "status": "error",
                    "error": str(exc),
                    "latency_ms": elapsed_ms,
                })
                state["_error"] = str(exc)
                state["_failed_node"] = node_name

        state["_execution_trace"] = self._execution_trace
        return state

    async def execute_async(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Execute graph with async support for parallel nodes."""
        # For simplicity, delegates to sync; extend for true async fan-out.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, initial_state)

    @property
    def execution_trace(self) -> list[dict[str, Any]]:
        return list(self._execution_trace)

    def visualize(self) -> str:
        """Generate a Mermaid diagram of the reasoning graph."""
        lines = ["graph TD"]
        for node_name, node in self._nodes.items():
            label = f"{node_name}[{node.description or node_name}]"
            lines.append(f"    {label}")
        for edge in self._edges:
            arrow = f"    {edge.source} -->|{edge.label}| {edge.target}" if edge.label else f"    {edge.source} --> {edge.target}"
            lines.append(arrow)
        return "\n".join(lines)


# ─── Pre-built Medical Reasoning Graph ───────────────────────────────────────

class MedicalReasoningGraph(ReasoningGraph):
    """
    Pre-configured reasoning graph for the medical QA pipeline.

    Topology:
        query_analysis → retrieval → context_ranking → reasoning
                                                          ↓
                          synthesis ← verification ← reasoning
                              ↓
                         safety_check → [diagnosis_branch] → output
    """

    def __init__(self, pipeline) -> None:
        super().__init__(
            name="medical_reasoning_graph",
            description="Multi-step medical reasoning with verification and safety",
        )
        self._pipeline = pipeline
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct the reasoning DAG from pipeline modules."""

        # Node: Query Analysis
        def _analyze(state: dict) -> dict:
            qa = self._pipeline.query_analyzer(query=state["query"])
            return {"query_analysis": qa, "search_query": qa.reformulated_query}

        # Node: Retrieval
        def _retrieve(state: dict) -> dict:
            raw = []
            if self._pipeline._retriever_fn:
                raw = self._pipeline._retriever_fn(state["search_query"])
            return {"raw_passages": raw}

        # Node: Context Ranking
        def _rank(state: dict) -> dict:
            passages = [
                {"content": p.get("content", ""), "source": p.get("source", "")}
                for p in state.get("raw_passages", [])
            ]
            if passages:
                ranked, quality, additional = self._pipeline.context_ranker(
                    query=state["search_query"], passages=passages
                )
            else:
                ranked, quality, additional = [], "insufficient", []
            context_str = "\n\n---\n\n".join(
                rc.content for rc in ranked if rc.content
            )
            return {
                "ranked_contexts": ranked,
                "context_str": context_str,
                "retrieval_quality": quality,
                "additional_queries": additional,
            }

        # Node: Reasoning
        def _reason(state: dict) -> dict:
            trace = self._pipeline.reasoner(
                query=state["query"],
                context=state["context_str"],
                query_analysis=state["query_analysis"],
            )
            return {"reasoning_trace": trace}

        # Node: Verification
        def _verify(state: dict) -> dict:
            v = self._pipeline.verifier(
                claim=state["reasoning_trace"].conclusion,
                evidence=state["context_str"],
            )
            return {"verification": v}

        # Node: Synthesis
        def _synthesize(state: dict) -> dict:
            answer, key_points, conf = self._pipeline.synthesizer(
                query=state["query"],
                reasoning_trace=state["reasoning_trace"],
                verification=state["verification"],
                context=state["context_str"],
            )
            return {"answer": answer, "key_points": key_points, "confidence_summary": conf}

        # Node: Safety
        def _safety(state: dict) -> dict:
            s = self._pipeline.safety_guard(query=state["query"], response=state["answer"])
            if s.disclaimers:
                state["answer"] = f"⚠️ {' | '.join(s.disclaimers)}\n\n{state['answer']}"
            return {"safety": s}

        # Node: Diagnosis (conditional)
        def _diagnose(state: dict) -> dict:
            from core.schemas import QueryIntent
            qa = state["query_analysis"]
            if qa.intent in (QueryIntent.DIAGNOSIS, QueryIntent.SYMPTOMS):
                diag = self._pipeline.diagnosis_module(
                    query=state["query"],
                    reasoning_trace=state["reasoning_trace"],
                    context=state["context_str"],
                )
                return {"structured_diagnosis": diag}
            return {"structured_diagnosis": None}

        # Build nodes
        self.add_node(GraphNode("analyze", NodeType.TRANSFORM, _analyze, description="Query Analysis"))
        self.add_node(GraphNode("retrieve", NodeType.TRANSFORM, _retrieve, description="Retrieval"))
        self.add_node(GraphNode("rank", NodeType.TRANSFORM, _rank, description="Context Ranking"))
        self.add_node(GraphNode("reason", NodeType.TRANSFORM, _reason, description="Medical Reasoning"))
        self.add_node(GraphNode("verify", NodeType.TRANSFORM, _verify, description="Fact Verification"))
        self.add_node(GraphNode("synthesize", NodeType.TRANSFORM, _synthesize, description="Answer Synthesis"))
        self.add_node(GraphNode("safety", NodeType.TRANSFORM, _safety, description="Safety Check"))
        self.add_node(GraphNode("diagnose", NodeType.BRANCH, _diagnose, description="Diagnosis Branch"))

        # Build edges
        self.add_edge(ReasoningEdge("analyze", "retrieve"))
        self.add_edge(ReasoningEdge("retrieve", "rank"))
        self.add_edge(ReasoningEdge("rank", "reason"))
        self.add_edge(ReasoningEdge("reason", "verify"))
        self.add_edge(ReasoningEdge("verify", "synthesize"))
        self.add_edge(ReasoningEdge("synthesize", "safety"))
        self.add_edge(ReasoningEdge("safety", "diagnose"))
