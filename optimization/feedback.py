"""
Feedback loop manager for continuous self-improvement.

Collects user feedback, evaluation results, and system metrics
to drive iterative pipeline optimization.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.logging_config import get_logger

logger = get_logger("feedback")


@dataclass
class FeedbackEntry:
    """A single feedback entry from a user or evaluator."""
    query: str
    response: str
    rating: float  # 0.0 to 1.0
    feedback_text: str = ""
    feedback_type: str = "user"  # user|evaluator|automatic
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "feedback_type": self.feedback_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class FeedbackLoop:
    """
    Manages feedback collection, storage, and conversion to
    training examples for optimization.

    Feedback Flow:
    1. Collect feedback entries (user ratings, evaluator scores)
    2. Persist to disk for durability
    3. Convert high-quality entries to training examples
    4. Trigger re-optimization when enough feedback accumulates

    This enables continuous self-improvement — the system gets
    better with each interaction.
    """

    def __init__(
        self,
        storage_path: str = "feedback_data",
        optimization_threshold: int = 20,
    ) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._entries: list[FeedbackEntry] = []
        self._optimization_threshold = optimization_threshold
        self._optimization_callback: Optional[Any] = None

    def add_feedback(self, entry: FeedbackEntry) -> None:
        """Add a feedback entry."""
        self._entries.append(entry)
        self._persist_entry(entry)

        logger.info(
            f"Feedback added: rating={entry.rating:.2f}, "
            f"type={entry.feedback_type}, "
            f"total_entries={len(self._entries)}"
        )

        # Check if re-optimization should be triggered
        if len(self._entries) >= self._optimization_threshold:
            self._maybe_trigger_optimization()

    def add_user_feedback(
        self, query: str, response: str, rating: float, text: str = ""
    ) -> None:
        """Convenience method for user feedback."""
        self.add_feedback(
            FeedbackEntry(
                query=query,
                response=response,
                rating=rating,
                feedback_text=text,
                feedback_type="user",
            )
        )

    def add_evaluator_feedback(
        self, query: str, response: str, scores: dict[str, float]
    ) -> None:
        """Add automated evaluator feedback."""
        composite = sum(scores.values()) / len(scores) if scores else 0
        self.add_feedback(
            FeedbackEntry(
                query=query,
                response=response,
                rating=composite,
                feedback_type="evaluator",
                metadata={"scores": scores},
            )
        )

    def get_high_quality_examples(
        self, threshold: float = 0.7
    ) -> list[dict[str, str]]:
        """Extract high-quality entries as training examples.

        Returns entries with rating >= threshold as QA pairs.
        """
        examples = []
        for entry in self._entries:
            if entry.rating >= threshold:
                examples.append({
                    "question": entry.query,
                    "answer": entry.response,
                })
        logger.info(
            f"Extracted {len(examples)} high-quality examples "
            f"from {len(self._entries)} total entries"
        )
        return examples

    def get_low_quality_examples(
        self, threshold: float = 0.4
    ) -> list[dict[str, str]]:
        """Extract low-quality entries for analysis."""
        return [
            {"question": e.query, "answer": e.response, "rating": e.rating}
            for e in self._entries
            if e.rating < threshold
        ]

    def set_optimization_callback(self, callback) -> None:
        """Register a callback to trigger when optimization is needed."""
        self._optimization_callback = callback

    def _maybe_trigger_optimization(self) -> None:
        """Trigger re-optimization if threshold is reached."""
        if self._optimization_callback is not None:
            logger.info(
                f"Triggering re-optimization after {len(self._entries)} feedback entries"
            )
            high_quality = self.get_high_quality_examples()
            self._optimization_callback(high_quality)
            self._entries.clear()  # Reset after optimization

    def _persist_entry(self, entry: FeedbackEntry) -> None:
        """Persist a feedback entry to disk."""
        filepath = self._storage_path / "feedback_log.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def load_from_disk(self) -> int:
        """Load persisted feedback entries."""
        filepath = self._storage_path / "feedback_log.jsonl"
        if not filepath.exists():
            return 0

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self._entries.append(
                        FeedbackEntry(
                            query=data["query"],
                            response=data["response"],
                            rating=data["rating"],
                            feedback_text=data.get("feedback_text", ""),
                            feedback_type=data.get("feedback_type", "user"),
                            timestamp=data.get("timestamp", 0),
                            metadata=data.get("metadata", {}),
                        )
                    )
                    count += 1

        logger.info(f"Loaded {count} feedback entries from disk")
        return count

    @property
    def stats(self) -> dict[str, Any]:
        """Get feedback statistics."""
        if not self._entries:
            return {"total": 0}

        ratings = [e.rating for e in self._entries]
        return {
            "total": len(self._entries),
            "mean_rating": sum(ratings) / len(ratings),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
            "by_type": {
                t: sum(1 for e in self._entries if e.feedback_type == t)
                for t in {"user", "evaluator", "automatic"}
            },
        }
