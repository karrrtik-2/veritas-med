"""
Training / evaluation dataset management for DSPy optimization.

Provides structured dataset loading, splitting, and example
generation compatible with DSPy optimizers.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import dspy

from config.logging_config import get_logger

logger = get_logger("datasets")


@dataclass
class MedicalExample:
    """A single medical QA example for training / evaluation."""
    question: str
    answer: str
    context: str = ""
    metadata: dict[str, Any] | None = None

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example for optimizer consumption."""
        return dspy.Example(
            question=self.question,
            answer=self.answer,
            context=self.context,
        ).with_inputs("question")


class MedicalDatasetManager:
    """
    Manages medical QA datasets for DSPy optimization.

    Supports:
    - Built-in seed examples for bootstrapping
    - JSON file loading
    - Train/dev/test splitting
    - Stratified sampling by query intent
    """

    # ── Built-in seed examples for bootstrapping ─────────────────────────
    SEED_EXAMPLES: list[MedicalExample] = [
        MedicalExample(
            question="What is acromegaly?",
            answer="Acromegaly is a hormonal disorder caused by excess growth hormone (GH) production, "
                   "usually from a pituitary adenoma. It leads to enlargement of hands, feet, and facial "
                   "features. Treatment includes surgery, medication, and radiation therapy.",
        ),
        MedicalExample(
            question="What are the common symptoms of diabetes mellitus type 2?",
            answer="Common symptoms include increased thirst (polydipsia), frequent urination (polyuria), "
                   "unexplained weight loss, fatigue, blurred vision, slow wound healing, and recurrent infections. "
                   "Many patients are asymptomatic in early stages.",
        ),
        MedicalExample(
            question="What is the recommended treatment for mild acne?",
            answer="Mild acne is typically treated with topical retinoids (tretinoin, adapalene), benzoyl peroxide, "
                   "or topical antibiotics (clindamycin). A consistent skincare routine with gentle cleansing is important. "
                   "Treatment response usually takes 6-8 weeks.",
        ),
        MedicalExample(
            question="What are the risk factors for cardiovascular disease?",
            answer="Major risk factors include hypertension, hyperlipidemia, diabetes, smoking, obesity, "
                   "physical inactivity, family history of premature CVD, and age. Modifiable risk factors "
                   "can be addressed through lifestyle changes and medication.",
        ),
        MedicalExample(
            question="How is hypertension diagnosed?",
            answer="Hypertension is diagnosed when blood pressure readings consistently exceed 130/80 mmHg "
                   "(ACC/AHA guidelines) or 140/90 mmHg (ESC/ESH guidelines). Diagnosis requires multiple "
                   "readings on separate occasions, ideally confirmed with ambulatory monitoring.",
        ),
        MedicalExample(
            question="What is the difference between Type 1 and Type 2 diabetes?",
            answer="Type 1 diabetes is an autoimmune condition where the immune system destroys insulin-producing "
                   "beta cells, requiring lifelong insulin therapy. Type 2 diabetes involves insulin resistance "
                   "and relative insulin deficiency, often managed with lifestyle changes and oral medications initially.",
        ),
        MedicalExample(
            question="What causes migraine headaches?",
            answer="Migraines are thought to involve neurovascular mechanisms including cortical spreading "
                   "depression, trigeminal nerve activation, and release of neuropeptides like CGRP. "
                   "Triggers include stress, hormonal changes, certain foods, sleep disruption, and sensory stimuli.",
        ),
        MedicalExample(
            question="What is the treatment for pneumonia?",
            answer="Treatment depends on the type and severity. Community-acquired pneumonia is typically treated "
                   "with antibiotics (amoxicillin, macrolides, or fluoroquinolones). Severe cases require "
                   "hospitalization with IV antibiotics. Viral pneumonia may require antivirals. Supportive care "
                   "includes rest, fluids, and antipyretics.",
        ),
        MedicalExample(
            question="What are the warning signs of a stroke?",
            answer="Warning signs follow the FAST mnemonic: Face drooping, Arm weakness, Speech difficulty, "
                   "Time to call emergency services. Additional signs include sudden severe headache, vision "
                   "changes, confusion, and difficulty walking. Immediate medical attention is critical.",
        ),
        MedicalExample(
            question="What is cancer staging?",
            answer="Cancer staging describes the extent of cancer spread. The TNM system evaluates Tumor size (T), "
                   "lymph Node involvement (N), and distant Metastasis (M). Stages range from I (localized) to "
                   "IV (metastatic). Staging guides treatment decisions and prognosis estimation.",
        ),
    ]

    def __init__(self, data_path: Optional[str] = None) -> None:
        self._data_path = Path(data_path) if data_path else None
        self._examples: list[MedicalExample] = []

    def load_seed_examples(self) -> list[MedicalExample]:
        """Load built-in seed examples."""
        self._examples = list(self.SEED_EXAMPLES)
        logger.info(f"Loaded {len(self._examples)} seed examples")
        return self._examples

    def load_from_json(self, filepath: str) -> list[MedicalExample]:
        """Load examples from a JSON file.

        Expected format:
        [{"question": "...", "answer": "...", "context": "..."}, ...]
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Dataset file not found: {filepath}. Using seed examples.")
            return self.load_seed_examples()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._examples = [
            MedicalExample(
                question=item["question"],
                answer=item.get("answer", ""),
                context=item.get("context", ""),
                metadata=item.get("metadata"),
            )
            for item in data
        ]
        logger.info(f"Loaded {len(self._examples)} examples from {filepath}")
        return self._examples

    def get_dspy_examples(self) -> list[dspy.Example]:
        """Convert all examples to DSPy format."""
        if not self._examples:
            self.load_seed_examples()
        return [ex.to_dspy_example() for ex in self._examples]

    def split(
        self,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
        """Split examples into train/dev/test sets."""
        examples = self.get_dspy_examples()
        rng = random.Random(seed)
        rng.shuffle(examples)

        n = len(examples)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)

        train = examples[:n_train]
        dev = examples[n_train : n_train + n_dev]
        test = examples[n_train + n_dev :]

        logger.info(
            f"Dataset split: train={len(train)}, dev={len(dev)}, test={len(test)}"
        )
        return train, dev, test

    def save_to_json(self, filepath: str) -> None:
        """Save current examples to JSON."""
        data = [
            {
                "question": ex.question,
                "answer": ex.answer,
                "context": ex.context,
                "metadata": ex.metadata,
            }
            for ex in self._examples
        ]
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} examples to {filepath}")
