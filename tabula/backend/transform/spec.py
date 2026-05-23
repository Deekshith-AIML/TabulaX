"""Transformation specification (the inference-to-execution contract).

Every transformation considered by the engine is first reduced to an explicit,
serialisable :class:`TransformationSpec`.  The spec records:

* what task is being performed (``kind``),
* the inferred input / output types and constraints,
* the assumptions made from the examples,
* the execution strategy chosen to implement it, and
* any ambiguity detected in the examples.

The spec is what the planner decides on, what the generators implement, what is
persisted with the transformation and what the API returns so the UI can show
the user *why* a strategy was chosen and flag uncertainty.

``kind`` values
---------------
``text_case`` / ``text_trim`` / ``text_replace`` / ``text_extract`` /
``text_delimiter`` / ``text_fallback``
    Deterministic string operations.  Implemented as reusable Python code.
``numeric``
    A numeric function (from the numerical fitter).  Reusable code.
``algorithmic``
    A known conversion (Roman numerals, dates, base64, ...).  Reusable code.
``exact_mapping``
    The examples ARE the mapping.  Implemented as a lookup table.
``semantic``
    The task requires understanding meaning (classification, sentiment,
    intent, entity extraction).  Implemented by model-backed execution.
``general_text``
    General rewriting / structuring that needs a model but is not semantic
    classification (e.g. extracting a domain from a URL in free-form text).
``unsupported``
    The engine could not determine a reliable rule; report, do not fabricate.

Strategies
----------
``deterministic``
    Pure Python code executed in the sandbox.
``model``
    Model-backed execution (batched, structured output, row-aligned).
``none``
    No strategy could be selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Spec kinds --------------------------------------------------------------- #
KIND_TEXT_CASE = "text_case"
KIND_TEXT_TRIM = "text_trim"
KIND_TEXT_REPLACE = "text_replace"
KIND_TEXT_EXTRACT = "text_extract"
KIND_TEXT_DELIMITER = "text_delimiter"
KIND_TEXT_FALLBACK = "text_fallback"
KIND_NUMERIC = "numeric"
KIND_ALGORITHMIC = "algorithmic"
KIND_EXACT_MAPPING = "exact_mapping"
KIND_SEMANTIC = "semantic"
KIND_NORMALIZATION = "semantic_normalization"
KIND_GENERAL_TEXT = "general_text"
KIND_UNSUPPORTED = "unsupported"

# Strategies --------------------------------------------------------------- #
STRATEGY_DETERMINISTIC = "deterministic"
STRATEGY_MODEL = "model"
STRATEGY_NONE = "none"

# LLM task vocabulary (what the examples demonstrate) ----------------------- #
TASK_EXACT_MAPPING = "exact_mapping"
TASK_SEMANTIC = "semantic"
TASK_EXTRACTION = "extraction"
TASK_NORMALIZATION = "normalization"
TASK_GENERAL_TEXT = "general_text"
TASK_AMBIGUOUS = "ambiguous"

VALID_TASKS = (
    TASK_EXACT_MAPPING,
    TASK_SEMANTIC,
    TASK_EXTRACTION,
    TASK_NORMALIZATION,
    TASK_GENERAL_TEXT,
    TASK_AMBIGUOUS,
)

# Human-readable family names shown to the user (spec §15: the UI must state
# which transformation family was inferred).
FAMILY_LABELS = {
    KIND_SEMANTIC: "Semantic Classification",
    KIND_NORMALIZATION: "Semantic Normalization",
    KIND_GENERAL_TEXT: "Text Transformation",
    KIND_EXACT_MAPPING: "Exact Mapping",
    KIND_NUMERIC: "Numerical",
    KIND_ALGORITHMIC: "Algorithmic",
    KIND_UNSUPPORTED: "Unclassified",
}


def family_label(kind: str) -> str:
    """Display name for a spec kind (falls back to the raw kind)."""
    return FAMILY_LABELS.get(kind, kind.replace("_", " ").title())


@dataclass
class TransformationSpec:
    """Explicit description of an inferred transformation."""

    kind: str
    source_column: str = ""
    output_type: str = "string"
    # Whether examples should be treated as the exhaustive mapping.
    exhaustive_mapping: bool = False
    # Model-backed tasks: fixed label set inferred from the examples (if any).
    allowed_labels: List[str] = field(default_factory=list)
    # Human-readable description of the inferred operation.
    summary: str = ""
    # Assumptions / interpretations the engine made (shown to the user).
    assumptions: List[str] = field(default_factory=list)
    # Ambiguities detected in the examples that the user should resolve.
    ambiguities: List[str] = field(default_factory=list)
    # Strategy chosen by the planner.
    strategy: str = STRATEGY_NONE
    # Whether a model is required (and configured) to execute.
    requires_model: bool = False
    # True when examples alone are not enough to execute without a model.
    model_required_flag: bool = False
    # Set when the planner could not choose a strategy.
    unsupported_reason: str = ""
    # The task the examples demonstrate (one of VALID_TASKS).  Distinguishes
    # classification / extraction / normalization / rewriting so the executor
    # prompts and the UI report the right transformation FAMILY.
    task: str = ""

    # ------------------------------------------------------------------ #
    def strategy_chosen(self) -> bool:
        return self.strategy in (STRATEGY_DETERMINISTIC, STRATEGY_MODEL)

    @property
    def family(self) -> str:
        """Human-readable transformation family for reports and the UI."""
        return family_label(self.kind)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable representation (stored with the transformation)."""
        return {
            "kind": self.kind,
            "source_column": self.source_column,
            "output_type": self.output_type,
            "exhaustive_mapping": self.exhaustive_mapping,
            "allowed_labels": list(self.allowed_labels),
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "ambiguities": list(self.ambiguities),
            "strategy": self.strategy,
            "requires_model": self.requires_model,
            "model_required": self.model_required_flag,
            "unsupported_reason": self.unsupported_reason,
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TransformationSpec":
        data = data or {}
        return cls(
            kind=str(data.get("kind", KIND_UNSUPPORTED)),
            source_column=str(data.get("source_column", "")),
            output_type=str(data.get("output_type", "string")),
            exhaustive_mapping=bool(data.get("exhaustive_mapping", False)),
            allowed_labels=[str(x) for x in (data.get("allowed_labels") or [])],
            summary=str(data.get("summary", "")),
            assumptions=[str(x) for x in (data.get("assumptions") or [])],
            ambiguities=[str(x) for x in (data.get("ambiguities") or [])],
            strategy=str(data.get("strategy", STRATEGY_NONE)),
            requires_model=bool(data.get("requires_model", False)),
            model_required_flag=bool(data.get("model_required", False)),
            unsupported_reason=str(data.get("unsupported_reason", "")),
            task=str(data.get("task", "")),
        )
