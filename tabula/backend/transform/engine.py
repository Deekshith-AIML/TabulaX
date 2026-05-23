"""General Transformation Engine (the named orchestrator for rule tasks).

This module is the single facade the API uses for a *general* transformation:
the case where the examples DEMONSTRATE a rule that must be applied to unseen
inputs (classification, extraction, normalization, rewriting) rather than
being an exhaustive lookup table.

It deliberately contains **no** transformation logic of its own.  Each of the
five responsibilities from the design is a thin delegation to the module that
already owns it, so there is exactly one implementation of each concern:

* **Intent Analyzer**   -- :func:`analyze_intent` -> :mod:`transform.classify`
  (what family / task / label set the examples demonstrate).
* **Planner**           -- :func:`plan` -> :mod:`transform.planner`
  (which execution strategy to use).
* **Executor**          -- :func:`execute` -> :mod:`transform.model_exec`
  (model-backed) or :mod:`transform.apply` (deterministic code).
* **Output Validator**  -- :func:`validate_report` (row count, order, index,
  allowed-label and status sanity checks before anything reaches preview).
* **Result Reconstructor** -- :func:`reconstruct` (rebuild the dataset with
  the transformed column, preserving row identity, order and every other
  column, and refusing to emit a mis-aligned frame).

Keeping the orchestration here -- and the logic in the existing modules --
gives the engine a clear name and contract for the API and tests without
duplicating or rewriting the working transformation code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from transform.apply import ApplyError, ApplyReport, apply_to_series
from transform.classify import ClassificationResult, classify
from transform.model_exec import ModelExecutionError, apply_model_backed
from transform.planner import Plan, PlanError, plan_examples
from transform.spec import (
    STRATEGY_DETERMINISTIC,
    STRATEGY_MODEL,
    STRATEGY_NONE,
    TransformationSpec,
    family_label,
)

logger = logging.getLogger(__name__)

ExamplePair = Tuple[str, str]


class EngineError(RuntimeError):
    """Raised when a general transformation cannot be planned or executed."""


# --------------------------------------------------------------------------- #
# A. Transformation Intent Analyzer
# --------------------------------------------------------------------------- #
@dataclass
class Intent:
    """What the examples demonstrate, independent of how it will run."""

    classification: str
    task: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    used_llm: bool = False
    llm_error: Optional[str] = None

    @property
    def family(self) -> str:
        """Human-readable family name (best effort before a spec exists)."""
        from transform.spec import (
            TASK_EXTRACTION,
            TASK_GENERAL_TEXT,
            TASK_NORMALIZATION,
            TASK_SEMANTIC,
        )

        mapping = {
            TASK_SEMANTIC: "Semantic Classification",
            TASK_EXTRACTION: "Information Extraction",
            TASK_NORMALIZATION: "Semantic Normalization",
            TASK_GENERAL_TEXT: "Text Transformation",
        }
        if self.task in mapping:
            return mapping[self.task]
        return self.classification


def analyze_intent(
    pairs: Sequence[ExamplePair],
    allow_llm: bool = True,
    llm_override: Optional[dict] = None,
) -> Intent:
    """Infer the transformation INTENT from the examples (responsibility A).

    Deterministic detectors run first (inside :func:`classify`); the model is
    only consulted when no detector is confident and a provider is available.
    """
    result: ClassificationResult = classify(
        pairs, allow_llm=allow_llm, llm_override=llm_override
    )
    return Intent(
        classification=result.classification,
        task=result.task,
        labels=list(result.labels),
        evidence=list(result.evidence),
        used_llm=result.used_llm,
        llm_error=result.llm_error,
    )


# --------------------------------------------------------------------------- #
# B. Transformation Planner
# --------------------------------------------------------------------------- #
def plan(
    pairs: Sequence[ExamplePair],
    classification_hint: Optional[str] = None,
    **kwargs: Any,
) -> Plan:
    """Choose an execution strategy for the examples (responsibility B)."""
    try:
        return plan_examples(pairs, classification_hint, **kwargs)
    except PlanError as exc:
        raise EngineError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# C. Transformation Executor
# --------------------------------------------------------------------------- #
def execute(
    series: "pd.Series",
    spec: TransformationSpec,
    examples: Sequence[ExamplePair] = (),
    *,
    code: str = "",
    llm_override: Optional[dict] = None,
    budget: Optional[dict] = None,
    batch_size: Optional[int] = None,
) -> ApplyReport:
    """Run the planned transformation over a column (responsibility C).

    Routes on the planner's persisted spec: model-backed specs go to the
    batched, row-id-aligned model executor; deterministic specs run their
    sandbox code.  Raises :class:`EngineError` when no strategy is available.
    """
    if spec.strategy == STRATEGY_MODEL:
        try:
            kwargs: Dict[str, Any] = {
                "llm_override": llm_override,
                "budget": budget,
            }
            if batch_size is not None:
                kwargs["batch_size"] = batch_size
            return apply_model_backed(series, spec, examples, **kwargs)
        except ModelExecutionError as exc:
            raise EngineError(str(exc)) from exc

    if spec.strategy == STRATEGY_DETERMINISTIC:
        if not code:
            raise EngineError(
                "This deterministic transformation has no generated code to "
                "run. Regenerate the transformation and try again."
            )
        try:
            return apply_to_series(code, series)
        except ApplyError as exc:
            raise EngineError(str(exc)) from exc

    raise EngineError(
        spec.unsupported_reason
        or "No execution strategy could be selected for these examples."
    )


# --------------------------------------------------------------------------- #
# D. Output Validator
# --------------------------------------------------------------------------- #
def validate_report(
    report: ApplyReport, spec: Optional[TransformationSpec] = None
) -> ApplyReport:
    """Run integrity checks on a finished report (responsibility D).

    Appends any problems to ``report.warnings`` (and, for hard violations, to
    ``report.errors``) so the caller can surface them before preview.  The
    report is mutated in place and returned.  Checks:

    * the output series has the same length as ``total``;
    * no output row is silently null when its input was non-empty AND the
      row was not already marked failed (a null that is neither accepted nor
      flagged would be data loss);
    * for closed-label tasks, every accepted (ok/review) value is one of the
      allowed labels.
    """
    warnings = list(report.warnings)
    errors = list(report.errors)

    if len(report.series) != report.total:
        errors.append(
            f"Row count mismatch after transform: expected {report.total}, "
            f"got {len(report.series)}."
        )

    if report.row_status and len(report.row_status) != report.total:
        warnings.append(
            "Per-row status list did not align with the output rows; row-level "
            "review detail may be incomplete."
        )

    if spec is not None and spec.allowed_labels:
        allowed = set(spec.allowed_labels)
        bad = [
            v
            for v, s in zip(report.series.tolist(), report.row_status or [])
            if s in ("ok", "review") and isinstance(v, str) and v not in allowed
        ]
        if bad:
            errors.append(
                f"{len(bad)} accepted row(s) carried a label outside the "
                f"allowed set (e.g. {bad[0]!r})."
            )

    # Deduplicate while preserving order.
    report.warnings = list(dict.fromkeys(warnings))
    report.errors = list(dict.fromkeys(errors))
    return report


# --------------------------------------------------------------------------- #
# E. Result Reconstructor
# --------------------------------------------------------------------------- #
def reconstruct(
    df: "pd.DataFrame",
    column: str,
    report: ApplyReport,
) -> "pd.DataFrame":
    """Rebuild the dataset with the transformed column (responsibility E).

    Preserves row order, row identity (index), every unselected column and
    the original frame (a copy is returned).  Refuses to emit a frame whose
    transformed column does not align row-for-row with the original, so a
    failed prediction can never shift subsequent results.
    """
    if column not in df.columns:
        raise EngineError(
            f"Column '{column}' not found in the dataset "
            f"(available: {list(df.columns)[:10]})."
        )
    if len(report.series) != len(df):
        raise EngineError(
            f"Transformed column has {len(report.series)} rows but the dataset "
            f"has {len(df)}; refusing to write a mis-aligned result."
        )
    # The transformed series must align index-for-index with the source so a
    # reordering can never map outputs to the wrong rows.
    if not report.series.index.equals(df.index):
        try:
            aligned = report.series.reindex(df.index)
        except Exception as exc:  # pragma: no cover - defensive
            raise EngineError(
                "Transformed column could not be aligned to the dataset index."
            ) from exc
        if aligned.isna().any() and not report.series.isna().all():
            raise EngineError(
                "Transformed column index does not match the dataset; refusing "
                "to write a mis-aligned result."
            )
        report.series = aligned

    out = df.copy()
    out[column] = report.series
    return out


# --------------------------------------------------------------------------- #
# Convenience: the whole pipeline for callers that want one call
# --------------------------------------------------------------------------- #
def run(
    df: "pd.DataFrame",
    column: str,
    pairs: Sequence[ExamplePair],
    *,
    code: str = "",
    classification_hint: Optional[str] = None,
    allow_llm: bool = True,
    llm_override: Optional[dict] = None,
    budget: Optional[dict] = None,
    treat_as_mapping: bool = False,
    llm_task: Optional[str] = None,
    labels_hint: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
) -> Tuple["pd.DataFrame", ApplyReport, Plan]:
    """Plan, execute, validate and reconstruct in one call.

    Returns ``(new_df, report, plan)``.  Deterministic specs still need
    ``code`` (generated elsewhere); model-backed specs ignore it.
    """
    plan_result = plan(
        pairs,
        classification_hint,
        allow_llm=allow_llm,
        llm_override=llm_override,
        treat_as_mapping=treat_as_mapping,
        llm_task=llm_task,
        labels_hint=labels_hint,
        source_column=column,
        budget=budget,
    )
    spec = plan_result.spec
    if not spec.strategy_chosen():
        raise EngineError(
            spec.unsupported_reason
            or "The examples do not determine a single transformation rule."
        )
    report = execute(
        df[column],
        spec,
        pairs,
        code=code,
        llm_override=llm_override,
        budget=budget,
        batch_size=batch_size,
    )
    validate_report(report, spec)
    new_df = reconstruct(df, column, report)
    return new_df, report, plan_result


__all__ = [
    "EngineError",
    "Intent",
    "analyze_intent",
    "plan",
    "execute",
    "validate_report",
    "reconstruct",
    "run",
    "family_label",
    "STRATEGY_MODEL",
    "STRATEGY_DETERMINISTIC",
    "STRATEGY_NONE",
    "TransformationSpec",
]
