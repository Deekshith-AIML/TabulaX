"""Transformation planner: decides HOW a transformation is implemented.

The planner is the single place where TabulaX chooses an execution strategy
for a set of example pairs.  It separates two fundamentally different tasks
that the previous implementation conflated:

* **exact mapping** -- the examples ARE the rule; a lookup table is correct.
* **rule inference** -- the examples DEMONSTRATE a rule that must be applied to
  previously unseen inputs (classification, sentiment, intent, extraction,
  rewriting, algorithms).  A lookup table is *not* a valid implementation.

Order of decision:

1. Deterministic detectors run first (algorithmic templates, numeric fits,
   date parsing, string operations).  They produce reusable Python code and
   never need a model.
2. If no detector matches, the open-ended task is resolved:

   * an explicit ``treat_as_mapping`` request -> exact mapping (lookup);
   * a client-supplied ``"String-based"`` classification -> the existing
     LLM-authored-code path for open string operations;
   * an LLM task decision (``exact_mapping`` / ``semantic`` / ``extraction`` /
     ``general_text`` / ``ambiguous``) when a model is available;
   * otherwise a conservative deterministic heuristic: short, closed,
     unrelated input/output pairs are treated as a mapping (legacy behaviour,
     and the only deterministic option left); sentence-like or free-form
     pairs are reported as needing a model instead of silently producing a
     lookup table that returns ``UNKNOWN`` for unseen data.

Nothing in this module knows about any particular dataset, column name, label
set or input value; every decision is made from the shape of the pairs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from extractors import is_date
from transform import numerical
from transform.classify import (
    ExamplePair,
    _affix_swap,
    _all_numeric,
    _looks_numeric_fit,
    _match_template,
    _string_pattern_evidence,
)
from transform.spec import (
    KIND_ALGORITHMIC,
    KIND_EXACT_MAPPING,
    KIND_GENERAL_TEXT,
    KIND_NORMALIZATION,
    KIND_NUMERIC,
    KIND_SEMANTIC,
    KIND_TEXT_CASE,
    KIND_TEXT_EXTRACT,
    KIND_TEXT_FALLBACK,
    KIND_TEXT_REPLACE,
    KIND_TEXT_TRIM,
    KIND_UNSUPPORTED,
    STRATEGY_DETERMINISTIC,
    STRATEGY_MODEL,
    STRATEGY_NONE,
    TASK_AMBIGUOUS,
    TASK_EXACT_MAPPING,
    TASK_EXTRACTION,
    TASK_GENERAL_TEXT,
    TASK_NORMALIZATION,
    TASK_SEMANTIC,
    TransformationSpec,
    VALID_TASKS,
)

logger = logging.getLogger(__name__)

# Heuristic bounds for "these pairs look like a closed mapping".  They describe
# SHAPE only (short, closed, unrelated values) -- never any particular data.
_MAX_LOOKUP_INPUT_LEN = 64
_MAX_LOOKUP_OUTPUT_LEN = 80
_MAX_AVG_LOOKUP_INPUT_LEN = 24.0
_SENTENCE_PUNCT = ".!?"
_MAX_LABELS = 12
_MAX_LABEL_LEN = 48
_MAX_LABEL_WORDS = 5


class PlanError(RuntimeError):
    """Raised when planning fails irrecoverably (contradictory examples)."""


# --------------------------------------------------------------------------- #
# Example-shape analysis (deterministic, dataset-agnostic)
# --------------------------------------------------------------------------- #
def find_contradictions(pairs: Sequence[ExamplePair]) -> List[str]:
    """Return messages for inputs that map to more than one output."""
    seen: Dict[str, str] = {}
    out: List[str] = []
    for s, t in pairs:
        if s in seen and seen[s] != t:
            out.append(
                f"input {s!r} maps to both {seen[s]!r} and {t!r}; the examples "
                "contradict each other"
            )
        seen.setdefault(s, t)
    return out


def _common_affix(values: Sequence[str]) -> Tuple[str, str]:
    """Return the (prefix, suffix) shared by every value."""
    if not values:
        return "", ""
    prefix = os.path.commonprefix(list(values))
    suffix = os.path.commonprefix([v[::-1] for v in values])[::-1]
    return prefix, suffix


def _affix_rule(pairs: Sequence[ExamplePair]) -> bool:
    """True when outputs share a fixed prefix/suffix absent from the inputs.

    e.g. ``foo -> Qq1foo``: every output starts with ``Qq1``.  That is a string
    operation (wrap in affixes), NOT a closed mapping.
    """
    if len(pairs) < 2:
        return False
    ins = [s.lower() for s, _ in pairs]
    outs = [t for _, t in pairs]
    prefix, suffix = _common_affix(outs)
    for affix in (prefix, suffix):
        if len(affix) >= 2 and all(affix.lower() not in i for i in ins):
            return True
    return False


def looks_like_closed_mapping(pairs: Sequence[ExamplePair]) -> bool:
    """Heuristic: do these pairs look like an exhaustive input->output table?

    A mapping-shaped pair set has short, closed, unrelated values (codes,
    names, labels).  Sentence-like or free-form inputs indicate a rule that
    must generalise, and outputs that contain their input (or share a fixed
    affix) indicate a string operation -- neither is a lookup table.
    """
    if len(pairs) < 2:
        return False
    ins = [s.strip() for s, _ in pairs]
    outs = [t.strip() for _, t in pairs]
    if any(not i or not o for i, o in zip(ins, outs)):
        return False
    if any(len(i) > _MAX_LOOKUP_INPUT_LEN for i in ins):
        return False
    if any(len(o) > _MAX_LOOKUP_OUTPUT_LEN for o in outs):
        return False
    # Containment in either direction means a substring/extract rule.
    if any(o.lower() in i.lower() or i.lower() in o.lower() for i, o in zip(ins, outs)):
        return False
    if _affix_rule(pairs):
        return False
    avg_in = sum(len(i) for i in ins) / len(ins)
    if avg_in > _MAX_AVG_LOOKUP_INPUT_LEN:
        return False
    if any(any(ch in i for ch in _SENTENCE_PUNCT) for i in ins):
        return False
    return True


def infer_label_set(pairs: Sequence[ExamplePair]) -> List[str]:
    """Closed set of output labels for classification-like example pairs.

    Only returned when the outputs look like category labels: a small closed
    set of short values that do not appear inside their inputs (an output that
    occurs in its input indicates extraction, which has no fixed label set).
    """
    outs = [t for _, t in pairs]
    labels = list(dict.fromkeys(outs))
    if not (2 <= len(labels) <= _MAX_LABELS):
        return []
    if any(len(l) > _MAX_LABEL_LEN or len(l.split()) > _MAX_LABEL_WORDS for l in labels):
        return []
    if any(o.lower() in s.lower() for s, o in pairs):
        return []
    return labels


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    """The planner's decision: a spec plus the details generators need."""

    spec: TransformationSpec
    classification: str
    # Matched / requested algorithmic template key.
    template: Optional[str] = None
    # Deterministic string operation detected (generate emits the code).
    string_op: Optional[str] = None
    # True when code must be authored by an LLM (open-ended string operation).
    llm_code_required: bool = False
    # Raw LLM task decision (exact_mapping/semantic/extraction/general_text).
    llm_task: Optional[str] = None
    # Whether planning itself consulted a model.
    used_llm: bool = False
    # Shared per-request LLM budget (created when planning used a model).
    budget: Optional[dict] = field(default=None, repr=False)
    # Human-readable notes about why the strategy was chosen.
    evidence: List[str] = field(default_factory=list)


def _string_kind(pairs: Sequence[ExamplePair]) -> Tuple[str, Optional[str]]:
    """Map detected string evidence to a spec kind + generator op name."""
    ins = [s for s, _ in pairs]
    outs = [t for _, t in pairs]
    if all(o == i.upper() for i, o in zip(ins, outs)) or all(
        o == i.lower() for i, o in zip(ins, outs)
    ) or all(o == i.title() for i, o in zip(ins, outs)):
        return KIND_TEXT_CASE, "case"
    if all(o == i.strip() for i, o in zip(ins, outs)):
        return KIND_TEXT_TRIM, "strip"
    if all(o == i.replace(" ", "") for i, o in zip(ins, outs)) or all(
        o == i[::-1] for i, o in zip(ins, outs)
    ):
        return KIND_TEXT_REPLACE, "replace"
    swap = _affix_swap(pairs)
    if swap:
        # Fixed prefix/suffix replacement: deterministic reusable code.
        return KIND_TEXT_REPLACE, swap[0]
    # Evidence exists (substring containment, etc.) but no deterministic code
    # generator: the LLM authors code for it (existing behaviour).
    return KIND_TEXT_FALLBACK, None


def _validate_labels(labels: Any) -> List[str]:
    """Validate a model-provided label set; ignore anything malformed."""
    if not isinstance(labels, (list, tuple)):
        return []
    clean: List[str] = []
    for item in labels:
        if not isinstance(item, str):
            return []
        text = item.strip()
        if not text or len(text) > 64:
            return []
        clean.append(text)
    if not (1 <= len(clean) <= 30):
        return []
    return list(dict.fromkeys(clean))


def _spec_for_task(
    task: str,
    pairs: Sequence[ExamplePair],
    labels_hint: Optional[Sequence[str]] = None,
    source_column: str = "",
) -> TransformationSpec:
    """Build the spec for an open-ended (non-deterministic) task."""
    labels = _validate_labels(labels_hint) or (
        infer_label_set(pairs) if task == TASK_SEMANTIC else []
    )
    if task == TASK_EXACT_MAPPING:
        return TransformationSpec(
            kind=KIND_EXACT_MAPPING,
            source_column=source_column,
            output_type="string",
            exhaustive_mapping=True,
            task=task,
            summary="Exact mapping: the examples are the complete rule.",
            assumptions=[
                "Examples are treated as an exhaustive mapping; inputs not in "
                "the examples are flagged as unmapped rather than guessed."
            ],
            strategy=STRATEGY_DETERMINISTIC,
        )
    if task == TASK_SEMANTIC:
        return TransformationSpec(
            kind=KIND_SEMANTIC,
            source_column=source_column,
            output_type="string",
            allowed_labels=labels,
            task=task,
            summary=(
                "Semantic classification: examples demonstrate a labelling "
                "rule that is applied to unseen inputs by the configured model."
            ),
            assumptions=[
                "Examples demonstrate a rule; unseen inputs are classified by "
                "the model, not matched exactly."
            ],
            strategy=STRATEGY_MODEL,
            requires_model=True,
            model_required_flag=True,
        )
    if task in (TASK_EXTRACTION, TASK_GENERAL_TEXT):
        return TransformationSpec(
            kind=KIND_GENERAL_TEXT,
            source_column=source_column,
            output_type="string",
            task=task,
            summary=(
                "General text transformation: the examples demonstrate a "
                "rule (extraction / reformatting) applied to unseen inputs "
                "by the configured model."
            ),
            assumptions=[
                "Examples demonstrate a rule; unseen inputs are transformed by "
                "the model, not matched exactly."
            ],
            strategy=STRATEGY_MODEL,
            requires_model=True,
            model_required_flag=True,
        )
    if task == TASK_NORMALIZATION:
        return TransformationSpec(
            kind=KIND_NORMALIZATION,
            source_column=source_column,
            output_type="string",
            task=task,
            summary=(
                "Semantic normalization: differently worded inputs are "
                "rewritten to one consistent canonical form by the "
                "configured model."
            ),
            assumptions=[
                "Examples demonstrate a canonical form; unseen inputs are "
                "normalized by meaning, not matched exactly."
            ],
            strategy=STRATEGY_MODEL,
            requires_model=True,
            model_required_flag=True,
        )
    # ambiguous / unknown
    return TransformationSpec(
        kind=KIND_UNSUPPORTED,
        source_column=source_column,
        summary="The examples do not determine a single transformation rule.",
        ambiguities=[
            "The examples are contradictory or too few to infer one rule. Add "
            "more examples that cover the intended behaviour, including at "
            "least one example per expected output category."
        ],
        strategy=STRATEGY_NONE,
        unsupported_reason=(
            "Could not infer a single transformation rule from the examples. "
            "Add more representative examples (including edge cases and one "
            "example per expected output category) and try again."
        ),
    )


def plan_examples(
    pairs: Sequence[ExamplePair],
    classification_hint: Optional[str] = None,
    *,
    template_hint: Optional[str] = None,
    allow_llm: bool = True,
    llm_override: Optional[dict] = None,
    treat_as_mapping: bool = False,
    llm_task: Optional[str] = None,
    labels_hint: Optional[Sequence[str]] = None,
    source_column: str = "",
    budget: Optional[dict] = None,
) -> Plan:
    """Choose an execution strategy for ``pairs``.

    ``classification_hint`` is the client's earlier ``/classify/`` answer; it
    is only trusted for routing the open-ended *String-based* family.  All
    deterministic decisions are re-derived from the examples themselves.
    ``llm_task`` / ``labels_hint`` carry a previous LLM task decision so that
    classify -> generate does not pay for a second model call.
    """
    import llm as _llm

    pairs = [(str(s), str(t)) for s, t in pairs]
    if not pairs:
        raise PlanError("At least one example pair is required.")

    contradictions = find_contradictions(pairs)
    if contradictions:
        raise PlanError("; ".join(contradictions))

    hint = (classification_hint or "").strip().lower()
    personal = bool((llm_override or {}).get("api_key"))
    can_llm = bool(allow_llm or personal)
    own_budget = budget
    if own_budget is None and can_llm:
        own_budget = {"remaining": _llm.settings.llm_max_calls_per_request}

    evidence: List[str] = []

    # ---- 1. Deterministic algorithmic template ---------------------------- #
    matched = _match_template(pairs)
    if matched:
        evidence.append(f"matched the built-in '{matched}' template exactly")
        return Plan(
            spec=TransformationSpec(
                kind=KIND_ALGORITHMIC,
                source_column=source_column,
                summary=f"Built-in '{matched}' conversion.",
                strategy=STRATEGY_DETERMINISTIC,
            ),
            classification="Algorithmic",
            template=matched,
            evidence=evidence,
        )

    # ---- 2. Deterministic numeric fit ------------------------------------- #
    if _all_numeric(pairs) and _looks_numeric_fit(pairs):
        try:
            evidence.append(numerical.describe_fit(pairs))
        except Exception:
            evidence.append("a numeric model fits the pairs within tolerance")
        return Plan(
            spec=TransformationSpec(
                kind=KIND_NUMERIC,
                source_column=source_column,
                output_type="number",
                summary="Numeric function fitted from the examples.",
                strategy=STRATEGY_DETERMINISTIC,
            ),
            classification="Numerical",
            evidence=evidence,
        )

    # ---- 3. Deterministic date handling ----------------------------------- #
    if all(is_date(s) and is_date(t) for s, t in pairs):
        evidence.append("all inputs and outputs parse as dates")
        return Plan(
            spec=TransformationSpec(
                kind=KIND_ALGORITHMIC,
                source_column=source_column,
                summary="Date handling (no built-in template matched).",
                strategy=STRATEGY_DETERMINISTIC,
            ),
            classification="Algorithmic",
            template=template_hint if template_hint else None,
            evidence=evidence,
        )

    # ---- 4. Deterministic string operations -------------------------------- #
    str_ev = _string_pattern_evidence(pairs)
    containment_ev: Optional[str] = None
    if str_ev:
        kind, op = _string_kind(pairs)
        if op is not None:
            # A deterministic generator exists (case/trim/replace/prefix...).
            evidence.append(str_ev)
            return Plan(
                spec=TransformationSpec(
                    kind=kind,
                    source_column=source_column,
                    summary=f"Deterministic string rule ({str_ev}).",
                    strategy=STRATEGY_DETERMINISTIC,
                ),
                classification="String-based",
                string_op=op,
                evidence=evidence,
            )
        # Evidence exists (substring containment etc.) but no deterministic
        # code generator.  A *String-based* hint keeps the LLM-authored-code
        # path below (after the explicit mapping request); a General hint
        # falls through to the task decision (extraction -> model-backed).
        containment_ev = str_ev

    # ---- 5. Numeric but no good fit: keep existing behaviour --------------- #
    if _all_numeric(pairs):
        evidence.append("all pairs are numeric, but no simple numeric model fits well")
        return Plan(
            spec=TransformationSpec(
                kind=KIND_NUMERIC,
                source_column=source_column,
                output_type="number",
                summary="Numeric intent; no simple model fits exactly.",
                strategy=STRATEGY_DETERMINISTIC,
            ),
            classification="Numerical",
            evidence=evidence,
        )

    # ---- 6. Open-ended: mapping vs rule inference -------------------------- #
    if treat_as_mapping:
        evidence.append("the user explicitly requested an exact mapping")
        return Plan(
            spec=_spec_for_task(
                TASK_EXACT_MAPPING, pairs, source_column=source_column
            ),
            classification="General",
            budget=own_budget,
            evidence=evidence,
        )

    # Open-ended *string* operations keep the LLM-authored-code path.
    if "string" in hint:
        evidence.append(
            "open-ended string operation: code is authored by the LLM and "
            "validated against the examples"
        )
        if containment_ev:
            evidence.append(containment_ev)
        return Plan(
            spec=TransformationSpec(
                kind=KIND_TEXT_FALLBACK,
                source_column=source_column,
                summary="Open-ended string transformation (LLM-authored code).",
                strategy=STRATEGY_DETERMINISTIC,
                requires_model=can_llm,
            ),
            classification="String-based",
            llm_code_required=True,
            budget=own_budget,
            evidence=evidence,
        )

    # An LLM task decision supplied by the caller (classify -> generate).
    task = (llm_task or "").strip().lower()
    used_llm = False
    # Why a *configured* model could not give us a task decision (rate limit,
    # quota, network, unusable response).  Empty when no call was needed or
    # the call succeeded.  Used to report the ACTUAL failure instead of the
    # misleading "configure a model" message.
    decision_failure: str = ""
    if task not in VALID_TASKS:
        task = ""
    if not task and can_llm:
        from transform.classify import _llm_task_classify

        decided, _labels, _amb, reason = _llm_task_classify(
            pairs, override=llm_override, budget=own_budget
        )
        used_llm = True
        if decided:
            task = decided
            labels_hint = _labels
            evidence.append(f"the LLM identified the task as '{decided}'")
        else:
            decision_failure = reason or "the model did not identify the task"
            evidence.append(f"the LLM could not classify the task ({reason})")

    if task:
        if task == TASK_AMBIGUOUS:
            evidence.append("the LLM judged the examples ambiguous")
            return Plan(
                spec=_spec_for_task(TASK_AMBIGUOUS, pairs, source_column=source_column),
                classification="General",
                used_llm=used_llm,
                llm_task=task,
                budget=own_budget,
                evidence=evidence,
            )
        if task == TASK_EXACT_MAPPING and not looks_like_closed_mapping(pairs):
            # The model called this an exact mapping, but the deterministic
            # shape heuristic says the inputs are rule-like (sentence-like /
            # free-form), not a closed table.  A lookup table here would flag
            # every unseen input as UNKNOWN, so we honour the spec rule
            # "lookup only for closed mappings" and route to the model-backed
            # path instead.  (An EXPLICIT user request for an exact mapping is
            # handled earlier and always wins; this only constrains a
            # model-suggested mapping.)
            evidence.append(
                "the model suggested an exact mapping, but the inputs are "
                "rule-like (not a closed table), so the examples are treated "
                "as a rule that generalises to unseen inputs"
            )
            task = TASK_SEMANTIC if labels_hint else TASK_GENERAL_TEXT
        spec = _spec_for_task(task, pairs, labels_hint, source_column)
        if spec.strategy == STRATEGY_MODEL and not can_llm:  # pragma: no cover
            spec.strategy = STRATEGY_NONE
            spec.unsupported_reason = (
                "This transformation requires a model but none is configured."
            )
        return Plan(
            spec=spec,
            classification="General",
            llm_task=task,
            used_llm=used_llm,
            budget=own_budget,
            evidence=evidence,
        )

    # No model decision available.  Only a mapping-shaped pair set may fall
    # back to a deterministic lookup; anything rule-like must report the
    # limitation instead of silently returning UNKNOWN for unseen inputs.
    if looks_like_closed_mapping(pairs):
        evidence.append(
            "the pairs look like a closed mapping and no model is available, "
            "so an exact lookup table was built (unseen inputs are flagged)"
        )
        return Plan(
            spec=_spec_for_task(
                TASK_EXACT_MAPPING, pairs, source_column=source_column
            ),
            classification="General",
            used_llm=used_llm,
            budget=own_budget,
            evidence=evidence,
        )

    if not can_llm:
        unsupported_reason = (
            "These examples look like a rule that must generalise to unseen "
            "inputs (classification, extraction, or rewriting), which needs a "
            "model. Configure OPENAI_API_KEY, a personal key in AI settings, "
            "or Ollama (LLM_PROVIDER=ollama) -- or explicitly mark the "
            "examples as an exact mapping."
        )
        evidence.append("no model available and the pairs are not a closed mapping")
    else:
        # A model IS configured -- it just could not answer (rate limit,
        # exhausted quota, network, unusable response).  Report the actual
        # failure; telling the user to configure a key they already have
        # would be wrong and unhelpful.
        unsupported_reason = (
            "These examples look like a rule that must generalise to unseen "
            "inputs (classification, extraction, or rewriting), which needs a "
            f"model, but the model request failed: "
            f"{decision_failure or 'the model did not return a usable task decision'}. "
            "You can also explicitly mark the examples as an exact mapping."
        )
        evidence.append(
            f"a model is configured but the task decision failed: "
            f"{decision_failure or 'unusable response'}"
        )

    spec = TransformationSpec(
        kind=KIND_UNSUPPORTED,
        source_column=source_column,
        summary="Rule-like examples that require a model to execute.",
        ambiguities=[
            "These examples demonstrate a rule that requires understanding "
            "the input (e.g. classification or extraction); an exact lookup "
            "table would return UNKNOWN for every unseen value. "
            + (
                "No model is configured yet."
                if not can_llm
                else "The configured model could not be reached right now "
                f"({decision_failure or 'no usable decision'})."
            )
        ],
        strategy=STRATEGY_NONE,
        requires_model=True,
        model_required_flag=True,
        unsupported_reason=unsupported_reason,
    )
    return Plan(
        spec=spec,
        classification="General",
        used_llm=used_llm,
        budget=own_budget,
        evidence=evidence,
    )
