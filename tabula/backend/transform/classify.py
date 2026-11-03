"""Transformation classification.

Classification decides which of the four families a set of (input, output)
example pairs belongs to:

* ``String-based``   -- text manipulation (case, trim, substrings, ...)
* ``Numerical``      -- a numeric function maps inputs to outputs
* ``Algorithmic``    -- a known conversion (dates, bases, encodings, ...)
* ``General``        -- a lookup / knowledge mapping

Deterministic detectors run first (numeric fit, template matching, pattern
heuristics).  The LLM is consulted **only** when no deterministic detector is
confident, and not at all when no LLM is configured.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from extractors import is_date
from sandbox import SandboxError, validate_transformation_code
from transform import numerical, templates

logger = logging.getLogger(__name__)

ExamplePair = Tuple[str, str]

VALID_CLASSES = ("String-based", "Numerical", "Algorithmic", "General")


@dataclass
class ClassificationResult:
    """Outcome of classifying a set of example pairs."""

    classification: str
    evidence: List[str] = field(default_factory=list)
    # For Algorithmic: the matched template key (deterministic match) if any.
    template: Optional[str] = None
    # True when an LLM was required to reach the decision.
    used_llm: bool = False
    # LLM task decision (exact_mapping/semantic/extraction/general_text/
    # ambiguous) when an open-ended task was classified by the model.
    task: Optional[str] = None
    # Closed label set the model reported for semantic classification.
    labels: List[str] = field(default_factory=list)
    # Why the model could not be consulted when it was supposed to be
    # (rate limit / quota / network / malformed response).  ``None`` when no
    # model was needed or the model answered successfully.
    llm_error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _arrow_split(line: str) -> Optional[Tuple[str, str]]:
    """Split ``line`` at the first recognized arrow (``→`` or ``->``).

    Everything before the arrow is the input and everything after it is the
    output, so values containing commas, spaces, dates or other arrows are
    preserved verbatim (only surrounding whitespace is trimmed later).
    """
    positions = [i for i in (line.find("→"), line.find("->")) if i >= 0]
    if not positions:
        return None
    i = min(positions)
    if line.startswith("->", i):
        return line[:i], line[i + 2:]
    return line[:i], line[i + 1:]


def _validate_labeled(
    labeled: Sequence[Tuple[str, str, str]]
) -> Tuple[List[ExamplePair], List[str]]:
    """Validate ``(label, input, output)`` triples.

    Returns ``(pairs, errors)``.  Errors name their origin (``Line 3`` /
    ``Example 2``) so callers can surface actionable messages; empty inputs or
    outputs and contradictory duplicates are rejected.
    """
    errors: List[str] = []
    seen: Dict[str, Tuple[str, str]] = {}
    pairs: List[ExamplePair] = []
    for label, src, tgt in labeled:
        s, t = src.strip(), tgt.strip()
        if not s:
            errors.append(f"{label}: missing input before the arrow.")
            continue
        if not t:
            errors.append(f"{label}: missing output after the arrow.")
            continue
        if s in seen:
            prev_t, prev_label = seen[s]
            if prev_t != t:
                errors.append(
                    f"{label}: contradictory duplicate - input {s!r} maps to "
                    f"{prev_t!r} ({prev_label}) and {t!r} ({label})."
                )
            continue  # identical duplicate: keep the first occurrence.
        seen[s] = (t, label)
        pairs.append((s, t))
    return pairs, errors


def parse_examples(raw: str) -> List[ExamplePair]:
    """Parse ``"input -> output"`` / ``"input → output"`` lines into pairs.

    One example per line; blank lines are ignored.  The line is split at the
    first recognized arrow so outputs (or inputs) containing commas, spaces,
    dates or Unicode survive untouched.  A legacy ``input,output`` line (no
    arrow anywhere) is still accepted for backwards compatibility.

    Raises ``ValueError`` listing every malformed line with its 1-based line
    number so callers can return a useful 400.
    """
    labeled: List[Tuple[str, str, str]] = []
    errors: List[str] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        split = _arrow_split(stripped)
        if split is None:
            if "," in stripped:
                src, _, tgt = stripped.partition(",")
                split = (src, tgt)
            else:
                errors.append(
                    f"Line {lineno}: expected 'input -> output' or "
                    f"'input \u2192 output' but found {stripped!r}."
                )
                continue
        labeled.append((f"Line {lineno}", split[0], split[1]))

    pairs, validation_errors = _validate_labeled(labeled)
    errors.extend(validation_errors)
    if errors:
        raise ValueError(" ".join(errors))
    if not pairs:
        raise ValueError("At least one example pair is required.")
    return pairs


def parse_examples_payload(items: Iterable[Tuple[str, str]]) -> List[ExamplePair]:
    """Validate a structured list of ``(input, output)`` pairs.

    Same rules as :func:`parse_examples` but for API payloads that already
    carry the pairs as data; errors are labelled ``Example N``.
    """
    labeled = [
        (f"Example {i}", str(src), str(tgt))
        for i, (src, tgt) in enumerate(items, start=1)
    ]
    pairs, errors = _validate_labeled(labeled)
    if errors:
        raise ValueError(" ".join(errors))
    if not pairs:
        raise ValueError("At least one example pair is required.")
    return pairs


def _try_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _all_numeric(pairs: Sequence[ExamplePair]) -> bool:
    return all(_try_float(s) is not None and _try_float(t) is not None for s, t in pairs)


def _looks_numeric_fit(pairs: Sequence[ExamplePair]) -> bool:
    """True when a numeric model explains the pairs to within a tight MSE."""
    xs = [_try_float(s) for s, _ in pairs]
    ys = [_try_float(t) for _, t in pairs]
    if any(v is None for v in xs + ys) or len(pairs) < 2:
        return False
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if len(np.unique(x)) < 2:
        return False
    try:
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        mse = float(np.mean(residuals**2))
        scale = max(float(np.var(y)), 1e-9)
        # Relative MSE: good fit AND meaningfully better than a constant.
        return mse < 1e-2 and mse / scale < 0.01
    except Exception:
        return False


def _match_template(pairs: Sequence[ExamplePair]) -> Optional[str]:
    """Return the key of the first template that reproduces every example.

    This makes template selection deterministic and LLM-free.  All templates
    are evaluated in a single subprocess for speed.
    """
    from sandbox import match_templates

    inputs = [s for s, _ in pairs]
    expected = [t for _, t in pairs]
    return match_templates(templates.TEMPLATES, inputs, expected, timeout=10.0)


def _prefix_boundary_cut(prefix: str) -> int:
    """Length of the prefix shortened to its last separator boundary."""
    for idx in range(len(prefix) - 1, -1, -1):
        if not prefix[idx].isalnum():
            return idx + 1
    return 0


def _suffix_boundary_cut(suffix: str) -> int:
    """Start index of the suffix beginning at its first separator."""
    for idx, ch in enumerate(suffix):
        if not ch.isalnum():
            return idx
    return len(suffix)


def _affix_swap(pairs: Sequence[ExamplePair]) -> Optional[Tuple[str, str, str]]:
    """Detect a fixed prefix/suffix replacement shared by every pair.

    Returns ``("prefix", from, to)`` / ``("suffix", from, to)`` when every
    output is its input with one fixed affix replaced by another (e.g.
    ``SKU-001 -> ITEM-001``), else ``None``.  Pure shape analysis: no column
    names, values or domains are assumed.
    """
    if len(pairs) < 2:
        return None
    ins = [s for s, _ in pairs]
    outs = [t for _, t in pairs]

    def holds(kind: str, frm: str, to: str) -> bool:
        if not frm or not to or frm == to:
            return False
        if kind == "prefix":
            return all(o == to + i[len(frm):] for i, o in zip(ins, outs))
        return all(o == i[: len(i) - len(frm)] + to for i, o in zip(ins, outs))

    # Prefix: shortest shared prefix ending at a separator boundary is
    # preferred ("SKU-" over "SKU-00") but the full shared prefix also counts.
    p_in = os.path.commonprefix(ins)
    p_out = os.path.commonprefix(outs)
    for a, b in (
        (p_in[: _prefix_boundary_cut(p_in)], p_out[: _prefix_boundary_cut(p_out)]),
        (p_in, p_out),
    ):
        if holds("prefix", a, b):
            return ("prefix", a, b)

    # Suffix (applied to the reversed strings, then reversed back).
    s_in = os.path.commonprefix([v[::-1] for v in ins])[::-1]
    s_out = os.path.commonprefix([v[::-1] for v in outs])[::-1]
    for a, b in (
        (s_in[_suffix_boundary_cut(s_in):], s_out[_suffix_boundary_cut(s_out):]),
        (s_in, s_out),
    ):
        if holds("suffix", a, b):
            return ("suffix", a, b)
    return None


def _string_pattern_evidence(pairs: Sequence[ExamplePair]) -> Optional[str]:
    """Cheap deterministic heuristics for obvious string transformations."""
    ins = [s for s, _ in pairs]
    outs = [t for _, t in pairs]

    if all(o == i.upper() for i, o in zip(ins, outs)):
        return "all outputs are the uppercase of their inputs"
    if all(o == i.lower() for i, o in zip(ins, outs)):
        return "all outputs are the lowercase of their inputs"
    if all(o == i.strip() for i, o in zip(ins, outs)):
        return "all outputs are their inputs with surrounding whitespace removed"
    if all(o == i.title() for i, o in zip(ins, outs)):
        return "all outputs are the title-cased form of their inputs"
    if all(o == i.replace(" ", "") for i, o in zip(ins, outs)):
        return "all outputs remove spaces from their inputs"
    swap = _affix_swap(pairs)
    if swap:
        kind, frm, to = swap
        return f"each output replaces the fixed input {kind} {frm!r} with {to!r}"
    if all(i in o for i, o in zip(ins, outs)):
        return "each output contains its input as a substring"
    if all(o in i for i, o in zip(ins, outs)):
        return "each output is a substring of its input"
    if all(o == i[::-1] for i, o in zip(ins, outs)):
        return "each output is the reverse of its input"
    return None


# --------------------------------------------------------------------------- #
# LLM-assisted classification (optional)
#
# ONE model call answers both questions the engine used to conflate:
#   * the display family (String-based / Numerical / Algorithmic / General), and
#   * the TASK the examples demonstrate -- whether they are an exhaustive
#     mapping (a lookup table is correct) or a rule that must generalise to
#     unseen inputs (classification, extraction, rewriting).
# The response is structured JSON and is validated; anything malformed falls
# back to the legacy plain-text family detection and reports task=None so the
# planner can decide without trusting the model.
# --------------------------------------------------------------------------- #
_TASK_INSTRUCTION = '''
Also decide the TASK the examples demonstrate:
- "exact_mapping": the listed inputs are a closed set and each one maps to
  exactly its listed output; the list itself IS the rule.
- "semantic": the outputs are category labels or judgements; the examples
  demonstrate a rule that must be applied to NEW inputs by understanding
  their meaning.
- "extraction": the output is a specific piece of information contained in
  the input (an id, date, name, domain, number, ...).
- "normalization": differently worded inputs are rewritten to one consistent
  canonical form (standardising product descriptions, tidying inconsistent
  text) without changing what they mean.
- "general_text": the output is a rewritten / reformatted / structured
  version of the input (informal -> formal, summarising, restructuring).
- "ambiguous": the examples contradict each other or are too few or too vague
  to determine a single rule.

Important: if the same output label would sensibly apply to many other
inputs, the task is a RULE (semantic/extraction/normalization/general_text),
NOT exact_mapping.
'''

_CLASSIFY_PROMPT_HEADER = """
Analyze the relationship between source and target data pairs and classify the
transformation type into exactly one of these categories:
1. String-based: text operations like case changes, formatting, substring extraction.
2. Numerical: mathematical operations, scaling, rounding, numeric conversions.
3. Algorithmic: known conversions (dates, bases, encodings) implementable as a
   deterministic algorithm.
4. General: knowledge or meaning-based tasks (classification, sentiment,
   intent, entity mapping, lookups).
"""


def _parse_classify_json(
    text: str,
) -> Optional[Tuple[Optional[str], Optional[str], List[str]]]:
    """Extract and validate ``(class, task, labels)`` from a JSON response."""
    import json as _json
    import re as _re

    from transform.spec import VALID_TASKS

    cleaned = text.strip()
    match = _re.search(r"```(?:json)?\s*(.*?)```", cleaned, _re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = _json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    cls = str(data.get("class") or data.get("classification") or "").strip()
    family = next((c for c in VALID_CLASSES if c.lower() == cls.lower()), None)

    task = str(data.get("task") or "").strip().lower()
    if task not in VALID_TASKS:
        task = None

    if family is None:
        # Models (notably mistral) often put the TASK name -- e.g. "semantic"
        # -- or a synonym ("classification") in the "class" field.  The family
        # is display-only; the meaningful part is the task/labels pair, which
        # must not be discarded because of a wrong family label.
        cls_lower = cls.lower()
        if cls_lower in VALID_TASKS:
            if task is None:
                task = cls_lower
        elif task is None:
            # Neither a family nor a task: the answer is genuinely unusable.
            return None
        family = "General"

    raw_labels = data.get("labels")
    labels: List[str] = []
    if isinstance(raw_labels, (list, tuple)) and all(
        isinstance(x, str) and x.strip() for x in raw_labels
    ):
        labels = list(dict.fromkeys(x.strip() for x in raw_labels))[:30]
    return family, task, labels


def _llm_infer(
    pairs: Sequence[ExamplePair],
    override: Optional[dict] = None,
    budget: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[str], List[str], str]:
    """One model call: ``(class, task, labels, reason)``; ``reason`` on None."""
    import json as _json

    import llm

    if not llm.llm_available() and not (override or {}).get("api_key"):
        return None, None, [], "no LLM provider is configured (OPENAI_API_KEY or Ollama)"

    serialized = ", ".join(
        f'({_json.dumps(s)} -> {_json.dumps(t)})' for s, t in pairs[:12]
    )
    from transform.spec import VALID_TASKS

    prompt = (
        _CLASSIFY_PROMPT_HEADER
        + _TASK_INSTRUCTION
        + "\nSource -> Target pairs:\n"
        + serialized
        + "\n\nRespond with ONLY JSON -- one JSON object and no other text:\n"
        '{"class": "...", "task": "...", "labels": [], "reason": "..."}\n'
        f'"class" MUST be exactly one of: {", ".join(VALID_CLASSES)}.\n'
        f'"task" MUST be exactly one of: {", ".join(VALID_TASKS)}.\n'
        '"labels" must list the complete closed set of allowed output labels '
        'when task is "semantic" and the outputs are category labels; '
        "otherwise use [].\n"
    )
    try:
        text = llm.invoke(prompt, max_tokens=200, override=override, budget=budget)
    except Exception as exc:
        # Surface the provider's actual failure (e.g. exhausted credits vs a
        # transient rate limit) so the UI can give actionable guidance.
        reason = str(exc).strip() or "the model did not respond"
        logger.warning("LLM classification failed: %s", reason[:300])
        return None, None, [], reason

    parsed = _parse_classify_json(text)
    if parsed is not None and parsed[1] is not None:
        family, task, labels = parsed
        return family, task, labels, ""

    # The model answered but in a form we cannot use (unrecognized "class"
    # and/or a missing "task").  One corrective retry shows it exactly what
    # was wrong and the only acceptable values; skipped when the per-request
    # budget has no call left.
    retried_text: Optional[str] = None
    budget_left = budget is None or int(
        budget.get("remaining", llm.settings.llm_max_calls_per_request)
    ) > 0
    if budget_left:
        corrective = (
            prompt
            + "\nYour previous response could not be used:\n"
            + text.strip()[:600]
            + '\nRespond with ONLY JSON again: "class" MUST be exactly one '
            f'of: {", ".join(VALID_CLASSES)}; "task" MUST be exactly one '
            f'of: {", ".join(VALID_TASKS)}.'
        )
        try:
            retried_text = llm.invoke(
                corrective, max_tokens=300, override=override, budget=budget
            )
        except Exception as exc:
            logger.warning(
                "LLM corrective classification retry failed: %s",
                str(exc).strip()[:300],
            )
        else:
            second = _parse_classify_json(retried_text)
            if second is not None and second[1] is not None:
                family, task, labels = second
                return family, task, labels, ""

    if parsed is not None:
        # Family usable but task missing: keep the family -- the planner
        # reports the missing task rather than losing the whole answer.
        family, task, labels = parsed
        return family, task, labels, ""

    # Legacy fallback: a plain class name in free text (never trusted for task).
    for candidate in (retried_text, text):
        if not candidate:
            continue
        lowered = candidate.strip().lower()
        for cls in VALID_CLASSES:
            if cls.lower() in lowered:
                return cls, None, [], ""
    return None, None, [], "the LLM did not return a recognized class name"


def _llm_classify(
    pairs: Sequence[ExamplePair],
    override: Optional[dict] = None,
    budget: Optional[dict] = None,
) -> Tuple[Optional[str], str, Optional[str], List[str]]:
    """Return ``(class, reason, task, labels)``; ``reason`` explains None."""
    family, task, labels, reason = _llm_infer(pairs, override, budget)
    return family, reason, task, labels


def _llm_task_classify(
    pairs: Sequence[ExamplePair],
    override: Optional[dict] = None,
    budget: Optional[dict] = None,
) -> Tuple[Optional[str], List[str], List[str], str]:
    """Return ``(task, labels, ambiguities, reason)`` for the planner."""
    _family, task, labels, reason = _llm_infer(pairs, override, budget)
    if task is None:
        return None, [], [], reason or "the model did not identify the task"
    ambiguities: List[str] = []
    if task == "ambiguous":
        ambiguities.append(
            "The model judged these examples ambiguous: they contradict each "
            "other or are too few to determine one rule."
        )
    return task, labels, ambiguities, ""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def classify(
    pairs: Sequence[ExamplePair],
    allow_llm: bool = True,
    llm_override: Optional[dict] = None,
) -> ClassificationResult:
    """Classify example pairs, preferring deterministic detectors.

    ``llm_override`` is an optional transient provider config (e.g. a personal
    API key for this request only) passed through to :mod:`llm`.
    """
    import llm as _llm

    budget: Optional[dict] = None
    if allow_llm:
        budget = {"remaining": _llm.settings.llm_max_calls_per_request}
    evidence: List[str] = []

    # 1. Deterministic algorithmic template match (highest specificity).
    matched = _match_template(pairs)
    if matched:
        evidence.append(f"matched the built-in '{matched}' conversion template exactly")
        return ClassificationResult(
            classification="Algorithmic",
            evidence=evidence,
            template=matched,
            used_llm=False,
        )

    # 2. Numerical fit.
    if _all_numeric(pairs) and _looks_numeric_fit(pairs):
        evidence.append("a linear/numeric model fits the pairs within tolerance")
        try:
            evidence.append(numerical.describe_fit(pairs))
        except Exception:
            pass
        return ClassificationResult(
            classification="Numerical", evidence=evidence, used_llm=False
        )

    # 3. Date-only pairs -> algorithmic (date handling).
    if all(is_date(s) and is_date(t) for s, t in pairs):
        evidence.append("all inputs and outputs parse as dates")
        return ClassificationResult(
            classification="Algorithmic", evidence=evidence, used_llm=False
        )

    # 4. Deterministic string heuristics.
    str_ev = _string_pattern_evidence(pairs)
    if str_ev:
        evidence.append(str_ev)
        return ClassificationResult(
            classification="String-based", evidence=evidence, used_llm=False
        )

    # 5. Numeric inputs but no good numeric fit -> still likely numerical intent.
    if _all_numeric(pairs):
        evidence.append("all pairs are numeric, but no simple numeric model fits well")
        try:
            evidence.append(numerical.describe_fit(pairs))
        except Exception:
            pass
        return ClassificationResult(
            classification="Numerical", evidence=evidence, used_llm=False
        )

    # 6. LLM fallback (optional).
    llm_error: Optional[str] = None
    if allow_llm or (llm_override or {}).get("api_key"):
        llm_cls, reason, task, labels = _llm_classify(
            pairs, override=llm_override, budget=budget
        )
        if llm_cls:
            # A deterministic template / numeric / date / string detector did
            # NOT match (steps 1-5 all fell through to reach here), so an
            # "Algorithmic" answer from the model cannot be backed by a real
            # algorithm -- honouring it would make generate() author a fake
            # transform(x) that fabricates (or nulls out) unseen inputs.  The
            # spec requires lookup/deterministic handling only for TRUE
            # patterns, so we downgrade to the rule-based "General" family and
            # let the TASK field carry the meaning.
            if llm_cls == "Algorithmic":
                evidence.append(
                    "the model suggested 'Algorithmic' but no deterministic "
                    "template/algorithm matched, so the General (rule-based) "
                    "path is used instead"
                )
                llm_cls = "General"
            evidence.append("classified by the LLM (no deterministic detector matched)")
            if task:
                evidence.append(f"the LLM identified the task as '{task}'")
            return ClassificationResult(
                classification=llm_cls,
                evidence=evidence,
                used_llm=True,
                task=task,
                labels=labels,
            )
        # A model was expected but could not answer (rate limit, quota,
        # network, unusable response).  Surface the reason so the UI can say
        # WHAT failed instead of implying nothing is configured.
        llm_error = reason
        evidence.append(
            f"no deterministic detector matched and {reason}; "
            "defaulting to 'General'. Provide more examples if this is wrong."
        )
    else:
        evidence.append("no deterministic detector matched (LLM disabled)")

    return ClassificationResult(
        classification="General",
        evidence=evidence,
        used_llm=False,
        llm_error=llm_error,
    )
