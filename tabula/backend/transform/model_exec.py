"""Model-backed execution for transformations that cannot be Python code.

Some transformations cannot be expressed as a reusable deterministic function
inferred from a handful of examples -- classification, sentiment, intent,
entity extraction and free-form rewriting all require understanding meaning.
The previous engine generated ``lookup_table.get(x, 'UNKNOWN')`` for these,
which silently labelled every unseen input as UNKNOWN.  This module executes
them with the configured model instead, with the guarantees the deterministic
sandbox path gives for code:

* values are processed in batches with structured (JSON) output;
* every value is addressed by a STABLE row id (``GEN-001`` ...) built from its
  position in the original series, and responses are matched by that id;
* responses are validated as a strict bijection onto the batch's row ids --
  a dropped, duplicated, unknown or reordered entry is rejected, so a
  malformed response can never shift or corrupt rows;
* per-row output is then validated (allowed label set, output type, and
  null-for-non-empty) before it is accepted;
* per-row STATUS (ok / review / failed / blank / null) and the model's
  advisory confidence are tracked so partial success and low-confidence rows
  are visible -- nothing is silently converted to null or dropped;
* a malformed response is retried once with a corrective instruction, then
  the whole batch is reported as failed -- never guessed;
* null / empty values pass through unchanged;
* one bad batch or one bad row never corrupts the successful rows around it;
  every failure is reported through the same ``ApplyReport`` the deterministic
  path uses, so preview / apply / export stay consistent;
* the provider that actually served the rows is recorded on the report.

The module never logs prompts or raw model output (they may contain user data)
and never sends secrets -- provider configuration stays in :mod:`llm`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

from transform.apply import ApplyReport
from transform.spec import KIND_NORMALIZATION, KIND_SEMANTIC, TransformationSpec

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 25
_MAX_ERROR_MESSAGES = 20
_MAX_EXAMPLES_IN_PROMPT = 20
# Model outputs with a self-reported confidence below this are accepted but
# flagged for human review (spec §8).  Advisory only, never calibrated.
_REVIEW_CONFIDENCE = 0.6


class ModelExecutionError(RuntimeError):
    """Raised when a model-backed transformation cannot be applied at all."""


# --------------------------------------------------------------------------- #
# Value semantics
# --------------------------------------------------------------------------- #
def _is_missing(value: Any) -> bool:
    """True for None / NaN / NaT (never for empty strings -- those are data)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _is_blank(value: Any) -> bool:
    """Empty or whitespace-only strings: preserved, never sent to the model."""
    return isinstance(value, str) and not value.strip()


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def _task_instructions(spec: TransformationSpec) -> str:
    if spec.kind == KIND_SEMANTIC:
        if spec.allowed_labels:
            labels = ", ".join(json.dumps(l) for l in spec.allowed_labels)
            allowed_line = f"Allowed labels: {labels}\n"
        else:
            allowed_line = "Allowed labels: (infer the closed label set from the examples)\n"
        return (
            "Task: classify each input into exactly ONE label.\n"
            + allowed_line
            + "- Decide by MEANING, not by exact string matching. Inputs that "
            "do not appear in the examples must still be classified.\n"
            + "- Never output a label outside the allowed set.\n"
        )
    if spec.kind == KIND_NORMALIZATION:
        return (
            "Task: normalize each input to one consistent canonical form.\n"
            "- Rewrite differently worded inputs to the same standard phrasing "
            "shown by the examples, preserving meaning.\n"
            "- Decide by MEANING, not by exact string matching; unseen inputs "
            "must still be normalized.\n"
        )
    return (
        "Task: apply the transformation demonstrated by the examples to each "
        "input.\n"
        "- Follow the pattern shown by the examples and generalise it to "
        "unseen inputs; do not match exact strings.\n"
        "- Preserve the output structure, type and formatting shown by the "
        "examples.\n"
    )


def _row_id(position: int) -> str:
    """Stable identifier for a row's position in the ORIGINAL series.

    Built from the global position (not the batch-local index) so an id
    means the same row across every batch, and a reordered model response
    can never be mistaken for a different row.
    """
    return f"GEN-{position + 1:03d}"


def _build_prompt(
    spec: TransformationSpec,
    examples: Sequence[Tuple[str, str]],
    batch: Sequence[Tuple[str, str]],
    correction: Optional[str] = None,
) -> str:
    """Build one batched prompt.

    ``batch`` is a sequence of ``(row_id, value)`` pairs.  The response
    contract is row-id-based (spec §7): the model returns one object per
    input id, so a dropped / duplicated / reordered entry is detected
    instead of silently shifting every later row.  The plain input array is
    still echoed so the model has the values in a simple form too.
    """
    lines: List[str] = [
        "You are applying a data transformation to one column of a dataset.",
        _task_instructions(spec),
    ]
    shown = list(examples[:_MAX_EXAMPLES_IN_PROMPT])
    if shown:
        lines.append(
            "Examples demonstrating the intended rule (input -> output):"
        )
        lines.extend(
            f"  {json.dumps(s, ensure_ascii=False)} -> {json.dumps(t, ensure_ascii=False)}"
            for s, t in shown
        )
    else:
        lines.append("No examples were provided; infer the rule from the task.")
    lines.append("")
    lines.append(
        f"Transform these {len(batch)} values. Return ONLY a JSON object of the "
        'form {"results": [{"row_id": "<id>", "output": <value>, '
        '"confidence": <0..1>}, ...]} with EXACTLY '
        f"{len(batch)} results, one for each row_id below (in any order). Use "
        "null as the output only for a value you genuinely cannot transform. "
        "Set confidence to your own 0-1 estimate of how certain you are of "
        "that output."
    )
    if correction:
        lines.append("")
        lines.append(
            f"Your previous response was rejected ({correction}). Return ONLY "
            "the JSON object, correctly formatted."
        )
    lines.append("")
    lines.append("Row ids (paired with the inputs below, in order):")
    lines.append(json.dumps([rid for rid, _v in batch], ensure_ascii=False))
    lines.append("Inputs (JSON array, in order):")
    lines.append(json.dumps([v for _rid, v in batch], ensure_ascii=False))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Response parsing / validation
#
# Three accepted response shapes, tried most-specific first:
#   1. {"results": [{"row_id": "GEN-001", "output": ..., "confidence": 0.9}]}
#      -- the preferred row-id contract: matched by stable id, so a dropped
#         or duplicated entry is caught instead of shifting later rows;
#   2. {"outputs": [...]} or a bare JSON array -- positional, count-validated;
#   3. [{"i": 0, "value": ...}] -- index-mapped, validated as a bijection.
# Every shape is validated as a strict bijection onto this batch's row ids /
# positions before any value is accepted.
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> Optional[Any]:
    """Return the parsed JSON payload (list or dict) or None."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    # Preferred shape: a JSON object ({"results": [...]} or {"outputs": [...]}).
    if cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(cleaned[start : end + 1])
            except ValueError:
                data = None
            if isinstance(data, dict):
                return data
    # Fallback: a bare JSON array.
    start, end = cleaned.find("["), cleaned.rfind("]")
    if 0 <= start < end:
        try:
            data = json.loads(cleaned[start : end + 1])
        except ValueError:
            return None
        if isinstance(data, list):
            return data
    return None


def _coerce_confidence(raw: Any) -> Optional[float]:
    """Coerce a model-provided confidence to 0-1; None when absent/unusable.

    An LLM's self-reported confidence is NOT a calibrated probability -- it
    is used only to flag rows for human review, never to accept or reject a
    value on its own.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value > 1.0 and value <= 100.0:
        value = value / 100.0  # accept 0-100 style confidences
    if value != value:  # NaN
        return None
    return max(0.0, min(1.0, value))


def _parse_results(
    text: str, row_ids: Sequence[str]
) -> Tuple[Optional[List[Any]], Optional[List[Optional[float]]], str]:
    """Validate a model response against this batch's row ids.

    Returns ``(outputs, confidences, error)``.  ``outputs`` is aligned to
    ``row_ids`` (same order) when successful; ``confidences`` may contain
    None for rows where the model gave none.  On any validation failure the
    whole batch is rejected (``outputs is None``) so a malformed response
    can never be silently mis-aligned.
    """
    expected = list(row_ids)
    n = len(expected)
    data = _extract_json(text)
    if data is None:
        return None, None, "the response was not valid JSON"

    # ---- 1. Row-id contract (preferred) --------------------------------- #
    results = None
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        results = data["results"]
    elif isinstance(data, list) and data and all(
        isinstance(item, dict) and "row_id" in item for item in data
    ):
        results = data

    if results is not None:
        by_id: dict = {}
        for item in results:
            if "row_id" not in item or "output" not in item:
                return None, None, "a result was missing 'row_id'/'output'"
            rid = str(item["row_id"]).strip()
            if rid in by_id:
                return None, None, f"row_id {rid!r} was answered more than once"
            by_id[rid] = (
                item["output"],
                _coerce_confidence(item.get("confidence")),
            )
        unknown = [rid for rid in by_id if rid not in set(expected)]
        if unknown:
            return None, None, (
                f"the response contained unknown row_id(s): "
                f"{', '.join(sorted(unknown)[:5])}"
            )
        missing = [rid for rid in expected if rid not in by_id]
        if missing:
            return None, None, (
                f"the response omitted {len(missing)} row(s): "
                f"{', '.join(missing[:5])}"
            )
        outputs = [by_id[rid][0] for rid in expected]
        confs = [by_id[rid][1] for rid in expected]
        return outputs, confs, ""

    # ---- Legacy shapes need a list -------------------------------------- #
    if isinstance(data, dict):
        if isinstance(data.get("outputs"), list):
            data = data["outputs"]
        else:
            return None, None, "the response JSON did not contain results/outputs"
    if not isinstance(data, list):  # pragma: no cover - defensive
        return None, None, "the response was not a JSON list"

    if data and all(isinstance(item, dict) for item in data) and "i" in data[0]:
        # Index-mapped objects: strict bijection onto range(n).
        by_index: List[Any] = [None] * n
        conf_by_index: List[Optional[float]] = [None] * n
        seen: set = set()
        for item in data:
            if "i" not in item or "value" not in item:
                return None, None, "an index-mapped item was missing 'i'/'value'"
            try:
                idx = int(item["i"])
            except (TypeError, ValueError):
                return None, None, "an index-mapped item had a non-integer 'i'"
            if not (0 <= idx < n):
                return None, None, f"index {idx} is out of range for {n} inputs"
            if idx in seen:
                return None, None, f"index {idx} was answered more than once"
            seen.add(idx)
            by_index[idx] = item["value"]
            conf_by_index[idx] = _coerce_confidence(item.get("confidence"))
        if len(seen) != n:
            return None, None, (
                f"expected {n} indexed outputs but received {len(seen)}"
            )
        return by_index, conf_by_index, ""

    # ---- Positional list: strict count check ---------------------------- #
    if len(data) != n:
        return None, None, f"expected {n} outputs but received {len(data)}"
    return data, [None] * n, ""


def _normalize_output(
    raw: Any, spec: TransformationSpec, confidence: Optional[float] = None
) -> Tuple[Any, Optional[str], bool]:
    """Validate one output value; return ``(value, error, needs_review)``.

    ``needs_review`` is set when the model reported low confidence for an
    otherwise-valid output (spec §8: accept, but flag for a human).  A
    self-reported confidence is advisory only -- it never turns an invalid
    output into an accepted one, and its absence never blocks acceptance.
    """
    if raw is None:
        return None, "the model returned null for a non-empty input", False
    if isinstance(raw, bool):
        text = "true" if raw else "false"
    elif isinstance(raw, (str, int, float)):
        text = str(raw)
    else:
        return None, (
            f"the model returned an unexpected output type ({type(raw).__name__})"
        ), False
    if spec.allowed_labels:
        canonical = {l.lower(): l for l in spec.allowed_labels}.get(text.strip().lower())
        if canonical is None:
            shown = ", ".join(spec.allowed_labels[:8])
            return None, (
                f"the model returned {text!r}, which is not one of the allowed "
                f"labels ({shown})"
            ), False
        text = canonical
    needs_review = confidence is not None and confidence < _REVIEW_CONFIDENCE
    return text, "", needs_review


def _max_tokens_for(batch_len: int) -> int:
    return max(300, min(4000, 40 * batch_len + 200))


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _infer_batch(
    spec: TransformationSpec,
    examples: Sequence[Tuple[str, str]],
    batch: Sequence[Tuple[str, str]],
    *,
    override: Optional[dict],
    budget: Optional[dict],
) -> Tuple[Optional[List[Any]], Optional[List[Optional[float]]], str]:
    """Run one batch through the model; retry once on an unusable response.

    ``batch`` is a list of ``(row_id, value)`` pairs.  Returns
    ``(outputs, confidences, error)`` aligned to ``batch`` on success; both
    are None on failure so the caller can report the batch without guessing.
    """
    import llm as _llm

    row_ids = [rid for rid, _v in batch]
    correction: Optional[str] = None
    error = "no response"
    for _attempt in range(2):
        prompt = _build_prompt(spec, examples, batch, correction=correction)
        try:
            text = _llm.invoke(
                prompt,
                max_tokens=_max_tokens_for(len(batch)),
                override=override,
                budget=budget,
            )
        except _llm.LLMUnavailableError:
            raise
        except Exception as exc:
            reason = str(exc).strip() or "no usable response"
            logger.warning(
                "Model-backed execution failed (%s): %s",
                type(exc).__name__,
                reason[:300],
            )
            return None, None, f"the model request failed: {reason}"
        outputs, confs, error = _parse_results(text or "", row_ids)
        if outputs is not None:
            return outputs, confs, ""
        correction = error
    return None, None, f"the model returned an unusable response twice ({error})"


def apply_model_backed(
    series: "pd.Series",
    spec: TransformationSpec,
    examples: Sequence[Tuple[str, str]] = (),
    *,
    llm_override: Optional[dict] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    budget: Optional[dict] = None,
) -> ApplyReport:
    """Apply a model-backed transformation to every value of ``series``.

    Returns an :class:`ApplyReport` (the same report type the deterministic
    sandbox path returns) so callers treat both execution modes identically.

    Guarantees (spec §7, §8, §10):

    * every value is addressed by a STABLE row id (``GEN-001`` ... built from
      its position in the original series), so a reordered or dropped model
      entry is detected rather than shifting later rows;
    * responses are validated as a strict bijection onto the batch's row ids
      before any value is accepted, then validated per-row (allowed labels,
      output type, null-for-non-empty);
    * nulls and blanks pass through untouched and are never sent to the model;
    * per-row status (ok / review / failed / blank / null) and confidence are
      tracked so partial success is visible -- nothing is silently dropped;
    * one failed batch or one failed row never corrupts the rows around it;
    * the provider that actually served the rows is recorded in the report.
    """
    import llm as _llm

    values = list(series)
    total = len(values)
    if total == 0:
        return ApplyReport(
            series=pd.Series([], dtype=object), total=0, failed=0, errors=[]
        )

    results: List[Any] = [None] * total
    confidences: List[Optional[float]] = [None] * total
    statuses: List[str] = ["ok"] * total
    failed = 0
    review = 0
    errors: List[str] = []
    warnings: List[str] = []
    providers_used: List[str] = []

    def note(message: str) -> None:
        if message not in errors and len(errors) < _MAX_ERROR_MESSAGES:
            errors.append(message)

    if budget is None:
        budget = {"remaining": _llm.settings.llm_max_calls_per_request}

    # Nulls and blanks pass through untouched (never counted as failures).
    pending: List[int] = []
    for idx, value in enumerate(values):
        if _is_missing(value):
            results[idx] = None
            statuses[idx] = "null"
        elif _is_blank(value):
            results[idx] = value
            statuses[idx] = "blank"
        else:
            pending.append(idx)

    batch_size = max(1, int(batch_size))
    stop_reason: Optional[str] = None
    for start in range(0, len(pending), batch_size):
        if stop_reason:
            for idx in pending[start:]:
                statuses[idx] = "failed"
            failed += len(pending) - start
            note(stop_reason)
            break
        chunk = pending[start : start + batch_size]
        # Stable row ids built from the ORIGINAL position, so the same row
        # keeps the same id across batches and a reordered response can never
        # be mistaken for a different row.
        batch = [(_row_id(i), str(values[i])) for i in chunk]
        try:
            outputs, confs, error = _infer_batch(
                spec, examples, batch, override=llm_override, budget=budget
            )
        except _llm.LLMUnavailableError as exc:
            # No usable provider (or the per-request call budget ran out):
            # nothing further can succeed -- report the rest as failed.
            stop_reason = str(exc)
            for idx in pending[start:]:
                statuses[idx] = "failed"
            failed += len(pending) - start
            note(stop_reason)
            break
        provider = _llm.last_provider()
        if provider and provider not in providers_used:
            providers_used.append(provider)
        if outputs is None:
            for idx in chunk:
                statuses[idx] = "failed"
            failed += len(chunk)
            note(f"{len(chunk)} rows could not be transformed: {error}")
            continue
        for position, (idx, raw) in enumerate(zip(chunk, outputs)):
            confidence = confs[position] if confs else None
            value, row_error, needs_review = _normalize_output(
                raw, spec, confidence
            )
            confidences[idx] = confidence
            if row_error:
                results[idx] = None
                statuses[idx] = "failed"
                failed += 1
                note(row_error)
            else:
                results[idx] = value
                if needs_review:
                    statuses[idx] = "review"
                    review += 1
                else:
                    statuses[idx] = "ok"

    if review:
        warnings.append(
            f"{review} row(s) were accepted with low model confidence and are "
            "flagged for review."
        )
    if failed:
        warnings.append(
            f"{failed} of {total} row(s) could not be transformed and are "
            "left empty; see the row errors for the reason."
        )

    out = pd.Series(results, index=series.index, dtype=object)
    # Prefer the provider that actually served rows; report the fallback
    # ("ollama") when the cloud path fell back locally.
    provider = providers_used[-1] if providers_used else ""
    return ApplyReport(
        series=out,
        total=total,
        failed=failed,
        errors=errors,
        review=review,
        provider=provider,
        family=spec.family,
        confidences=confidences,
        row_status=statuses,
        warnings=warnings,
    )
