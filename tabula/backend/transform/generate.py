"""Transformation code generation.

For each classification family we produce a ``transform`` function.  Where a
deterministic generator suffices (numerical fits, algorithmic templates, exact
lookup tables, obvious string operations) **no LLM is used**.  The LLM is only
consulted for genuinely open-ended string patterns and for deciding whether an
open-ended ``General`` task is an exact mapping or a rule that must generalise
(semantics / extraction / rewriting).

Two invariants hold for every generated transformation:

* code that will be executed in the sandbox is validated against **all** of
  the user's examples before it is returned -- generating code is not proof
  that the transformation is correct;
* a ``General`` task is routed by :mod:`transform.planner`: a lookup table is
  only built when the examples ARE the mapping.  Rule-inference tasks are
  executed by the configured model at apply time instead of being reduced to
  ``lookup_table.get(x, 'UNKNOWN')``, which silently mislabels every unseen
  input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd

from sandbox import SandboxError, validate_transformation_code
from transform import numerical, templates
from transform.classify import ExamplePair, _affix_swap
from transform.spec import (
    KIND_ALGORITHMIC,
    KIND_EXACT_MAPPING,
    KIND_NUMERIC,
    KIND_TEXT_FALLBACK,
    KIND_TEXT_REPLACE,
    STRATEGY_DETERMINISTIC,
    STRATEGY_MODEL,
    TransformationSpec,
)

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when a transformation cannot be generated (e.g. LLM unavailable)."""


@dataclass
class GeneratedTransformation:
    """A generated transformation plus metadata about how it was produced."""

    code: str
    classification: str
    # Human-readable evidence / limitations.  Not a calibrated confidence.
    description: str
    used_llm: bool = False
    # Optional matched template key for Algorithmic transformations.
    template: Optional[str] = None
    # Candidate alternatives when the rule was ambiguous.
    candidates: List[str] = field(default_factory=list)
    # Execution strategy chosen by the planner.
    strategy: str = STRATEGY_DETERMINISTIC
    # Explicit transformation specification (inference -> execution contract).
    spec: Optional[TransformationSpec] = None
    # The examples the transformation was inferred from (stored with it so
    # model-backed execution can demonstrate the rule at apply time).
    examples: List[ExamplePair] = field(default_factory=list)
    # Ambiguities the user should resolve (never silently ignored).
    ambiguities: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validation of generated code against the user's examples
# --------------------------------------------------------------------------- #
def validate_code_against_examples(
    code: str, pairs: Sequence[ExamplePair]
) -> List[str]:
    """Return mismatch messages for ``code`` vs every example (empty == exact).

    This is the inference-to-execution boundary check: code is only accepted
    when it actually reproduces every example the user supplied.
    """
    from transform.apply import ApplyError, apply_to_series

    if not pairs:
        return []
    try:
        report = apply_to_series(
            code, pd.Series([s for s, _ in pairs], dtype=object)
        )
    except ApplyError as exc:
        return [f"the code failed to run on the examples: {exc}"]
    mismatches: List[str] = []
    for (s, t), got in zip(pairs, report.series.tolist()):
        if str(got) != str(t):
            mismatches.append(f"input {s!r} produced {got!r}, expected {t!r}")
    return mismatches


# --------------------------------------------------------------------------- #
# Deterministic string generators
# --------------------------------------------------------------------------- #
def _try_deterministic_string(pairs: Sequence[ExamplePair]) -> Optional[str]:
    """Return code for an obvious string operation, or None."""
    ins = [s for s, _ in pairs]
    outs = [t for _, t in pairs]

    def code(body: str) -> str:
        return (
            "def transform(x):\n"
            "    try:\n"
            "        s = '' if x is None else str(x)\n"
            f"        {body}\n"
            "    except Exception:\n"
            "        return x\n"
        )

    if all(o == i.upper() for i, o in zip(ins, outs)):
        return code("return s.upper()")
    if all(o == i.lower() for i, o in zip(ins, outs)):
        return code("return s.lower()")
    if all(o == i.strip() for i, o in zip(ins, outs)):
        return code("return s.strip()")
    if all(o == i.title() for i, o in zip(ins, outs)):
        return code("return s.title()")
    if all(o == i.replace(" ", "") for i, o in zip(ins, outs)):
        return code('return s.replace(" ", "")')
    if all(o == i[::-1] for i, o in zip(ins, outs)):
        return code("return s[::-1]")
    # Fixed prefix/suffix replacement (e.g. SKU-xxxx -> ITEM-xxxx): a reusable
    # deterministic rule inferred from shape, never a closed lookup table.
    swap = _affix_swap(pairs)
    if swap:
        kind, frm, to = swap
        if kind == "prefix":
            return code(
                f"if s.startswith({frm!r}):\n"
                f"            return {to!r} + s[len({frm!r}):]\n"
                "        return s"
            )
        return code(
            f"if s.endswith({frm!r}):\n"
            f"            return s[:len(s) - len({frm!r})] + {to!r}\n"
            "        return s"
        )
    return None


# --------------------------------------------------------------------------- #
# Deterministic lookup generator (General, exact mapping only)
# --------------------------------------------------------------------------- #
def _exact_lookup_code(pairs: Sequence[ExamplePair]) -> str:
    """Build an exact-match lookup table from the examples (no LLM, no guessing).

    Only used when the planner determined the examples ARE the mapping
    (``kind == exact_mapping``).  Unmapped inputs are flagged as 'UNKNOWN'
    rather than guessed -- that is the correct behaviour for an exhaustive
    mapping, and only for an exhaustive mapping.
    """
    mapping = {s: t for s, t in pairs}
    items = ", ".join(f"{k!r}: {v!r}" for k, v in mapping.items())
    return (
        "lookup_table = {" + items + "}\n\n"
        "def transform(x):\n"
        "    return lookup_table.get(x, 'UNKNOWN')\n"
    )


def _model_backed_code(
    spec: TransformationSpec, pairs: Sequence[ExamplePair]
) -> str:
    """Honest, runnable code for a model-backed transformation (spec §11).

    Rather than a placeholder comment, this emits the REAL execution routine
    the engine performs: batch the values, address each by a stable row id,
    ask the model for structured JSON, validate it as a bijection onto the
    row ids (plus the allowed label set when the task defines one), and align
    results row-for-row.  The model call is injected as an ``invoke`` callable
    so the code is self-contained and does not hardcode any provider or key;
    a caller wires it to whichever provider the engine used.

    The generated code, the explanation and the actual execution therefore
    describe the same transformation.
    """
    import json as _json

    labels = list(spec.allowed_labels or [])
    label_block = (
        "ALLOWED_LABELS = " + _json.dumps(labels) + "\n"
        if labels
        else "ALLOWED_LABELS = []  # free-form output; no closed label set\n"
    )
    task = spec.task or ("semantic" if spec.kind == KIND_SEMANTIC else "general_text")
    examples_literal = _json.dumps([[s, t] for s, t in pairs], ensure_ascii=False)

    return f'''"""Model-backed transformation generated by TabulaX.

Family: {spec.family}
Task:   {task}

This transformation cannot be a pure deterministic rule: the examples
demonstrate a meaning-based behaviour that must generalise to UNSEEN inputs.
TabulaX therefore applies it with the configured AI model.  The routine below
is exactly what the engine runs at apply time -- batch, address each row by a
stable id, request structured JSON, validate it, and align row-for-row.  Pass
your own ``invoke(prompt) -> str`` (wired to OpenAI, Ollama, or any chat
model) to execute it; the engine's provider layer already handles fallback.
"""

import json

BATCH_SIZE = 25
{label_block}EXAMPLES = {examples_literal}  # (input, output) demonstrations


def _prompt(batch):
    labels = ("Allowed labels: " + ", ".join(map(repr, ALLOWED_LABELS)))
    if not ALLOWED_LABELS:
        labels = "Allowed labels: (infer from the examples)"
    ex = "\\n".join(
        f"  {{json.dumps(s)}} -> {{json.dumps(t)}}" for s, t in EXAMPLES
    )
    return (
        "You are applying a data transformation to one column.\\n"
        f"Task: {{TASK_INSTRUCTIONS}}\\n"
        f"{{labels}}\\n"
        "Examples:\\n" + ex + "\\n\\n"
        f"Transform these {{len(batch)}} values. Return ONLY JSON of the form "
        '{{\\"results\\": [{{\\"row_id\\": \\"<id>\\", \\"output\\": <value>}}]}} '
        "with exactly one result per row_id, in any order.\\n"
        "Row ids: " + json.dumps([rid for rid, _ in batch]) + "\\n"
        "Inputs:  " + json.dumps([v for _, v in batch])
    )


TASK_INSTRUCTIONS = (
    "classify each input into exactly ONE of the allowed labels by MEANING."
    if ALLOWED_LABELS
    else "apply the rule demonstrated by the examples to each input by meaning."
)


def _parse(text, expected_ids):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{{"), cleaned.rfind("}}")
    data = json.loads(cleaned[start:end + 1]) if 0 <= start < end else None
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError("model returned no results list")
    by_id = {{str(r["row_id"]).strip(): r.get("output") for r in results
             if isinstance(r, dict) and "row_id" in r}}
    missing = [rid for rid in expected_ids if rid not in by_id]
    if missing or len(by_id) != len(expected_ids):
        raise ValueError(f"row-id mismatch (missing {{len(missing)}})")
    return [by_id[rid] for rid in expected_ids]


def _validate(value, row_id):
    if value is None:
        raise ValueError(f"{{row_id}}: model returned null")
    text = str(value)
    if ALLOWED_LABELS:
        canon = {{l.lower(): l for l in ALLOWED_LABELS}}
        text = canon.get(text.strip().lower())
        if text is None:
            raise ValueError(f"{{row_id}}: label outside allowed set")
    return text


def transform(values):
    """Transform a list of values; returns a list of the same length.

    A row the model could not transform is returned as None and is NOT
    silently dropped or shifted -- row order and count are always preserved.
    """
    if not values:
        return []
    invoke = globals().get("INVOKE")  # caller injects the model callable
    if invoke is None:
        raise RuntimeError(
            "Wire INVOKE = lambda prompt: <your model call> before running; "
            "this transformation needs a model provider."
        )
    out = [None] * len(values)
    for start in range(0, len(values), BATCH_SIZE):
        chunk = list(enumerate(values[start:start + BATCH_SIZE], start=start))
        batch = [(f"GEN-{{i + 1:03d}}", str(v)) for i, v in chunk]
        try:
            raw = _parse(invoke(_prompt(batch)), [rid for rid, _ in batch])
            for (rid, _), value in zip(batch, raw):
                out[int(rid.split("-")[1]) - 1] = _validate(value, rid)
        except Exception as exc:  # report, never guess or shift rows
            print(f"batch at row {{start}} failed: {{exc}}")
    return out
'''


# --------------------------------------------------------------------------- #
# LLM-assisted generation (optional)
# --------------------------------------------------------------------------- #
def _llm_generate(
    prompt: str,
    max_tokens: int = 500,
    override: Optional[dict] = None,
    budget: Optional[dict] = None,
) -> Optional[str]:
    import llm
    import re as _re

    if not llm.llm_available() and not (override or {}).get("api_key"):
        return None
    try:
        full = llm.invoke(
            prompt, max_tokens=max_tokens, override=override, budget=budget
        )
    except Exception as exc:
        logger.warning("LLM generation failed: %s", type(exc).__name__)
        return None
    # Strip an echoed prompt, then extract a fenced code block.
    if full.startswith(prompt):
        full = full[len(prompt):]
    match = _re.findall(r"```(?:python)?\s*(.*?)```", full, _re.DOTALL)
    if match:
        return match[-1].strip()
    # Fall back to any def transform block.
    match = _re.search(r"(def transform\(.*?)(?:\n\S|\Z)", full, _re.DOTALL)
    if match:
        return match.group(1).strip()
    return full.strip() or None


def _is_valid(code: Optional[str]) -> bool:
    if not code:
        return False
    try:
        validate_transformation_code(code)
        return True
    except SandboxError:
        return False


def _llm_code_for_examples(
    pairs: Sequence[ExamplePair],
    prompt: str,
    *,
    what: str,
    max_tokens: int = 500,
    override: Optional[dict] = None,
    budget: Optional[dict] = None,
) -> Optional[str]:
    """Author code with the LLM and validate it against ALL examples.

    Returns validated code, or ``None`` when the model produced nothing
    usable.  A first attempt that fails on any example gets ONE corrective
    retry showing the failures; if that also fails, :class:`GenerationError`
    is raised -- a transformation that cannot reproduce the examples is never
    presented as successful.
    """
    code = _llm_generate(prompt, max_tokens=max_tokens, override=override, budget=budget)
    if not _is_valid(code):
        return None
    mismatches = validate_code_against_examples(code, pairs)
    if not mismatches:
        return code
    retry_prompt = (
        prompt
        + "\n\nYour previous attempt failed on these examples: "
        + "; ".join(mismatches[:3])
        + ". Fix the function so it reproduces EVERY example exactly, then "
        "return only the corrected function in a ```python``` code block."
    )
    code2 = _llm_generate(
        retry_prompt, max_tokens=max_tokens, override=override, budget=budget
    )
    if _is_valid(code2) and not validate_code_against_examples(code2, pairs):
        return code2
    raise GenerationError(
        f"The generated {what} did not reproduce your examples "
        f"({'; '.join(mismatches[:2])}). Add more representative examples."
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate(
    pairs: Sequence[ExamplePair],
    classification: str,
    template: Optional[str] = None,
    allow_llm: bool = True,
    llm_override: Optional[dict] = None,
    *,
    treat_as_mapping: bool = False,
    task_hint: Optional[str] = None,
    labels_hint: Optional[Sequence[str]] = None,
    source_column: str = "",
    budget: Optional[dict] = None,
) -> GeneratedTransformation:
    """Generate a transformation for the given pairs and class.

    ``llm_override`` is an optional transient provider config (e.g. a personal
    API key for this request only) passed through to :mod:`llm`.
    ``treat_as_mapping`` forces the General family to treat the examples as an
    exhaustive mapping.  ``task_hint`` / ``labels_hint`` carry a previous
    ``/classify/`` decision so classify -> generate costs one model call.
    """
    import llm as _llm

    classification = classification.strip()
    personal_key = bool((llm_override or {}).get("api_key"))
    if budget is None and (allow_llm or personal_key):
        budget = {"remaining": _llm.settings.llm_max_calls_per_request}

    def det_spec(kind: str, summary: str) -> TransformationSpec:
        return TransformationSpec(
            kind=kind,
            source_column=source_column,
            summary=summary,
            strategy=STRATEGY_DETERMINISTIC,
        )

    # ---- Numerical (always deterministic) -------------------------------- #
    if "numerical" in classification.lower():
        try:
            code = numerical.generate_numeric_code(pairs)
        except numerical.NumericalFitError as exc:
            raise GenerationError(str(exc)) from exc
        if not _is_valid(code):  # pragma: no cover - defensive
            raise GenerationError("Generated numerical code failed validation.")
        description = numerical.describe_fit(pairs)
        return GeneratedTransformation(
            code=code,
            classification="Numerical",
            description=description,
            used_llm=False,
            strategy=STRATEGY_DETERMINISTIC,
            spec=det_spec(KIND_NUMERIC, description),
            examples=list(pairs),
        )

    # ---- Algorithmic (template-first, deterministic) ---------------------- #
    if "algorithmic" in classification.lower():
        if template and template in templates.TEMPLATES:
            code = templates.get_template(template)
            description = (
                f"Uses the built-in '{template}' conversion: "
                + templates.TEMPLATE_DESCRIPTIONS.get(template, "")
            )
            return GeneratedTransformation(
                code=code,
                classification="Algorithmic",
                description=description,
                used_llm=False,
                template=template,
                strategy=STRATEGY_DETERMINISTIC,
                spec=det_spec(KIND_ALGORITHMIC, description),
                examples=list(pairs),
            )
        # Try to match a template deterministically now.
        from transform.classify import _match_template

        matched = _match_template(pairs)
        if matched:
            description = (
                f"Uses the built-in '{matched}' conversion: "
                + templates.TEMPLATE_DESCRIPTIONS.get(matched, "")
            )
            return GeneratedTransformation(
                code=templates.get_template(matched),
                classification="Algorithmic",
                description=description,
                used_llm=False,
                template=matched,
                strategy=STRATEGY_DETERMINISTIC,
                spec=det_spec(KIND_ALGORITHMIC, description),
                examples=list(pairs),
            )
        # No template matched; require the LLM to refine, else error.
        if not (allow_llm or personal_key):
            raise GenerationError(
                "No algorithmic template matched these examples and the LLM is "
                "unavailable. Provide examples matching a known conversion "
                f"({', '.join(templates.list_template_types()[:6])}, ...) or enable an LLM."
            )
        # LLM refinement of a generic algorithmic function.
        example_text = "\n".join(f"Input: {s} -> Output: {t}" for s, t in pairs[:4])
        prompt = (
            "Write a Python function `transform(x)` that reproduces these "
            "input->output transformations exactly:\n"
            f"{example_text}\n\n"
            "Return only the function in a ```python``` code block."
        )
        code = _llm_code_for_examples(
            pairs, prompt, what="algorithmic function",
            override=llm_override, budget=budget,
        )
        if not code:
            raise GenerationError(
                "Could not generate a valid algorithmic transformation. "
                "Provide a few more representative examples."
            )
        return GeneratedTransformation(
            code=code,
            classification="Algorithmic",
            description=(
                "Generated by the LLM from your examples (no built-in template "
                "matched). Verified against every example."
            ),
            used_llm=True,
            strategy=STRATEGY_DETERMINISTIC,
            spec=det_spec(
                KIND_ALGORITHMIC,
                "Open-ended algorithmic rule authored by the LLM and verified "
                "against the examples.",
            ),
            examples=list(pairs),
        )

    # ---- String-based ----------------------------------------------------- #
    if "string" in classification.lower():
        det = _try_deterministic_string(pairs)
        if det:
            return GeneratedTransformation(
                code=det,
                classification="String-based",
                description="Deterministic string rule inferred from the examples "
                "(no LLM used).",
                used_llm=False,
                strategy=STRATEGY_DETERMINISTIC,
                spec=det_spec(
                    KIND_TEXT_REPLACE
                    if _affix_swap(pairs)
                    else KIND_TEXT_FALLBACK,
                    "Deterministic string rule inferred from the shape of the "
                    "examples.",
                ),
                examples=list(pairs),
            )
        if not (allow_llm or personal_key):
            raise GenerationError(
                "This string transformation is not one of the built-in "
                "deterministic rules and the LLM is unavailable. Provide "
                "examples of a simple case/trim/substring rule, or configure "
                "an LLM provider (OPENAI_API_KEY, a personal key, or Ollama)."
            )
        example_text = "\n".join(f"Input: {s} -> Output: {t}" for s, t in pairs[:4])
        prompt = (
            "Write a Python function `transform(x)` that performs this string "
            f"transformation:\n{example_text}\n\n"
            "Use only standard string methods. Return only the function in a "
            "```python``` code block."
        )
        code = _llm_code_for_examples(
            pairs, prompt, what="string transformation",
            max_tokens=400, override=llm_override, budget=budget,
        )
        if not code:
            raise GenerationError(
                "Could not generate a valid string transformation. Provide more examples."
            )
        return GeneratedTransformation(
            code=code,
            classification="String-based",
            description="Generated by the LLM from your examples and verified "
            "against all of them before being accepted.",
            used_llm=True,
            strategy=STRATEGY_DETERMINISTIC,
            spec=det_spec(
                KIND_TEXT_FALLBACK,
                "Open-ended string rule authored by the LLM and verified "
                "against the examples.",
            ),
            examples=list(pairs),
        )

    # ---- General: planner-driven (exact mapping vs rule inference) -------- #
    from transform.planner import Plan, PlanError, plan_examples

    try:
        plan = plan_examples(
            pairs,
            classification_hint=classification,
            template_hint=template,
            allow_llm=allow_llm,
            llm_override=llm_override,
            treat_as_mapping=treat_as_mapping,
            llm_task=task_hint,
            labels_hint=labels_hint,
            source_column=source_column,
            budget=budget,
        )
    except PlanError as exc:
        raise GenerationError(str(exc)) from exc

    return _generate_from_plan(
        plan, pairs,
        allow_llm=allow_llm or personal_key,
        llm_override=llm_override,
        budget=budget,
        source_column=source_column,
    )


def _generate_from_plan(
    plan: "Plan",
    pairs: Sequence[ExamplePair],
    *,
    allow_llm: bool,
    llm_override: Optional[dict] = None,
    budget: Optional[dict] = None,
    source_column: str = "",
) -> GeneratedTransformation:
    """Turn a planner decision into the executable transformation.

    The plan -- not the client-supplied family -- decides what happens: the
    deterministic detectors re-derive Algorithmic / Numerical / String rules
    from the shape of the pairs even when the caller said ``General``.
    """
    from transform.spec import STRATEGY_NONE

    spec = plan.spec
    evidence = "; ".join(plan.evidence)

    if spec.strategy == STRATEGY_NONE:
        raise GenerationError(
            spec.unsupported_reason
            or "The examples do not determine a single transformation rule."
        )

    if spec.strategy == STRATEGY_MODEL:
        # Semantic / extraction / normalization / rewriting tasks: executed by
        # the configured model at apply time (see transform.model_exec).  There
        # is no reusable sandbox rule, and no lookup table that would mislabel
        # every unseen input as UNKNOWN.  The generated code is NOT a comment
        # placeholder -- it is the real batching / row-id / validation routine
        # the engine runs, parameterised by a model `invoke` callable so the
        # code, the explanation and the actual execution all agree (spec §11).
        return GeneratedTransformation(
            code=_model_backed_code(spec, pairs),
            classification=plan.classification,
            description=(
                spec.summary
                + " Rows are processed in batches by the configured AI model "
                "when you apply; inputs unseen in the examples are transformed "
                "by meaning, not by exact match. The generated code shows the "
                "exact batching and validation the engine performs."
                + (f" Evidence: {evidence}." if evidence else "")
            ),
            used_llm=True,
            strategy=STRATEGY_MODEL,
            spec=spec,
            examples=list(pairs),
            ambiguities=list(spec.ambiguities),
        )

    # ---- Deterministic plans re-derive the matching generator ------------- #
    if plan.classification == "Numerical":
        try:
            code = numerical.generate_numeric_code(pairs)
        except numerical.NumericalFitError as exc:
            raise GenerationError(str(exc)) from exc
        if not _is_valid(code):  # pragma: no cover - defensive
            raise GenerationError("Generated numerical code failed validation.")
        description = numerical.describe_fit(pairs)
        return GeneratedTransformation(
            code=code, classification="Numerical", description=description,
            used_llm=False, strategy=STRATEGY_DETERMINISTIC,
            spec=spec, examples=list(pairs),
        )

    if plan.classification == "Algorithmic":
        if plan.template and plan.template in templates.TEMPLATES:
            description = (
                f"Uses the built-in '{plan.template}' conversion: "
                + templates.TEMPLATE_DESCRIPTIONS.get(plan.template, "")
            )
            return GeneratedTransformation(
                code=templates.get_template(plan.template),
                classification="Algorithmic", description=description,
                used_llm=False, template=plan.template,
                strategy=STRATEGY_DETERMINISTIC, spec=spec,
                examples=list(pairs),
            )
        if not allow_llm:
            raise GenerationError(
                "These examples look like an algorithmic conversion but no "
                "built-in template matched and the LLM is unavailable. "
                "Provide examples matching a known conversion "
                f"({', '.join(templates.list_template_types()[:6])}, ...) or "
                "enable an LLM."
            )
        example_text = "\n".join(f"Input: {s} -> Output: {t}" for s, t in pairs[:4])
        prompt = (
            "Write a Python function `transform(x)` that reproduces these "
            "input->output transformations exactly:\n"
            f"{example_text}\n\n"
            "Return only the function in a ```python``` code block."
        )
        code = _llm_code_for_examples(
            pairs, prompt, what="algorithmic function",
            override=llm_override, budget=budget,
        )
        if not code:
            raise GenerationError(
                "Could not generate a valid algorithmic transformation. "
                "Provide a few more representative examples."
            )
        return GeneratedTransformation(
            code=code, classification="Algorithmic",
            description=(
                "Generated by the LLM from your examples (no built-in template "
                "matched). Verified against every example."
            ),
            used_llm=True, strategy=STRATEGY_DETERMINISTIC, spec=spec,
            examples=list(pairs),
        )

    if plan.classification == "String-based":
        if plan.llm_code_required:
            if not allow_llm:
                raise GenerationError(
                    "This string transformation is not one of the built-in "
                    "deterministic rules and the LLM is unavailable. Provide "
                    "examples of a simple case/trim/substring rule, or "
                    "configure an LLM provider (OPENAI_API_KEY, a personal "
                    "key, or Ollama)."
                )
            example_text = "\n".join(
                f"Input: {s} -> Output: {t}" for s, t in pairs[:4]
            )
            prompt = (
                "Write a Python function `transform(x)` that performs this "
                f"string transformation:\n{example_text}\n\n"
                "Use only standard string methods. Return only the function "
                "in a ```python``` code block."
            )
            code = _llm_code_for_examples(
                pairs, prompt, what="string transformation",
                max_tokens=400, override=llm_override, budget=budget,
            )
            if not code:
                raise GenerationError(
                    "Could not generate a valid string transformation. "
                    "Provide more examples."
                )
            return GeneratedTransformation(
                code=code, classification="String-based",
                description=(
                    "Generated by the LLM from your examples and verified "
                    "against all of them before being accepted."
                ),
                used_llm=True, strategy=STRATEGY_DETERMINISTIC, spec=spec,
                examples=list(pairs),
            )
        det = _try_deterministic_string(pairs)
        if not det:  # pragma: no cover - planner only picks ops it can infer
            raise GenerationError(
                "Could not build the inferred string rule. Provide more examples."
            )
        return GeneratedTransformation(
            code=det, classification="String-based",
            description="Deterministic string rule inferred from the examples "
            "(no LLM used).",
            used_llm=False, strategy=STRATEGY_DETERMINISTIC, spec=spec,
            examples=list(pairs),
        )

    # ---- Exact mapping: the examples ARE the rule -------------------------- #
    code = _exact_lookup_code(pairs)
    return GeneratedTransformation(
        code=code,
        classification="General",
        description=(
            "Exact-match lookup built from your examples (the examples are the "
            "complete rule). Inputs not listed in the examples are flagged as "
            "'UNKNOWN' rather than guessed."
            + (f" Evidence: {evidence}." if evidence else "")
        ),
        used_llm=plan.used_llm,
        strategy=STRATEGY_DETERMINISTIC,
        spec=TransformationSpec(
            kind=KIND_EXACT_MAPPING,
            source_column=source_column or spec.source_column,
            output_type="string",
            exhaustive_mapping=True,
            summary="Exact mapping: the examples are the complete rule.",
            assumptions=[
                "Examples are treated as an exhaustive mapping; inputs not in "
                "the examples are flagged as unmapped rather than guessed."
            ],
            strategy=STRATEGY_DETERMINISTIC,
        ),
        examples=list(pairs),
    )
