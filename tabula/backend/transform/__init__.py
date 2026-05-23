"""TabulaX transformation engine.

Public surface re-exported for convenience:

* :func:`classify`      -- classify example pairs into one of four families
* :func:`generate`      -- generate a validated transformation (code or spec)
* :func:`apply_to_series` -- apply a validated transform to a pandas Series
* :func:`plan_examples` -- choose the execution strategy for example pairs
* :func:`apply_model_backed` -- execute a model-backed transformation
"""

from transform.apply import ApplyError, ApplyReport, apply_to_series
from transform.classify import (
    ClassificationResult,
    VALID_CLASSES,
    classify,
    parse_examples,
    parse_examples_payload,
)
from transform.generate import (
    GeneratedTransformation,
    GenerationError,
    generate,
    validate_code_against_examples,
)
from transform.model_exec import (
    DEFAULT_BATCH_SIZE,
    ModelExecutionError,
    apply_model_backed,
)
from transform.planner import Plan, PlanError, looks_like_closed_mapping, plan_examples
from transform.spec import (
    STRATEGY_DETERMINISTIC,
    STRATEGY_MODEL,
    STRATEGY_NONE,
    TransformationSpec,
)
from transform.templates import TEMPLATE_DESCRIPTIONS, TEMPLATES, list_template_types

__all__ = [
    "ApplyError",
    "ApplyReport",
    "apply_to_series",
    "ClassificationResult",
    "VALID_CLASSES",
    "classify",
    "parse_examples",
    "parse_examples_payload",
    "GeneratedTransformation",
    "GenerationError",
    "generate",
    "validate_code_against_examples",
    "DEFAULT_BATCH_SIZE",
    "ModelExecutionError",
    "apply_model_backed",
    "Plan",
    "PlanError",
    "looks_like_closed_mapping",
    "plan_examples",
    "STRATEGY_DETERMINISTIC",
    "STRATEGY_MODEL",
    "STRATEGY_NONE",
    "TransformationSpec",
    "TEMPLATE_DESCRIPTIONS",
    "TEMPLATES",
    "list_template_types",
]
