"""Apply a validated transformation to a column of values with reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from config import settings
from sandbox import SandboxError, apply_transform, validate_transformation_code


class ApplyError(RuntimeError):
    """Raised when a transformation cannot be applied at all."""


@dataclass
class ApplyReport:
    """Result of applying a transformation to a pandas Series.

    Beyond the original (series, total, failed, errors) contract, the report
    now carries the per-row validation status and the metadata the UI needs
    to state whether a transformation actually succeeded (spec §8, §15):

    * ``review`` -- rows accepted but flagged for human review (low model
      confidence, or an ambiguous output).  These are NOT failures.
    * ``provider`` -- which model provider served the rows ("" for purely
      deterministic execution, so the UI can say "no model used").
    * ``family`` -- the transformation family (Semantic Classification, ...).
    * ``confidences`` -- per-row model confidence (None when not provided or
      deterministic).  Advisory only; never treated as calibrated.
    * ``row_status`` -- per-row status string, aligned to ``series``
      ("ok" / "review" / "failed" / "blank" / "null").
    * ``warnings`` -- validation warnings for the user (e.g. review counts).
    """

    series: "pd.Series"
    total: int
    failed: int
    # Distinct failure messages (capped) so the UI can show why rows failed.
    errors: List[str] = field(default_factory=list)
    review: int = 0
    provider: str = ""
    family: str = ""
    confidences: List[Optional[float]] = field(default_factory=list)
    row_status: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return 0.0 if self.total == 0 else (self.total - self.failed) / self.total

    @property
    def succeeded(self) -> int:
        """Rows transformed and accepted (including those flagged to review)."""
        return self.total - self.failed

    def status_counts(self) -> Dict[str, int]:
        """Tally of per-row statuses (used by reports and the UI)."""
        counts = {"ok": 0, "review": 0, "failed": 0, "blank": 0, "null": 0}
        for status in self.row_status:
            counts[status] = counts.get(status, 0) + 1
        return counts


def _is_missing(value: object) -> bool:
    """True for None / NaN / NaT (mirrors the model-backed path)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value)) and not isinstance(value, str)
    except (TypeError, ValueError):
        return False


def _is_blank(value: object) -> bool:
    """Empty or whitespace-only strings are data we must not rewrite."""
    return isinstance(value, str) and not value.strip()


def apply_to_series(
    code: str,
    series: "pd.Series",
    timeout: Optional[float] = None,
    memory_limit_mb: Optional[int] = None,
    *,
    preserve_blank: bool = True,
) -> ApplyReport:
    """Validate ``code`` and apply ``transform`` to every element of ``series``.

    With ``preserve_blank=True`` (the default, used by text / algorithmic /
    lookup rules) blank cells and missing values (``None`` / ``NaN`` / ``NaT``)
    pass through untouched, exactly as the model-backed path does: running the
    transform on them would turn an empty cell into ``'NAN'`` / ``'NONE'``
    (``str(nan)``), silently corrupting data.

    Numeric transformations opt out (``preserve_blank=False``): their generated
    code deliberately maps null-like inputs to the explicit ``"MISSING"``
    sentinel (and unparseable ones to ``nan``), which is that family's
    documented contract.  Deciding this per family is why it is a parameter
    rather than an unconditional rule.

    Raises :class:`ApplyError` if the code is rejected by the sandbox or the
    process-level limits are exceeded.  Per-row failures are reported, not
    raised, so partial success is visible to the caller.
    """
    try:
        validate_transformation_code(code)
    except SandboxError as exc:
        raise ApplyError(f"Transformation code was rejected: {exc}") from exc

    values = series.tolist()
    if not values:
        return ApplyReport(series=series.copy(), total=0, failed=0, errors=[])

    if preserve_blank:
        # Only real values go through the transform; blanks keep their exact
        # original form (empty string stays empty, NaN stays NaN).
        positions = [
            i for i, v in enumerate(values) if not (_is_missing(v) or _is_blank(v))
        ]
    else:
        positions = list(range(len(values)))
    if not positions:
        return ApplyReport(series=series.copy(), total=len(values), failed=0,
                           errors=[])

    try:
        outcome = apply_transform(
            code,
            [values[i] for i in positions],
            timeout=timeout or settings.sandbox_timeout_seconds,
            memory_limit_mb=memory_limit_mb or settings.sandbox_memory_limit_mb,
        )
    except SandboxError as exc:
        raise ApplyError(str(exc)) from exc

    new_values = list(values)
    for position, transformed in zip(positions, outcome.values):
        new_values[position] = transformed
    new_series = pd.Series(new_values, index=series.index, dtype=object)
    return ApplyReport(
        series=new_series,
        total=len(values),
        failed=outcome.failed_rows,
        errors=outcome.errors,
    )


def row_count_matches(original_len: int, transformed: "pd.Series") -> bool:
    """Ensure the transformation did not change the number of rows."""
    return len(transformed) == original_len
