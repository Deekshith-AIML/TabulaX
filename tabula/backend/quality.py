"""Automatic data profiling and transformation preflight validation.

Profiling is read-only: it never mutates the dataset.  Large frames are
profiled on a bounded random sample and clearly marked ``approximate`` so the
UI can label estimates honestly instead of silently lying.

Ordinary quality checks never call an LLM and never send data off the machine.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Frames larger than this are profiled on a random sample (marked approximate).
SAMPLE_THRESHOLD = 20_000
SAMPLE_SIZE = 20_000
HISTOGRAM_BINS = 10
TOP_CATEGORIES = 8
MAX_SAMPLES = 5

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)

Severity = str  # "info" | "warning" | "error"


def _issue(severity: Severity, message: str) -> Dict[str, str]:
    return {"severity": severity, "message": message}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def _infer_column(series: pd.Series) -> Tuple[str, Dict[str, Any]]:
    """Return (family, details) for one column.

    Families: ``empty``, ``integer``, ``number``, ``boolean``, ``datetime``,
    ``string``, ``mixed``.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty", {}

    if pd.api.types.is_bool_dtype(series):
        return "boolean", {}
    if pd.api.types.is_integer_dtype(series):
        return "integer", {}
    if pd.api.types.is_float_dtype(series):
        return "number", {}
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", {}

    # Object columns: look at the actual python types present.
    types = {type(v).__name__ for v in non_null.head(1000)}
    if types <= {"bool"}:
        return "boolean", {}
    if types <= {"int", "float", "np.int64", "np.float64"}:
        return "integer" if types <= {"int", "np.int64"} else "number", {}

    if types == {"str"} or types <= {"str"}:
        # Could still be mostly numeric strings or dates.
        sample = non_null.astype(str).head(2000)
        num_ok = pd.to_numeric(sample, errors="coerce").notna().mean()
        date_ok = pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean()
        if num_ok >= 0.9:
            return "number", {"numeric_strings": True, "numeric_parse_rate": round(float(num_ok), 4)}
        if date_ok >= 0.9:
            return "datetime", {"date_strings": True, "date_parse_rate": round(float(date_ok), 4)}
        return "string", {}

    if len(types) > 1:
        return "mixed", {"python_types": sorted(types)}
    return "string", {"python_types": sorted(types)}


def _numeric_profile(series: pd.Series, family: str) -> Optional[Dict[str, Any]]:
    if family not in ("integer", "number"):
        # Numeric-looking strings still get a profile (marked parsed).
        if family != "string":
            return None
        parsed = pd.to_numeric(series, errors="coerce")
        if parsed.notna().mean() < 0.5 or parsed.notna().sum() == 0:
            return None
        invalid = int(series.notna().sum() - parsed.notna().sum())
        base = _numeric_stats(parsed)
        if base is None:
            return None
        base["invalid_numeric_values"] = invalid
        base["parsed_from_strings"] = True
        return base

    valid = pd.to_numeric(series, errors="coerce")
    base = _numeric_stats(valid)
    if base is None:
        return None
    if family == "number":
        invalid = int(series.notna().sum() - valid.notna().sum())
        if invalid:
            base["invalid_numeric_values"] = invalid
    return base


def _numeric_stats(valid: pd.Series) -> Optional[Dict[str, Any]]:
    v = valid.dropna()
    if len(v) == 0:
        return None
    q1 = float(v.quantile(0.25))
    q3 = float(v.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = int(((v < lower) | (v > upper)).sum()) if iqr > 0 else 0
    counts, edges = np.histogram(v.to_numpy(dtype=float), bins=HISTOGRAM_BINS)
    histogram = [
        {
            "start": round(float(edges[i]), 6),
            "end": round(float(edges[i + 1]), 6),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]
    return {
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": round(float(v.mean()), 6),
        "median": round(float(v.median()), 6),
        "std": round(float(v.std()), 6) if len(v) > 1 else 0.0,
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "histogram": histogram,
        "outliers": {
            "method": "IQR (1.5x interquartile range)",
            "count": outliers,
            "lower_bound": round(float(lower), 6),
            "upper_bound": round(float(upper), 6),
            "explanation": (
                "Values below lower_bound or above upper_bound are flagged as "
                "possible outliers using Tukey's 1.5*IQR rule."
            ),
        },
    }


def _temporal_profile(series: pd.Series, family: str) -> Optional[Dict[str, Any]]:
    if family == "datetime" and pd.api.types.is_datetime64_any_dtype(series):
        return {
            "parseable_pct": 100.0,
            "invalid_count": 0,
            "mixed_formats": False,
            "min": series.min().isoformat() if series.notna().any() else None,
            "max": series.max().isoformat() if series.notna().any() else None,
        }
    if family not in ("string", "mixed", "number"):
        return None
    sample = series.dropna().astype(str).head(5000)
    if len(sample) == 0:
        return None
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    ok = int(parsed.notna().sum())
    if ok / len(sample) < 0.5:
        return None
    fmts = set()
    for s in sample[parsed.notna()].head(200):
        fmts.add(re.sub(r"\d", "9", str(s))[:20])
    return {
        "parseable_pct": _pct(ok, len(sample)),
        "invalid_count": int(len(sample) - ok),
        "mixed_formats": len(fmts) > 3,
        "detected_format_patterns": sorted(fmts)[:5],
    }


def _text_profile(series: pd.Series) -> Optional[Dict[str, Any]]:
    sample = series.dropna().astype(str).head(5000)
    if len(sample) == 0:
        return None
    ws = int((sample != sample.str.strip()).sum())
    emails = sample[sample.str.contains("@", regex=False)]
    email_like = int(len(emails))
    email_bad = int((~emails.str.match(_EMAIL_RE)).sum()) if email_like else 0
    urls = sample[sample.str.startswith(("http://", "https://"), na=False)]
    url_like = int(len(urls))
    url_bad = int((~urls.str.match(_URL_RE)).sum()) if url_like else 0
    if not any((ws, email_like, url_like)):
        return None
    return {
        "whitespace_issues": ws,
        "email_like_count": email_like,
        "email_invalid_count": email_bad,
        "url_like_count": url_like,
        "url_invalid_count": url_bad,
    }


def profile_dataframe(
    df: pd.DataFrame,
    *,
    source: Optional[str] = None,
    source_name: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute a full read-only quality profile of ``df``."""
    total_rows = int(len(df))
    working = df
    approximate = False
    if total_rows > SAMPLE_THRESHOLD:
        working = df.sample(n=min(SAMPLE_SIZE, total_rows), random_state=42)
        approximate = True

    n = int(len(working))
    overview_warnings: List[Dict[str, str]] = []

    if total_rows == 0:
        overview_warnings.append(_issue("warning", "The dataset has no rows."))

    # Exact duplicate detection on the (possibly sampled) frame; internal
    # bookkeeping columns are not real data.
    view = working.drop(columns=["_tabula_row_id"], errors="ignore")
    dup_mask = view.duplicated(keep=False)
    duplicate_rows = int(view.duplicated(keep="first").sum())

    columns_out: List[Dict[str, Any]] = []
    columns_with_missing = 0
    potential_keys: List[str] = []

    for name in view.columns:
        series = view[name]
        null_count = int(series.isna().sum())
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            blank_count = int(series.fillna("").astype(str).str.strip().eq("").sum()) - null_count
            blank_count = max(blank_count, 0)
        else:
            blank_count = 0
        missing_total = null_count + blank_count
        if missing_total:
            columns_with_missing += 1

        family, family_details = _infer_column(series)
        unique_count = int(series.nunique(dropna=True))
        constant = unique_count <= 1 and n > 1

        issues: List[Dict[str, str]] = []
        if constant and n > 1:
            issues.append(_issue("info", "Constant column (only one distinct value)."))
        if family == "mixed":
            issues.append(
                _issue("warning", "Mixed data types in this column.")
            )
        if n and _pct(missing_total, n) >= 50:
            issues.append(_issue("warning", "Half or more of the values are missing."))

        entry: Dict[str, Any] = {
            "name": str(name),
            "family": family,
            "dtype": str(series.dtype),
            **family_details,
            "null_count": null_count,
            "blank_count": blank_count,
            "missing_pct": _pct(missing_total, n),
            "unique_count": unique_count,
            "unique_pct": _pct(unique_count, n),
            "duplicate_value_count": int(n - null_count - unique_count) if n else 0,
            "is_constant": bool(constant),
            "is_mixed_type": family == "mixed",
            "samples": [
                _safe(v) for v in series.dropna().unique()[:MAX_SAMPLES].tolist()
            ],
            "issues": issues,
            "numeric": None,
            "temporal": None,
            "categorical": None,
            "text": None,
        }

        num = _numeric_profile(series, family)
        if num:
            entry["numeric"] = num
        tmp = _temporal_profile(series, family)
        if tmp:
            entry["temporal"] = tmp
        txt = _text_profile(series)
        if txt:
            entry["text"] = txt

        if family in ("string", "mixed", "empty") or (
            family in ("integer", "number") and unique_count > 1 and unique_count <= 25
        ):
            vc = series.dropna().astype(str).value_counts().head(TOP_CATEGORIES)
            if len(vc) > 0:
                total_non_null = int(series.notna().sum()) or 1
                entry["categorical"] = {
                    "top_values": [
                        {"value": str(idx), "count": int(cnt),
                         "pct": _pct(int(cnt), total_non_null)}
                        for idx, cnt in vc.items()
                    ],
                }

        # Potential key: unique and fully populated (labelled as potential).
        if n > 1 and unique_count == n and missing_total == 0:
            potential_keys.append(str(name))

        columns_out.append(entry)

    warnings = list(overview_warnings)
    if duplicate_rows:
        warnings.append(
            _issue("warning", f"{duplicate_rows} exact duplicate row(s) detected.")
        )
    if columns_with_missing:
        warnings.append(
            _issue(
                "info",
                f"{columns_with_missing} column(s) contain missing values.",
            )
        )
    if approximate:
        warnings.append(
            _issue(
                "info",
                f"Statistics computed on a random sample of {n} of {total_rows} "
                "rows (large dataset).",
            )
        )

    return {
        "generated_at": _now_iso(),
        "source": source,
        "source_name": source_name,
        "file_size_bytes": file_size_bytes,
        "total_rows": total_rows,
        "profiled_rows": n,
        "approximate": approximate,
        "overview": {
            "rows": total_rows,
            "columns": int(len(view.columns)),
            "columns_with_missing": columns_with_missing,
            "duplicate_rows": duplicate_rows,
            "warnings": warnings,
        },
        "columns": columns_out,
        "potential_keys": potential_keys,
    }


def _safe(v: Any) -> Any:
    try:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            f = float(v)
            return None if np.isnan(f) else f
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.isoformat()
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)
    except Exception:  # pragma: no cover - defensive
        return str(v)


# --------------------------------------------------------------------------- #
# Transformation preflight validation
# --------------------------------------------------------------------------- #
def validate_examples_syntax(pairs: Sequence[Tuple[Any, Any]]) -> List[str]:
    """Blocking syntax problems in the supplied example pairs."""
    errors: List[str] = []
    if not pairs:
        errors.append("At least one input -> output example is required.")
        return errors
    seen: Dict[str, str] = {}
    for i, (inp, out) in enumerate(pairs, start=1):
        key = str(inp)
        if key in seen and seen[key] != str(out):
            errors.append(
                f"Example {i} contradicts an earlier example for the same input "
                f"('{inp}' maps to both '{seen[key]}' and '{out}')."
            )
        seen[key] = str(out)
    return errors


def validate_examples_for_column(
    working_df: pd.DataFrame,
    column: str,
    pairs: Sequence[Tuple[Any, Any]],
    *,
    code: Optional[str] = None,
    apply_fn=None,
    sample_size: int = 200,
) -> Dict[str, Any]:
    """Preflight a transformation before it is applied to the whole column.

    ``apply_fn`` is ``apply_to_series``-compatible; when ``code`` is given it
    is used to check the rule against every example and a sample of real
    values.  Returns ``{errors, warnings, info, checks}`` -- ``errors`` are
    blocking, ``warnings`` are reviewable but not fatal.
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    syntax_errors = validate_examples_syntax(pairs)
    if syntax_errors:
        return {
            "ok": False,
            "errors": syntax_errors,
            "warnings": [],
            "info": [],
            "column": column,
        }

    col_exists = column in working_df.columns
    column_type = None
    has_nulls = False
    if not col_exists:
        errors.append(
            f"Column '{column}' is not in the current dataset. "
            "Refresh and select a column that exists."
        )
    else:
        series = working_df[column]
        family, _ = _infer_column(series)
        column_type = family
        has_nulls = bool(series.isna().any())
        if has_nulls:
            info.append(
                f"Column '{column}' contains null/missing values; make sure the "
                "rule handles them (or supply a null example)."
            )

    example_inputs = [p[0] for p in pairs]
    example_outputs = [p[1] for p in pairs]

    # Output type vs column type.
    out_numeric = all(_looks_numeric(o) for o in example_outputs)
    out_datetime = all(_looks_datetimeish(o) for o in example_outputs)
    if column_type in ("integer", "number") and not out_numeric:
        warnings.append(
            "The column is numeric but the example outputs look non-numeric; "
            "the transformed values will change type."
        )
    if column_type == "datetime" and not out_datetime and not out_numeric:
        warnings.append(
            "The column is datetime-like but the example outputs do not look "
            "like dates; verify the intended output format."
        )
    if not out_numeric and column_type in ("integer", "number"):
        pass  # already warned

    checks: Dict[str, Any] = {
        "examples_checked": len(pairs),
        "examples_passed": 0,
        "example_failures": [],
        "sample_checked": 0,
        "sample_failures": 0,
        "sample_failure_examples": [],
    }

    if code and apply_fn is not None and col_exists and not syntax_errors:
        # 1. The candidate rule must reproduce every supplied example.
        try:
            report = apply_fn(code, pd.Series(example_inputs, dtype=object))
            for i, (inp, expected) in enumerate(zip(example_inputs, example_outputs)):
                got = report.series.iloc[i] if i < len(report.series) else None
                if _values_match(got, expected):
                    checks["examples_passed"] += 1
                else:
                    checks["example_failures"].append(
                        {"input": _safe(inp), "expected": _safe(expected), "got": _safe(got)}
                    )
            if checks["example_failures"]:
                errors.append(
                    f"The generated rule does not reproduce "
                    f"{len(checks['example_failures'])} of {len(pairs)} example(s): "
                    "regenerate with more examples before applying."
                )
        except Exception as exc:
            errors.append(f"The generated rule failed to run on the examples: {exc}")

        # 2. Representative real values: report failure rate, do not block on
        #    partial failures (the apply step surfaces those too), but block
        #    when every sampled value fails.
        sample = working_df[column].dropna().head(sample_size)
        if len(sample) > 0:
            try:
                report = apply_fn(code, sample)
                checks["sample_checked"] = int(report.total)
                checks["sample_failures"] = int(report.failed)
                if report.failed and report.failed == report.total:
                    errors.append(
                        "The rule failed on every sampled value of this column; "
                        "it is unsafe to apply."
                    )
                elif report.failed:
                    warnings.append(
                        f"The rule failed on {report.failed} of {report.total} "
                        "sampled values; check the examples."
                    )
                if report.errors:
                    checks["sample_failure_examples"] = report.errors[:5]
            except Exception as exc:
                errors.append(f"The generated rule failed to run on the column: {exc}")

    if has_nulls:
        null_example = any(
            (o is None) or (isinstance(o, float) and np.isnan(o)) or str(o).strip() == ""
            for o in example_outputs
        )
        if not null_example and not any(_is_nullish(i) for i in example_inputs):
            warnings.append(
                "The column contains nulls but no example covers a null input; "
                "null rows may become errors or unexpected values."
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "column": column,
        "column_type": column_type,
        "column_exists": col_exists,
        "checks": checks,
    }


def _looks_numeric(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    try:
        float(str(v).strip())
        return True
    except (TypeError, ValueError):
        return False


def _looks_datetimeish(v: Any) -> bool:
    if isinstance(v, (pd.Timestamp, datetime)):
        return True
    s = str(v)
    return bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", s))


def _is_nullish(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower() in ("", "nan", "none", "null")


def _values_match(got: Any, expected: Any) -> bool:
    if _is_nullish(got) and _is_nullish(expected):
        return True
    if _looks_numeric(got) and _looks_numeric(expected):
        try:
            return abs(float(got) - float(expected)) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(got).strip() == str(expected).strip()
