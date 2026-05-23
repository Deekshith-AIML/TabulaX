"""Deterministic numerical transformation fitting.

Given a set of numeric (input, output) example pairs we fit several candidate
models and emit the best one as a ``transform`` function.  No LLM is involved.

Improvements over the original implementation:

* Candidate models are scored with a held-out-aware metric (AIC) in addition to
  MSE, which penalises unnecessary complexity and reduces overfitting on tiny
  example sets.
* Missing / non-numeric handling is explicit.
* The generated function validates its own output and never raises.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


class NumericalFitError(ValueError):
    """Raised when no model can be fitted to the provided pairs."""


def _linear(x, a, b):
    return a * x + b


def _exponential(x, a, b):
    return a * np.exp(b * x)


def _polynomial(x, a, b, c):
    return a * x**2 + b * x + c


def _rational(x, a, b, c):
    return (a * x + b) / (x + c)


def _logarithmic(x, a, b):
    # Guard against non-positive x inside the fit by masking.
    return a * np.log(np.clip(x, 1e-12, None)) + b


def _parse_pairs(pairs) -> Tuple[np.ndarray, np.ndarray]:
    """Parse examples into numeric arrays, rejecting non-numeric values.

    Invalid inputs/outputs are reported explicitly (``NumericalFitError``)
    instead of being silently dropped, so the caller sees why a fit failed.
    """
    xs: List[float] = []
    ys: List[float] = []
    bad: List[Tuple[str, str]] = []
    for src, tgt in pairs:
        try:
            xs.append(float(src))
            ys.append(float(tgt))
        except (TypeError, ValueError):
            bad.append((str(src), str(tgt)))
    if bad:
        shown = ", ".join(f"({s!r} -> {t!r})" for s, t in bad[:3])
        more = " ..." if len(bad) > 3 else ""
        raise NumericalFitError(
            f"Non-numeric example(s) in a numerical transformation: {shown}{more}. "
            "Numerical examples must be numbers on both sides."
        )
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _aic(n: int, mse: float, k: int) -> float:
    """Akaike information criterion for a model with ``k`` params.

    Lower is better.  Guards against log(0).
    """
    mse = max(mse, 1e-12)
    return n * np.log(mse) + 2 * k


def _is_exact(pred, y) -> bool:
    """True when the model reproduces every example to within tight tolerance."""
    return bool(
        np.all(np.isfinite(pred))
        and np.allclose(pred, y, rtol=1e-6, atol=1e-6)
    )


# Simpler families win ties between equally-exact candidates: a linear rule is
# easier to reason about (and less likely to overfit tiny example sets) than an
# exponential/logarithmic/polynomial/rational curve.
_SIMPLICITY_ORDER = {
    "linear": 0,
    "exponential": 1,
    "logarithmic": 2,
    "polynomial": 3,
    "rational": 4,
}


def fit_best_model(pairs) -> Dict[str, object]:
    """Fit candidate models and return the best one.

    Selection policy (in order):

    1. Candidates that reproduce **every** example (within tolerance) win over
       inexact ones — the rule must explain all supplied data.
    2. Among exact candidates the simplest rule wins: fewest parameters, then a
       fixed family order (linear before exponential/logarithmic, and no
       polynomial/rational fits when a simpler exact rule exists).
    3. With no exact candidate, the AIC-best fit is used only if it is close
       (relative MSE within 1% of the target variance); otherwise
       :class:`NumericalFitError` reports that the examples do not support a
       reliable rule.

    Returns a dict with keys ``name``, ``params``, ``mse``, ``aic``,
    ``exact`` and ``predict`` (a callable).
    """
    x, y = _parse_pairs(pairs)
    n = len(x)
    if n < 2:
        raise NumericalFitError(
            "At least 2 numeric input/output pairs are required for a numerical fit."
        )

    candidates = [
        ("linear", _linear, 2),
        ("exponential", _exponential, 2),
        ("logarithmic", _logarithmic, 2),
    ]
    if n >= 3:
        candidates.append(("polynomial", _polynomial, 3))
        candidates.append(("rational", _rational, 3))

    from scipy.optimize import OptimizeWarning, curve_fit

    scored: List[Dict[str, object]] = []
    for name, func, k in candidates:
        # Skip models that are mathematically invalid for the data domain.
        if name == "logarithmic" and np.any(x <= 0):
            continue
        try:
            with warnings.catch_warnings():
                # Exactly-determined fits (e.g. 2 points, 2 params) cannot
                # estimate a covariance; that is fine for our use.
                warnings.simplefilter("ignore", OptimizeWarning)
                with np.errstate(all="ignore"):
                    params, _ = curve_fit(func, x, y, method="lm", maxfev=10000)
                    pred = func(x, *params)
                    if not np.all(np.isfinite(pred)):
                        continue
                    mse = float(np.mean((y - pred) ** 2))
            scored.append(
                {
                    "name": name,
                    "params": [float(p) for p in params],
                    "mse": mse,
                    "aic": float(_aic(n, mse, k)),
                    "k": k,
                    "exact": _is_exact(pred, y),
                    "predict": lambda v, f=func, p=params: f(v, *p),
                }
            )
        except Exception:
            continue

    if not scored:
        raise NumericalFitError("No numerical model could be fitted to the data.")

    exact = [d for d in scored if d["exact"]]
    if exact:
        # Every exact candidate explains all the data; prefer the simplest.
        exact.sort(key=lambda d: (d["k"], _SIMPLICITY_ORDER.get(str(d["name"]), 99)))
        return exact[0]

    # No exact rule: accept only a defensible approximate fit, else report the
    # ambiguity instead of silently guessing.
    scored.sort(key=lambda d: (d["aic"], d["mse"]))
    best = scored[0]
    var_y = float(np.var(y))
    limit = max(1e-2 * var_y, 1e-12) if var_y > 0 else 1e-12
    if float(best["mse"]) > limit:
        raise NumericalFitError(
            f"No candidate model reproduces these examples closely enough "
            f"(best was {best['name']} with MSE={best['mse']:.4g}). "
            "Provide more representative examples or check that the outputs "
            "follow a consistent rule."
        )
    return best


def _format_params(params: List[float]) -> str:
    return ", ".join(f"{p:.6f}" for p in params)


def _fmt(p: float) -> str:
    """Format a parameter for code generation, snapping noise to zero."""
    if abs(p) < 5e-10:
        return "0.000000"
    return f"{p:.6f}"


def generate_numeric_code(pairs) -> str:
    """Return a ``transform`` function (as source) for the best numeric fit.

    The chosen model has been validated against every example (see
    :func:`fit_best_model`).  The generated function strips non-numeric
    characters, returns the literal string ``"MISSING"`` for null-like inputs
    and ``nan`` for unparseable ones.  When every example target is a whole
    number the result is rounded to an integer so the output matches the
    examples' apparent type.
    """
    best = fit_best_model(pairs)
    name = best["name"]
    a, b, *rest = best["params"]
    c = rest[0] if rest else None

    if name == "linear":
        expr = f"{_fmt(a)} * x_num + {_fmt(b)}"
    elif name == "exponential":
        expr = f"{_fmt(a)} * np.exp({_fmt(b)} * x_num)"
    elif name == "logarithmic":
        expr = f"{_fmt(a)} * np.log(max(x_num, 1e-12)) + {_fmt(b)}"
    elif name == "polynomial":
        expr = f"{_fmt(a)} * x_num**2 + {_fmt(b)} * x_num + {_fmt(c)}"
    elif name == "rational":
        expr = f"({_fmt(a)} * x_num + {_fmt(b)}) / (x_num + {_fmt(c)})"
    else:  # pragma: no cover - defensive
        expr = f"{_fmt(a)} * x_num + {_fmt(b)}"

    # Decide whether to coerce results to int (all example targets whole numbers).
    try:
        _, ys = _parse_pairs(pairs)
        as_int = bool(len(ys) and np.all(np.isclose(ys, np.round(ys))))
    except Exception:
        as_int = False

    if as_int:
        finish = "        val = float(" + expr + ")\n        return int(round(val)) if np.isfinite(val) else np.nan"
    else:
        finish = "        return float(" + expr + ")"

    return f'''import re
import numpy as np
import pandas as pd

def transform(x):
    if x is None:
        return "MISSING"
    try:
        if pd.isna(x):
            return "MISSING"
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in ("", "n/a", "na", "null", "nan", "none"):
        return "MISSING"
    try:
        x_str = re.sub(r"[^0-9eE.\\-+]", "", str(x)).strip()
        x_num = float(x_str)
    except (ValueError, TypeError):
        return np.nan
    try:
{finish}
    except Exception:
        return np.nan
'''


def describe_fit(pairs) -> str:
    """Human-readable evidence for the chosen model (not a calibrated probability)."""
    try:
        best = fit_best_model(pairs)
        x, y = _parse_pairs(pairs)
    except NumericalFitError as exc:
        return str(exc)
    pred = best["predict"](x)
    if _is_exact(pred, y):
        fit_note = (
            "it reproduces every example exactly; equally-exact candidates "
            "were ranked by simplicity (fewer parameters and simpler model "
            "family first)"
        )
    else:
        fit_note = (
            "it is the best-scoring candidate by AIC (approximate fit - no "
            "candidate reproduced every example exactly)"
        )
    return (
        f"Selected a {best['name']} model (MSE={best['mse']:.4g}) fitted to "
        f"{len(x)} numeric example pairs; {fit_note}. This is evidence for "
        "the rule, not a calibrated confidence score."
    )
