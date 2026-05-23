"""Tests for classification, generation, numerical fitting and templates.

All tests run with the LLM disabled (see conftest) so results are deterministic
and no API key is required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transform import (
    GenerationError,
    apply_to_series,
    classify,
    generate,
    list_template_types,
    parse_examples,
)
from transform.numerical import NumericalFitError, fit_best_model, generate_numeric_code


# --------------------------------------------------------------------------- #
# Example parsing
# --------------------------------------------------------------------------- #
def test_parse_examples_arrow():
    assert parse_examples("a -> b\nc -> d") == [("a", "b"), ("c", "d")]


def test_parse_examples_comma():
    assert parse_examples("a,b\nc,d") == [("a", "b"), ("c", "d")]


def test_parse_examples_rejects_malformed():
    with pytest.raises(ValueError):
        parse_examples("no separator here")
    with pytest.raises(ValueError):
        parse_examples("   \n  ")


def test_parse_examples_trims_whitespace():
    assert parse_examples("  apple   ->   APPLE  ") == [("apple", "APPLE")]


def test_parse_examples_unicode_arrow():
    from transform import parse_examples as pe

    assert pe("7 \u2192 70\n8 \u2192 80") == [("7", "70"), ("8", "80")]


def test_parse_examples_mixed_arrows_split_at_first():
    from transform import parse_examples as pe

    assert pe("a -> b \u2192 c") == [("a", "b \u2192 c")]
    assert pe("a \u2192 b -> c") == [("a", "b -> c")]


def test_parse_examples_preserves_commas_spaces_unicode():
    from transform import parse_examples as pe

    assert pe("New York, NY \u2192 NY") == [("New York, NY", "NY")]
    assert pe("caf\u00e9 au lait -> Caf\u00c9 AU LAIT") == [
        ("caf\u00e9 au lait", "Caf\u00c9 AU LAIT")
    ]


def test_parse_examples_reports_line_numbers():
    from transform import parse_examples as pe

    with pytest.raises(ValueError) as exc:
        pe("a -> b\nnot a valid line\nc -> d")
    msg = str(exc.value)
    assert "Line 2" in msg
    assert "Line 1" not in msg  # only the malformed line is reported


def test_parse_examples_missing_input_or_output_has_line_number():
    from transform import parse_examples as pe

    with pytest.raises(ValueError) as exc:
        pe("a -> b\n   \nc ->")
    assert "Line 3" in str(exc.value)
    assert "missing output" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        pe("-> out")
    assert "Line 1" in str(exc.value)
    assert "missing input" in str(exc.value)


def test_parse_examples_contradictory_duplicates_rejected():
    from transform import parse_examples as pe

    with pytest.raises(ValueError) as exc:
        pe("x -> 1\ny -> 2\nx -> 3")
    msg = str(exc.value)
    assert "contradictory" in msg
    assert "Line 1" in msg and "Line 3" in msg


def test_parse_examples_identical_duplicates_deduped():
    from transform import parse_examples as pe

    assert pe("x -> 1\ny -> 2\nx -> 1") == [("x", "1"), ("y", "2")]


def test_parse_examples_payload_uses_example_labels():
    from transform import parse_examples_payload

    assert parse_examples_payload([("a", "b"), ("c", "d")]) == [("a", "b"), ("c", "d")]
    with pytest.raises(ValueError) as exc:
        parse_examples_payload([("a", "b"), ("a", "c")])
    assert "Example 2" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        parse_examples_payload([("a", " ")])
    assert "Example 1" in str(exc.value)
    assert "missing output" in str(exc.value)


# --------------------------------------------------------------------------- #
# Classification (deterministic, LLM off)
# --------------------------------------------------------------------------- #
def test_classify_uppercase_is_string():
    r = classify(parse_examples("apple -> APPLE\nbanana -> BANANA"), allow_llm=False)
    assert r.classification == "String-based"
    assert r.used_llm is False


def test_classify_linear_numeric():
    r = classify(parse_examples("1 -> 3\n2 -> 5\n3 -> 7"), allow_llm=False)
    assert r.classification == "Numerical"
    assert r.used_llm is False


def test_classify_template_match_is_algorithmic():
    r = classify(parse_examples("10 -> 1010\n5 -> 101"), allow_llm=False)
    assert r.classification == "Algorithmic"
    assert r.template == "decimal_to_binary"
    assert r.used_llm is False


def test_classify_dates_is_algorithmic():
    r = classify(parse_examples("2023-04-12 -> Wednesday\n2023-04-13 -> Thursday"), allow_llm=False)
    assert r.classification == "Algorithmic"
    assert r.template == "date_to_weekday"


def test_classify_lookup_defaults_general_without_llm():
    r = classify(parse_examples("Microsoft -> Satya Nadella\nApple -> Tim Cook"), allow_llm=False)
    assert r.classification == "General"


def test_classify_returns_evidence_not_confidence():
    r = classify(parse_examples("apple -> APPLE\nbanana -> BANANA"), allow_llm=False)
    assert r.evidence, "classification must provide evidence"
    assert all(isinstance(e, str) for e in r.evidence)


def test_classify_unknown_pattern_falls_back_to_general_off_llm():
    r = classify(parse_examples("ABC123 -> XYZ789\nDEF456 -> UVW012"), allow_llm=False)
    assert r.classification == "General"
    assert r.used_llm is False


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generate_string_deterministic_no_llm():
    pairs = parse_examples("apple -> APPLE\nbanana -> BANANA")
    g = generate(pairs, "String-based", allow_llm=False)
    assert g.used_llm is False
    out = apply_to_series(g.code, pd.Series(["apple", "banana"]))
    assert out.series.tolist() == ["APPLE", "BANANA"]


def test_generate_template_algorithmic_no_llm():
    pairs = parse_examples("10 -> 1010\n5 -> 101")
    g = generate(pairs, "Algorithmic", template="decimal_to_binary", allow_llm=False)
    assert g.used_llm is False
    assert "bin(" in g.code
    out = apply_to_series(g.code, pd.Series(["10", "5"]))
    assert out.series.tolist() == ["1010", "101"]


def test_generate_general_exact_lookup_no_llm():
    pairs = parse_examples("France -> Paris\nJapan -> Tokyo")
    g = generate(pairs, "General", allow_llm=False)
    assert g.used_llm is False
    out = apply_to_series(g.code, pd.Series(["France", "Canada"]))
    assert out.series.tolist() == ["Paris", "UNKNOWN"]


def test_generate_requires_llm_for_open_string_without_key():
    # An open-ended pattern with no deterministic rule and no LLM -> clear error.
    pairs = parse_examples("foo -> ZzfooZz\nbar -> ZzbarZz")
    with pytest.raises(GenerationError) as exc:
        generate(pairs, "String-based", allow_llm=False)
    assert "OPENAI_API_KEY" in str(exc.value) or "LLM" in str(exc.value)


def test_generate_numerical_never_needs_llm():
    pairs = parse_examples("1 -> 3\n2 -> 5\n3 -> 7")
    g = generate(pairs, "Numerical", allow_llm=False)
    assert g.used_llm is False
    out = apply_to_series(g.code, pd.Series([4, 5]))
    # 4 -> 9, 5 -> 11
    assert int(out.series.iloc[0]) == 9
    assert int(out.series.iloc[1]) == 11


# --------------------------------------------------------------------------- #
# Numerical fitting
# --------------------------------------------------------------------------- #
def test_fit_linear_exact():
    best = fit_best_model([("1", "3"), ("2", "5"), ("3", "7")])
    assert best["name"] == "linear"
    assert best["mse"] < 1e-6


def test_fit_requires_two_points():
    with pytest.raises(NumericalFitError):
        fit_best_model([("1", "3")])


def test_fit_non_numeric_raises_or_skips():
    with pytest.raises(NumericalFitError):
        fit_best_model([("abc", "def"), ("ghi", "jkl")])


def test_numeric_code_handles_missing_and_non_numeric():
    code = generate_numeric_code([("1", "3"), ("2", "5"), ("3", "7")])
    out = apply_to_series(code, pd.Series([None, "abc", "4", "n/a"]))
    vals = out.series.tolist()
    assert vals[0] == "MISSING"
    assert vals[3] == "MISSING"
    # examples 1->3, 2->5, 3->7 fit y=2x+1, so 4 -> 9
    assert int(vals[2]) == 9


def test_numeric_code_integer_output_when_targets_whole():
    code = generate_numeric_code([("1", "3"), ("2", "5")])
    out = apply_to_series(code, pd.Series(["3"]))
    # Ensure it's an int, not a float string
    assert int(out.series.iloc[0]) == 7
    assert "int(" in code or "int(round" in code


def test_multiplication_by_ten_prefers_simplest_rule():
    """7->70, 8->80 must infer x10 (not an exponential curve through the points)."""
    from transform import parse_examples as pe

    pairs = pe("7 -> 70\n8 -> 80")
    g = generate(pairs, "Numerical", allow_llm=False)
    assert g.used_llm is False
    assert "np.exp" not in g.code and "np.log" not in g.code
    assert "linear" in g.description
    out = apply_to_series(g.code, pd.Series(["6", "5", "7", "8"]))
    assert [int(v) for v in out.series.tolist()] == [60, 50, 70, 80]


def test_exact_exponential_wins_when_linear_not_exact():
    """1->2, 2->4, 3->8: exponential explains every example, linear does not."""
    pairs = [("1", "2"), ("2", "4"), ("3", "8")]
    best = fit_best_model(pairs)
    assert best["name"] == "exponential"
    assert best["exact"] is True
    code = generate_numeric_code(pairs)
    out = apply_to_series(code, pd.Series(["4"]))
    assert int(out.series.iloc[0]) == 16


def test_n_polynomial_avoided_when_simpler_exact_rule_exists():
    # Data lies on a line, so a quadratic would also fit exactly - linear must win.
    best = fit_best_model([("1", "3"), ("2", "5"), ("3", "7"), ("4", "9")])
    assert best["name"] == "linear"
    assert best["exact"] is True


def test_non_numeric_examples_rejected_explicitly():
    with pytest.raises(NumericalFitError) as exc:
        generate_numeric_code([("1", "3"), ("abc", "def")])
    assert "Non-numeric" in str(exc.value)


def test_unsupportable_examples_report_ambiguity_instead_of_guessing():
    # Four alternating points: no simple candidate reproduces them all, and
    # quadratic/rational fits miss badly - the error must say so.
    pairs = [("1", "10"), ("2", "1000"), ("3", "10"), ("4", "1000"), ("5", "10")]
    with pytest.raises(NumericalFitError) as exc:
        fit_best_model(pairs)
    assert "closely enough" in str(exc.value)


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list_template_types())
def test_every_template_validates_in_sandbox(key):
    from sandbox import validate_transformation_code
    from transform import templates

    validate_transformation_code(templates.get_template(key))


def test_template_descriptions_exist_for_all():
    from transform import templates

    assert set(templates.TEMPLATES) == set(templates.TEMPLATE_DESCRIPTIONS)


# --------------------------------------------------------------------------- #
# Applying: row counts, failures, schema
# --------------------------------------------------------------------------- #
def test_apply_preserves_row_count_and_index():
    code = "def transform(x):\n    return str(x) + '!'"
    s = pd.Series(["a", "b", "c"], index=[10, 11, 12])
    rep = apply_to_series(code, s)
    assert len(rep.series) == 3
    assert rep.series.index.tolist() == [10, 11, 12]


def test_apply_reports_partial_failure_rate():
    code = "def transform(x):\n    return 10 / float(str(x))"
    rep = apply_to_series(code, pd.Series(["2", "0", "5"]))
    assert rep.total == 3
    assert rep.failed == 1
    assert 0 < rep.success_rate < 1


def test_apply_rejects_malicious_code():
    from transform.apply import ApplyError

    with pytest.raises(ApplyError):
        apply_to_series("import os\ndef transform(x):\n    return os.getcwd()", pd.Series(["a"]))


def test_apply_empty_series():
    rep = apply_to_series("def transform(x):\n    return x", pd.Series([], dtype=object))
    assert rep.total == 0
    assert rep.failed == 0
