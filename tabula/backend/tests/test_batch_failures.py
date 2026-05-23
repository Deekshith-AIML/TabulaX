"""Batched model execution -- failure and contract tests (MOCKED).

Proves the row-id batch contract is enforced so a malformed, incomplete or
mis-counted model response can never silently shift or corrupt rows, and that
provider failures are reported per-row rather than crashing or lying.

No real provider is contacted: ``llm.invoke`` is always monkeypatched.
"""

from __future__ import annotations

import json
import re
from typing import List

import pandas as pd
import pytest

import llm
from transform import apply_model_backed
from transform.model_exec import (
    _coerce_confidence,
    _parse_results,
    _row_id,
)
from transform.spec import (
    KIND_SEMANTIC,
    STRATEGY_MODEL,
    TransformationSpec,
)

OVERRIDE = {"api_key": "test-personal-key"}


def _spec(labels=None, kind=KIND_SEMANTIC):
    return TransformationSpec(
        kind=kind, strategy=STRATEGY_MODEL, requires_model=True,
        allowed_labels=list(labels or []),
    )


def _ids(prompt: str) -> List[str]:
    m = re.search(r"Row ids \(paired with the inputs below, in order\):\n(\[[^\]]*\])",
                  prompt)
    return json.loads(m.group(1))


def _inputs(prompt: str) -> List[str]:
    m = re.search(r"Inputs \(JSON array, in order\):\n(\[.*\])", prompt, re.DOTALL)
    return json.loads(m.group(1))


# --------------------------------------------------------------------------- #
# Row-id contract
# --------------------------------------------------------------------------- #
class TestRowIdContract:
    def test_row_ids_are_stable_and_position_based(self):
        assert _row_id(0) == "GEN-001"
        assert _row_id(7) == "GEN-008"
        assert _row_id(999) == "GEN-1000"

    def test_reordered_response_matched_by_id_not_position(self, monkeypatch):
        """A model that returns results out of order must still land right."""
        def invoke(prompt, max_tokens=400, override=None, budget=None):
            ids = _ids(prompt)
            inputs = _inputs(prompt)
            pairs = list(zip(ids, inputs))
            pairs.reverse()  # deliberately reorder
            results = [{"row_id": rid, "output": v.upper()} for rid, v in pairs]
            return json.dumps({"results": results})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(["alpha", "beta", "gamma"]), _spec(), [],
            llm_override=OVERRIDE,
        )
        assert report.series.tolist() == ["ALPHA", "BETA", "GAMMA"]
        assert report.failed == 0

    def test_missing_row_id_rejects_batch(self, monkeypatch):
        def invoke(prompt, max_tokens=400, override=None, budget=None):
            ids = _ids(prompt)
            # omit the last id
            results = [{"row_id": rid, "output": "X"} for rid in ids[:-1]]
            return json.dumps({"results": results})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(pd.Series(["a", "b", "c"]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 3
        assert report.series.tolist() == [None, None, None]
        assert any("omitted" in e for e in report.errors)

    def test_duplicate_row_id_rejects_batch(self, monkeypatch):
        def invoke(prompt, max_tokens=400, override=None, budget=None):
            ids = _ids(prompt)
            results = [{"row_id": ids[0], "output": "X"},
                       {"row_id": ids[0], "output": "Y"},
                       {"row_id": ids[2], "output": "Z"}]
            return json.dumps({"results": results})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(pd.Series(["a", "b", "c"]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 3
        assert any("more than once" in e for e in report.errors)

    def test_unknown_row_id_rejects_batch(self, monkeypatch):
        def invoke(prompt, max_tokens=400, override=None, budget=None):
            ids = _ids(prompt)
            results = [{"row_id": rid, "output": "X"} for rid in ids]
            results[0]["row_id"] = "GHOST-999"
            return json.dumps({"results": results})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(pd.Series(["a", "b"]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 2
        assert any("unknown row_id" in e for e in report.errors)


# --------------------------------------------------------------------------- #
# Malformed / wrong-count responses
# --------------------------------------------------------------------------- #
class TestMalformedResponses:
    def test_malformed_json_twice_fails_batch_cleanly(self, monkeypatch):
        monkeypatch.setattr("llm.invoke", lambda *a, **k: "sorry, not json")
        report = apply_model_backed(pd.Series(["a", "b"]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 2
        assert report.series.tolist() == [None, None]
        assert any("unusable response" in e or "not valid JSON" in e
                   for e in report.errors)

    def test_wrong_count_positional_rejected(self, monkeypatch):
        monkeypatch.setattr("llm.invoke", lambda *a, **k: '["only-one"]')
        report = apply_model_backed(pd.Series(["a", "b"]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 2
        assert any("expected 2 outputs" in e for e in report.errors)

    def test_one_bad_batch_does_not_corrupt_the_good_batch(self, monkeypatch):
        calls = {"n": 0}

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            calls["n"] += 1
            ids = _ids(prompt)
            if calls["n"] == 1:  # first batch good
                return json.dumps({"results": [
                    {"row_id": rid, "output": "OK"} for rid in ids]})
            return "garbage"      # second batch bad

        monkeypatch.setattr("llm.invoke", invoke)
        # batch_size=2 over 4 rows -> batches [0,1] good, [2,3] bad
        report = apply_model_backed(pd.Series(["a", "b", "c", "d"]), _spec(), [],
                                    llm_override=OVERRIDE, batch_size=2)
        assert report.series.tolist()[:2] == ["OK", "OK"]
        assert report.series.tolist()[2:] == [None, None]
        assert report.failed == 2
        statuses = report.row_status
        assert statuses[:2] == ["ok", "ok"]
        assert statuses[2:] == ["failed", "failed"]

    def test_null_output_for_nonempty_input_is_failure(self, monkeypatch):
        def invoke(prompt, max_tokens=400, override=None, budget=None):
            ids = _ids(prompt)
            return json.dumps({"results": [
                {"row_id": rid, "output": None} for rid in ids]})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(pd.Series(["a", "b"]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 2
        assert any("null" in e for e in report.errors)


# --------------------------------------------------------------------------- #
# Null / blank pass-through (never sent to the model)
# --------------------------------------------------------------------------- #
class TestNullAndBlank:
    def test_nulls_and_blanks_pass_through_untouched(self, monkeypatch):
        seen = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            ids = _ids(prompt)
            seen.extend(_inputs(prompt))
            return json.dumps({"results": [
                {"row_id": rid, "output": "OK"} for rid in ids]})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series([None, "", "   ", "real"]), _spec(), [], llm_override=OVERRIDE,
        )
        assert report.series.tolist()[0] is None
        assert report.series.tolist()[1] == ""
        assert report.series.tolist()[2] == "   "
        assert report.series.tolist()[3] == "OK"
        assert report.failed == 0
        assert seen == ["real"]           # blanks never reach the model
        assert report.row_status[:3] == ["null", "blank", "blank"]

    def test_all_nulls_makes_no_model_call(self, monkeypatch):
        def invoke(*a, **k):
            raise AssertionError("should not be called for all-null input")

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(pd.Series([None, None]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 0
        assert report.total == 2
        assert report.series.tolist() == [None, None]


# --------------------------------------------------------------------------- #
# Provider failures surfaced per-row
# --------------------------------------------------------------------------- #
class TestProviderFailures:
    def test_provider_exception_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(
            "llm.invoke", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        report = apply_model_backed(pd.Series(["a", "b"]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 2
        assert any("request failed" in e for e in report.errors)

    def test_unavailable_provider_stops_remaining_rows(self, monkeypatch):
        monkeypatch.setattr(
            "llm.invoke",
            lambda *a, **k: (_ for _ in ()).throw(
                llm.LLMUnavailableError("no provider configured")
            ),
        )
        report = apply_model_backed(pd.Series(["a", "b", "c"]), _spec(), [],
                                    llm_override=OVERRIDE)
        assert report.failed == 3
        assert report.row_status == ["failed", "failed", "failed"]
        assert any("provider" in e.lower() for e in report.errors)

    def test_budget_exhaustion_marks_remaining_failed(self, monkeypatch):
        def invoke(prompt, max_tokens=400, override=None, budget=None):
            llm._spend_budget(budget)  # real accounting; will raise when empty
            ids = _ids(prompt)
            return json.dumps({"results": [
                {"row_id": rid, "output": "OK"} for rid in ids]})

        monkeypatch.setattr("llm.invoke", invoke)
        # Force a tiny budget: first batch succeeds, second hits the limit.
        budget = {"remaining": 1}
        report = apply_model_backed(
            pd.Series(["a", "b", "c", "d"]), _spec(), [],
            llm_override=OVERRIDE, budget=budget, batch_size=2,
        )
        assert report.series.tolist()[:2] == ["OK", "OK"]
        assert report.failed == 2
        assert any("limit" in e.lower() for e in report.errors)


# --------------------------------------------------------------------------- #
# Confidence coercion helpers
# --------------------------------------------------------------------------- #
class TestConfidenceHelper:
    @pytest.mark.parametrize("raw,expected", [
        (0.5, 0.5), (1, 1.0), (0, 0.0), (50, 0.5), (85, 0.85), (100, 1.0),
        (None, None), ("bad", None), (float("nan"), None), (-3, 0.0), (250, 1.0),
    ])
    def test_coerce(self, raw, expected):
        got = _coerce_confidence(raw)
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# _parse_results directly (unit)
# --------------------------------------------------------------------------- #
class TestParseResults:
    def test_rowid_shape_aligned_to_batch_order(self):
        text = json.dumps({"results": [
            {"row_id": "GEN-002", "output": "b"},
            {"row_id": "GEN-001", "output": "a"},
        ]})
        outputs, confs, err = _parse_results(text, ["GEN-001", "GEN-002"])
        assert err == ""
        assert outputs == ["a", "b"]
        assert confs == [None, None]

    def test_confidence_extracted(self):
        text = json.dumps({"results": [
            {"row_id": "GEN-001", "output": "a", "confidence": 0.7},
        ]})
        outputs, confs, err = _parse_results(text, ["GEN-001"])
        assert err == ""
        assert confs == [pytest.approx(0.7)]

    def test_results_missing_output_key(self):
        text = json.dumps({"results": [{"row_id": "GEN-001"}]})
        outputs, _, err = _parse_results(text, ["GEN-001"])
        assert outputs is None
        assert "row_id" in err

    def test_legacy_outputs_shape_still_accepted(self):
        outputs, confs, err = _parse_results('{"outputs": ["a", "b"]}',
                                             ["GEN-001", "GEN-002"])
        assert err == ""
        assert outputs == ["a", "b"]

    def test_legacy_index_mapped_accepted(self):
        text = json.dumps([
            {"i": 1, "value": "b"}, {"i": 0, "value": "a"},
        ])
        outputs, _, err = _parse_results(text, ["GEN-001", "GEN-002"])
        assert err == ""
        assert outputs == ["a", "b"]

    def test_not_json(self):
        outputs, _, err = _parse_results("hello world", ["GEN-001"])
        assert outputs is None
        assert "JSON" in err
