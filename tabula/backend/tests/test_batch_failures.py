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
    _parse_label_map,
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


# --------------------------------------------------------------------------- #
# Corrective retry for INVENTED labels (closed set, spec §7 row-id contract)
# --------------------------------------------------------------------------- #
class TestLabelCorrectionRetry:
    def test_invented_label_triggers_focused_forced_choice_retry(self, monkeypatch):
        """A label outside the closed set triggers ONE focused forced-choice
        re-ask covering only the offending rows; the fixed answer is accepted."""
        calls: List[str] = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            llm._spend_budget(budget)
            calls.append(prompt)
            if len(calls) == 1:
                ids = _ids(prompt)
                # first pass invents a label for the second row
                outs = ["Urgent", "NotALabel"]
                return json.dumps({"results": [
                    {"row_id": rid, "output": o} for rid, o in zip(ids, outs)]})
            # focused re-ask: only the offending row, answered as an id map
            return json.dumps({"GEN-002": "Question"})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(["help", "mystery"]), _spec(["Urgent", "Question"]), [],
            llm_override=OVERRIDE,
        )
        assert report.failed == 0
        assert report.series.tolist() == ["Urgent", "Question"]
        assert len(calls) == 2
        # The retry is FOCUSED: only the offending row, the closed set, and a
        # verbatim-choice instruction -- not another full-batch round trip.
        retry = calls[1]
        assert "GEN-002" in retry
        assert "GEN-001" not in retry
        assert "mystery" in retry  # the offending row's input is re-shown
        assert '"Urgent"' in retry and '"Question"' in retry
        assert "exactly ONE label" in retry
        assert "NEVER invent a new label" in retry
        assert '"results"' not in retry

    def test_label_retry_skipped_when_budget_exhausted(self, monkeypatch):
        """No budget left for a second call: the row fails with the clear
        per-row label error instead of raising a surprise budget error."""
        calls: List[str] = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            llm._spend_budget(budget)
            calls.append(prompt)
            ids = _ids(prompt)
            return json.dumps({"results": [
                {"row_id": rid, "output": "NotALabel"} for rid in ids]})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(["mystery"]), _spec(["Urgent", "Question"]), [],
            llm_override=OVERRIDE, budget={"remaining": 1},
        )
        assert report.failed == 1
        assert report.series.tolist() == [None]
        assert any("allowed" in e for e in report.errors)
        assert len(calls) == 1  # the corrective call could not be afforded

    def test_wrong_case_label_needs_no_retry(self, monkeypatch):
        """A wrong-case label is canonicalised on the first pass -- it is not
        an offender, so no second model call is spent."""
        calls: List[str] = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            calls.append(prompt)
            ids = _ids(prompt)
            return json.dumps({"results": [
                {"row_id": rid, "output": "urgent"} for rid in ids]})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(["help"]), _spec(["Urgent"]), [],
            llm_override=OVERRIDE,
        )
        assert report.series.tolist() == ["Urgent"]
        assert report.failed == 0
        assert len(calls) == 1

    def test_unfixed_label_still_fails_after_retry(self, monkeypatch):
        """When the forced choice does not help, the row fails with the per-row
        allowed-label error -- never silently accepted."""
        calls: List[str] = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            calls.append(prompt)
            if len(calls) == 1:
                ids = _ids(prompt)
                return json.dumps({"results": [
                    {"row_id": rid, "output": "NotALabel"} for rid in ids]})
            return json.dumps({"GEN-001": "StillNotALabel"})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(["mystery"]), _spec(["Urgent", "Question"]), [],
            llm_override=OVERRIDE,
        )
        assert report.failed == 1
        assert report.series.tolist() == [None]
        assert any("allowed" in e for e in report.errors)
        assert len(calls) == 2  # the focused retry ran, and did not help


# --------------------------------------------------------------------------- #
# _parse_label_map: strict values, lenient envelope, no row shifting
# --------------------------------------------------------------------------- #
class TestParseLabelMap:
    ALLOWED = ["Account Access", "Billing", "Hardware", "Network"]

    def test_plain_id_map(self):
        fixed = _parse_label_map(
            '{"GEN-006": "Hardware", "GEN-008": "Network"}',
            self.ALLOWED, ["GEN-006", "GEN-008"],
        )
        assert fixed == {"GEN-006": "Hardware", "GEN-008": "Network"}

    def test_markdown_fenced_map(self):
        fixed = _parse_label_map(
            '```json\n{"GEN-006": "Hardware"}\n```', self.ALLOWED, ["GEN-006"]
        )
        assert fixed == {"GEN-006": "Hardware"}

    def test_wrapped_labels_map(self):
        fixed = _parse_label_map(
            '{"labels": {"GEN-006": "hardware"}}', self.ALLOWED, ["GEN-006"]
        )
        assert fixed == {"GEN-006": "Hardware"}  # canonical casing restored

    def test_invented_label_discarded(self):
        fixed = _parse_label_map(
            '{"GEN-006": "Screen Issues"}', self.ALLOWED, ["GEN-006"]
        )
        assert fixed == {}

    def test_unknown_row_id_ignored(self):
        fixed = _parse_label_map(
            '{"GEN-999": "Billing", "GEN-006": "Hardware"}',
            self.ALLOWED, ["GEN-006"],
        )
        assert fixed == {"GEN-006": "Hardware"}

    def test_non_json_returns_empty(self):
        assert _parse_label_map("Hardware", self.ALLOWED, ["GEN-006"]) == {}


# --------------------------------------------------------------------------- #
# Closed-label tasks use the forced-choice prompt as the PRIMARY call
# --------------------------------------------------------------------------- #
class TestClosedLabelPrimaryPath:
    LABELS = ["Account Access", "Billing", "Hardware", "Network"]
    EXAMPLES = [
        ("Cannot sign in with correct password", "Account Access"),
        ("Charged twice for subscription", "Billing"),
        ("Laptop randomly shuts down", "Hardware"),
        ("Wi-Fi disconnects during calls", "Network"),
    ]
    ROWS = [
        "Cannot sign in even with the correct password",
        "Subscription charged twice this month",
        "Screen flickers after I connect the monitor",
        "VPN connection keeps dropping",
    ]
    EXPECTED = ["Account Access", "Billing", "Hardware", "Network"]

    @staticmethod
    def _spec(labels):
        return TransformationSpec(
            kind=KIND_SEMANTIC, strategy=STRATEGY_MODEL, requires_model=True,
            allowed_labels=list(labels), task="semantic",
        )

    def test_prompt_is_forced_choice_with_examples_and_row_ids(self):
        from transform.model_exec import _build_prompt

        batch = [("GEN-001", "a"), ("GEN-002", "b")]
        prompt = _build_prompt(self._spec(self.LABELS), self.EXAMPLES, batch)
        assert "Choose exactly ONE label for each input" in prompt
        assert "copied verbatim from the allowed list" in prompt
        assert "NEVER invent" in prompt
        assert "CLOSEST allowed label" in prompt
        assert '"Account Access"' in prompt and '"Network"' in prompt
        assert "GEN-001" in prompt and "GEN-002" in prompt
        # still carries the row-id markers the parsers/tests rely on
        assert "Row ids (paired with the inputs below, in order):" in prompt
        assert "Inputs (JSON array, in order):" in prompt
        # not the diffuse results-array contract
        assert '{"results"' not in prompt

    def test_direct_id_map_accepted_in_one_call(self, monkeypatch):
        calls: List[str] = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            calls.append(prompt)
            return json.dumps({
                "GEN-001": "Account Access", "GEN-002": "Billing",
                "GEN-003": "Hardware", "GEN-004": "Network",
            })

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(self.ROWS), self._spec(self.LABELS), self.EXAMPLES,
            llm_override=OVERRIDE,
        )
        assert report.failed == 0
        assert report.series.tolist() == self.EXPECTED
        assert len(calls) == 1  # no repair pass needed
        assert report.row_status == ["ok"] * 4

    def test_partial_map_keeps_good_labels_and_retries_only_missing(
        self, monkeypatch,
    ):
        calls: List[str] = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            calls.append(prompt)
            if len(calls) == 1:
                # forgot GEN-003
                return json.dumps({
                    "GEN-001": "Account Access", "GEN-002": "Billing",
                    "GEN-004": "Network",
                })
            return json.dumps({"GEN-003": "Hardware"})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(self.ROWS), self._spec(self.LABELS), self.EXAMPLES,
            llm_override=OVERRIDE,
        )
        assert report.failed == 0
        assert report.series.tolist() == self.EXPECTED
        assert len(calls) == 2
        # the retry is scoped to the missing row only
        assert "GEN-003" in calls[1]
        assert "GEN-001" not in calls[1]

    def test_invented_label_in_map_triggers_focused_retry(self, monkeypatch):
        calls: List[str] = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps({
                    "GEN-001": "Account Access", "GEN-002": "Billing",
                    "GEN-003": "Screen Issues", "GEN-004": "VPN",
                })
            return json.dumps({"GEN-003": "Hardware", "GEN-004": "Network"})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(self.ROWS), self._spec(self.LABELS), self.EXAMPLES,
            llm_override=OVERRIDE,
        )
        assert report.failed == 0
        assert report.series.tolist() == self.EXPECTED
        assert len(calls) == 2
        assert "GEN-003" in calls[1] and "GEN-004" in calls[1]

    def test_map_without_allowed_value_still_fails_after_retry(self, monkeypatch):
        calls: List[str] = []

        def invoke(prompt, max_tokens=400, override=None, budget=None):
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps({
                    "GEN-001": "Account Access", "GEN-002": "Billing",
                    "GEN-003": "Hardware", "GEN-004": "Screen Issues",
                })
            # the forced choice still answers with an invented label
            return json.dumps({"GEN-004": "StillNotALabel"})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(
            pd.Series(self.ROWS), self._spec(self.LABELS), self.EXAMPLES,
            llm_override=OVERRIDE,
        )
        assert report.failed == 1
        assert report.series.tolist()[:3] == ["Account Access", "Billing", "Hardware"]
        assert report.series.tolist()[3] is None
        assert any("allowed" in e for e in report.errors)

    def test_non_label_task_keeps_results_array_prompt(self):
        from transform.model_exec import _build_prompt
        from transform.spec import KIND_GENERAL_TEXT

        spec = TransformationSpec(
            kind=KIND_GENERAL_TEXT, strategy=STRATEGY_MODEL, requires_model=True,
        )
        prompt = _build_prompt(spec, self.EXAMPLES, [("GEN-001", "a")])
        assert '{"results"' in prompt
        assert "Choose exactly ONE label" not in prompt
