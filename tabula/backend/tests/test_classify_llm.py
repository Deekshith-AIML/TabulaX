"""LLM classify-response handling: tolerant parsing + one corrective retry.

The model (mistral via Ollama) answers the classify prompt with VALID JSON
but sometimes puts the TASK name in the "class" field:

    {"class": "semantic", "task": "exact_mapping", "labels": [...]}

That answer used to be discarded wholesale -> "the LLM did not return a
recognized class name" -> both /classify/ and the generate flow failed for
perfectly good model output.  These tests pin the recovery behaviour and the
single corrective retry, with NO real model call.
"""

from __future__ import annotations

import json

import llm
from transform.classify import _llm_infer, _parse_classify_json

PAIRS = [
    ("Cannot sign in even with the correct password", "Account Access"),
    ("Subscription charged twice this month", "Billing"),
]

# Personal-key override: tests run with the LLM disabled (conftest), and this
# is how the existing engine tests pass the availability gate in _llm_infer.
OVERRIDE = {"api_key": "test-personal-key"}

GOOD = {
    "class": "General",
    "task": "semantic",
    "labels": ["Account Access", "Billing"],
    "reason": "closed category labels decided by meaning",
}

# The exact response mistral gave for the user's 2-example input (captured
# live by diag_classify_raw.py): valid JSON, task name in the class field.
MISTRAL_STYLE = {
    "class": "semantic",
    "task": "exact_mapping",
    "labels": ["Account Access", "Billing"],
    "reason": "specific issues map to the respective categories.",
}


# --------------------------------------------------------------------------- #
# _parse_classify_json: recover usable answers, reject unusable ones
# --------------------------------------------------------------------------- #
class TestParseClassifyJson:
    def test_task_name_in_class_field_is_recovered(self):
        family, task, labels = _parse_classify_json(json.dumps(MISTRAL_STYLE))
        assert family == "General"
        assert task == "exact_mapping"
        assert labels == ["Account Access", "Billing"]

    def test_task_name_in_class_field_fills_missing_task(self):
        raw = json.dumps({"class": "semantic", "labels": ["A"], "reason": "x"})
        family, task, labels = _parse_classify_json(raw)
        assert family == "General"
        assert task == "semantic"
        assert labels == ["A"]

    def test_valid_family_with_missing_task_still_parses(self):
        family, task, _ = _parse_classify_json(json.dumps({"class": "General"}))
        assert family == "General"
        assert task is None

    def test_valid_family_and_task_unchanged(self):
        family, task, labels = _parse_classify_json(json.dumps(GOOD))
        assert (family, task) == ("General", "semantic")
        assert labels == ["Account Access", "Billing"]

    def test_unusable_response_is_still_rejected(self):
        assert _parse_classify_json("total garbage") is None
        assert _parse_classify_json(json.dumps({"foo": "bar"})) is None
        # Neither a family nor a task in "class", and no usable task either.
        assert (
            _parse_classify_json(
                json.dumps({"class": "nonsense", "task": "not_a_task"})
            )
            is None
        )

    def test_markdown_fenced_json_still_parses(self):
        raw = '```json\n{"class": "General", "task": "semantic"}\n```'
        family, task, _ = _parse_classify_json(raw)
        assert (family, task) == ("General", "semantic")


# --------------------------------------------------------------------------- #
# _llm_infer: one corrective retry when the answer cannot be used
# --------------------------------------------------------------------------- #
def _patch_invoke(monkeypatch, script):
    """Serve scripted answers from llm.invoke; return the call recorder."""
    calls = []

    def _invoke(prompt, max_tokens=400, override=None, budget=None):
        llm._spend_budget(budget)  # exercise the real budget accounting
        calls.append(prompt)
        item = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm, "invoke", _invoke)
    return calls


class TestLlmInferCorrectiveRetry:
    def test_good_first_answer_needs_no_retry(self, monkeypatch):
        calls = _patch_invoke(monkeypatch, [json.dumps(GOOD)])
        family, task, labels, reason = _llm_infer(PAIRS, override=OVERRIDE, budget={"remaining": 3})
        assert (family, task, reason) == ("General", "semantic", "")
        assert labels == ["Account Access", "Billing"]
        assert len(calls) == 1

    def test_mistranslated_class_field_needs_no_retry(self, monkeypatch):
        """The live mistral response now parses on the FIRST attempt."""
        calls = _patch_invoke(monkeypatch, [json.dumps(MISTRAL_STYLE)])
        family, task, labels, reason = _llm_infer(PAIRS, override=OVERRIDE, budget={"remaining": 3})
        assert (family, task, reason) == ("General", "exact_mapping", "")
        assert labels == ["Account Access", "Billing"]
        assert len(calls) == 1

    def test_unusable_answer_is_corrected_by_one_retry(self, monkeypatch):
        bad = "I think this is a classification task."
        calls = _patch_invoke(
            monkeypatch, [bad, json.dumps(GOOD)]
        )
        family, task, labels, reason = _llm_infer(PAIRS, override=OVERRIDE, budget={"remaining": 3})
        assert (family, task, reason) == ("General", "semantic", "")
        assert labels == ["Account Access", "Billing"]
        assert len(calls) == 2
        # The retry shows the model exactly what was rejected and the only
        # acceptable values.
        assert "could not be used" in calls[1]
        assert bad in calls[1]
        assert "String-based" in calls[1]

    def test_retry_skipped_when_budget_has_no_call_left(self, monkeypatch):
        bad = "I think this is a classification task."
        calls = _patch_invoke(monkeypatch, [bad])
        family, task, labels, reason = _llm_infer(PAIRS, override=OVERRIDE, budget={"remaining": 1})
        assert family is None
        assert "recognized class name" in reason
        assert len(calls) == 1  # the budget was spent on the first call only

    def test_provider_error_returns_immediately_without_retry(self, monkeypatch):
        calls = _patch_invoke(monkeypatch, [RuntimeError("rate limited")])
        family, task, labels, reason = _llm_infer(PAIRS, override=OVERRIDE, budget={"remaining": 3})
        assert family is None
        assert "rate limited" in reason
        assert len(calls) == 1

    def test_family_without_task_is_kept_when_retry_fails(self, monkeypatch):
        """A usable family must not be lost because the retry could not help."""
        no_task = json.dumps({"class": "General", "reason": "x"})
        calls = _patch_invoke(monkeypatch, [no_task, RuntimeError("boom")])
        family, task, labels, reason = _llm_infer(PAIRS, override=OVERRIDE, budget={"remaining": 3})
        assert family == "General"
        assert task is None
        assert reason == ""
        assert len(calls) == 2

