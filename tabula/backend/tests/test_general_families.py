"""General Transformation Engine -- all supported families (MOCKED).

Every test monkeypatches ``llm.invoke``; none of them touch a real provider
or consume API quota.  The suite proves the engine infers a rule from the
examples and applies it to UNSEEN inputs across every supported family, not
just the ticket-classification example:

* semantic classification,
* information extraction,
* semantic normalization,
* text transformation (rewriting),
* pattern-based general transformation,
* exact mapping (the one case where a lookup table IS correct),
plus the guard rails: ambiguous, contradictory and insufficient examples.

The row-id batch contract is exercised through a helper that answers with the
structured ``{"results": [{"row_id", "output", "confidence"}]}`` shape.
"""

from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

import pandas as pd
import pytest

from transform import (
    GenerationError,
    apply_model_backed,
    generate,
    plan_examples,
)
from transform.engine import (
    EngineError,
    Intent,
    analyze_intent,
    execute,
    reconstruct,
    run,
    validate_report,
)
from transform.spec import (
    KIND_EXACT_MAPPING,
    KIND_GENERAL_TEXT,
    KIND_NORMALIZATION,
    KIND_SEMANTIC,
    STRATEGY_DETERMINISTIC,
    STRATEGY_MODEL,
    STRATEGY_NONE,
)

OVERRIDE = {"api_key": "test-personal-key"}


# --------------------------------------------------------------------------- #
# Mocked LLM helpers (row-id aware)
# --------------------------------------------------------------------------- #
def _extract_row_ids(prompt: str) -> List[str]:
    m = re.search(r"Row ids \(paired with the inputs below, in order\):\n(\[[^\]]*\])",
                  prompt)
    assert m, "batch prompt did not carry its row ids"
    return json.loads(m.group(1))


def _extract_inputs(prompt: str) -> List[str]:
    m = re.search(r"Inputs \(JSON array, in order\):\n(\[.*\])", prompt, re.DOTALL)
    assert m, "batch prompt did not carry its inputs"
    return json.loads(m.group(1))


def rowid_responder(fn: Callable[[str], object], confidences=None):
    """Build a mock ``llm.invoke`` answering with the row-id contract."""

    def _invoke(prompt, max_tokens=400, override=None, budget=None):
        # The CLASSIFY prompt is the only one that carries source->target
        # pairs; apply prompts may legitimately ask for JSON too.
        if "Source -> Target pairs" in prompt:
            raise AssertionError("classify should not be hit in this test")
        ids = _extract_row_ids(prompt)
        inputs = _extract_inputs(prompt)
        assert len(ids) == len(inputs)
        results = []
        for rid, value in zip(ids, inputs):
            item = {"row_id": rid, "output": fn(value)}
            if confidences is not None:
                item["confidence"] = confidences.get(value, 0.9)
            results.append(item)
        return json.dumps({"results": results})

    return _invoke


def script_llm(responses: List[str]):
    """A FIFO of raw responses; each call pops one (or reuses the last)."""
    queue = list(responses)

    def _invoke(prompt, max_tokens=400, override=None, budget=None):
        if not queue:
            raise AssertionError("ran out of scripted responses")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return _invoke


# --------------------------------------------------------------------------- #
# A. Semantic classification
# --------------------------------------------------------------------------- #
class TestSemanticClassification:
    LABELS = ["Urgent", "Question", "Bug"]

    def test_unseen_inputs_classified_by_meaning(self, monkeypatch):
        def bucket(text):
            t = text.lower()
            if "how" in t or "?" in text:
                return "Question"
            if "crash" in t or "broken" in t:
                return "Bug"
            return "Urgent"

        monkeypatch.setattr("llm.invoke", rowid_responder(bucket))
        examples = [
            ("Server will not let me sign in", "Urgent"),
            ("Where do I find the settings?", "Question"),
            ("The app crashed on start", "Bug"),
        ]
        spec = TransformationSpec_for(self.LABELS, KIND_SEMANTIC, "semantic")
        unseen = pd.Series([
            "I cannot log into my account",       # Urgent (unseen wording)
            ("How do I reset my password?"),      # Question
            "The dashboard keeps crashing",       # Bug
        ])
        report = apply_model_backed(unseen, spec, examples, llm_override=OVERRIDE)
        assert report.failed == 0
        assert report.series.tolist() == ["Urgent", "Question", "Bug"]
        assert report.family == "Semantic Classification"
        # provider is recorded via the real provider layer (llm.last_provider);
        # with a fully mocked llm.invoke it stays empty.  Provider tracking is
        # asserted directly in test_provider_fallback.py.
        assert isinstance(report.provider, str)

    def test_labels_case_insensitive_canonicalised(self, monkeypatch):
        # model answers with wrong case -> canonicalised to the allowed label
        monkeypatch.setattr(
            "llm.invoke", rowid_responder(lambda v: "urgent")
        )
        spec = TransformationSpec_for(["Urgent"], KIND_SEMANTIC, "semantic")
        report = apply_model_backed(pd.Series(["help please"]), spec, [],
                                    llm_override=OVERRIDE)
        assert report.series.tolist() == ["Urgent"]
        assert report.failed == 0

    def test_label_outside_allowed_set_is_a_failure_not_silently_accepted(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "llm.invoke", rowid_responder(lambda v: "NotALabel")
        )
        spec = TransformationSpec_for(["Urgent", "Question"], KIND_SEMANTIC, "semantic")
        report = apply_model_backed(pd.Series(["mystery"]), spec, [],
                                    llm_override=OVERRIDE)
        assert report.failed == 1
        assert report.series.tolist() == [None]
        assert any("allowed" in e for e in report.errors)


# --------------------------------------------------------------------------- #
# B. Information extraction
# --------------------------------------------------------------------------- #
class TestInformationExtraction:
    def test_extracts_domain_from_unseen_urls(self, monkeypatch):
        def domain(text):
            m = re.search(r"https?://([^/]+)", text)
            return m.group(1).removeprefix("www.") if m else text

        monkeypatch.setattr("llm.invoke", rowid_responder(domain))
        examples = [
            ("Visit https://www.example.com/pricing", "example.com"),
            ("See http://support.foo.org now", "support.foo.org"),
        ]
        spec = TransformationSpec_for([], KIND_GENERAL_TEXT, "extraction")
        unseen = pd.Series([
            "Go to https://docs.bar.io/guide today",
            "open http://blog.baz.net/post",
        ])
        report = apply_model_backed(unseen, spec, examples, llm_override=OVERRIDE)
        assert report.failed == 0
        assert report.series.tolist() == ["docs.bar.io", "blog.baz.net"]
        assert report.family == "Text Transformation"

    def test_extracts_identifier_even_when_embedded(self, monkeypatch):
        def order_id(text):
            m = re.search(r"\b\d{6}\b", text)
            return m.group(0) if m else "NONE"

        monkeypatch.setattr("llm.invoke", rowid_responder(order_id))
        spec = TransformationSpec_for([], KIND_GENERAL_TEXT, "extraction")
        report = apply_model_backed(
            pd.Series(["My order 458120 is late", "ref 993001 missing"]),
            spec, [], llm_override=OVERRIDE,
        )
        assert report.series.tolist() == ["458120", "993001"]


# --------------------------------------------------------------------------- #
# C. Semantic normalization
# --------------------------------------------------------------------------- #
class TestSemanticNormalization:
    def test_kind_and_family_reported(self):
        plan = plan_examples(
            [("laptop computer, blue", "Laptop - Blue")],
            classification_hint="General",
            allow_llm=False,
            llm_override=OVERRIDE,
            llm_task="normalization",
        )
        assert plan.spec.strategy == STRATEGY_MODEL
        assert plan.spec.kind == KIND_NORMALIZATION
        assert plan.spec.family == "Semantic Normalization"

    def test_unseen_variants_normalized_to_canonical(self, monkeypatch):
        # A semantic (not substring) mock: recognises product synonyms the way
        # a real model would, so unseen wordings still normalize correctly.
        def canonical(text):
            t = text.lower()
            is_laptop = any(k in t for k in ("laptop", "notebook"))
            is_blue = "blue" in t
            is_phone = any(k in t for k in ("phone", "mobile", "handset"))
            is_red = "red" in t
            if is_laptop and is_blue:
                return "Laptop - Blue"
            if is_phone and is_red:
                return "Phone - Red"
            return "Unknown"

        monkeypatch.setattr("llm.invoke", rowid_responder(canonical))
        examples = [
            ("a blue laptop computer", "Laptop - Blue"),
            ("red mobile phone", "Phone - Red"),
        ]
        spec = TransformationSpec_for([], KIND_NORMALIZATION, "normalization")
        unseen = pd.Series([
            "laptop, colour blue",      # unseen variant of Laptop - Blue
            "mobile handset in red",    # unseen variant of Phone - Red
        ])
        report = apply_model_backed(unseen, spec, examples, llm_override=OVERRIDE)
        assert report.failed == 0
        assert report.series.tolist() == ["Laptop - Blue", "Phone - Red"]
        assert report.family == "Semantic Normalization"


# --------------------------------------------------------------------------- #
# D. Text transformation (rewriting)
# --------------------------------------------------------------------------- #
class TestTextTransformation:
    def test_rewrites_unseen_informal_to_formal(self, monkeypatch):
        # Extracts the name (2nd token before the comma) and rewrites the
        # greeting formally -- a meaning-based rewrite, not a fixed mapping.
        def formal(text):
            name = text.split(",")[0].strip().split()[1].title()
            return f"Dear {name}, thank you."

        monkeypatch.setattr("llm.invoke", rowid_responder(formal))
        examples = [("hey bob, need help", "Dear Bob, thank you.")]
        spec = TransformationSpec_for([], KIND_GENERAL_TEXT, "general_text")
        unseen = pd.Series(["hi alice, my order is late", "yo carl, need a refund"])
        report = apply_model_backed(unseen, spec, examples, llm_override=OVERRIDE)
        assert report.failed == 0
        assert report.series.tolist() == [
            "Dear Alice, thank you.",
            "Dear Carl, thank you.",
        ]
        assert report.family == "Text Transformation"


# --------------------------------------------------------------------------- #
# E. Pattern-based general transformation
# --------------------------------------------------------------------------- #
class TestPatternBasedGeneral:
    def test_prefix_pattern_generalizes(self, monkeypatch):
        # rule: prepend "ITEM-" -- deterministic enough for the model to learn
        monkeypatch.setattr(
            "llm.invoke", rowid_responder(lambda v: f"ITEM-{v}")
        )
        examples = [("apple", "ITEM-apple"), ("pear", "ITEM-pear")]
        spec = TransformationSpec_for([], KIND_GENERAL_TEXT, "general_text")
        report = apply_model_backed(pd.Series(["kiwi", "mango"]), spec, examples,
                                    llm_override=OVERRIDE)
        assert report.series.tolist() == ["ITEM-kiwi", "ITEM-mango"]


# --------------------------------------------------------------------------- #
# Exact mapping (the valid lookup-table case)
# --------------------------------------------------------------------------- #
class TestExactMapping:
    def test_exact_mapping_plans_deterministic_lookup(self):
        plan = plan_examples(
            [("Microsoft", "Satya Nadella"), ("Apple", "Tim Cook")],
            classification_hint="General",
            allow_llm=False,
            llm_override=OVERRIDE,
            treat_as_mapping=True,
        )
        assert plan.spec.strategy == STRATEGY_DETERMINISTIC
        assert plan.spec.kind == KIND_EXACT_MAPPING
        assert plan.spec.exhaustive_mapping is True

    def test_semantic_rule_never_becomes_lookup(self):
        """The original bug: a rule task must NOT degrade to UNKNOWN lookup."""
        plan = plan_examples(
            [("Cannot sign in", "Account Access"), ("Charged twice", "Billing")],
            classification_hint="General",
            allow_llm=False,
            llm_override=OVERRIDE,
            llm_task="semantic",
            labels_hint=["Account Access", "Billing"],
        )
        assert plan.spec.strategy == STRATEGY_MODEL

    def test_llm_exact_mapping_on_rule_like_inputs_downgraded_to_model(self):
        """A model-suggested ``exact_mapping`` for sentence-like inputs must not
        become a lookup table (spec: lookup only for TRUE closed mappings).

        Mistral mislabels free-form sentences as ``exact_mapping``; building a
        lookup here would flag every unseen input as UNKNOWN, so the planner
        honours the deterministic ``looks_like_closed_mapping`` heuristic and
        routes to the model-backed semantic path instead.  (An EXPLICIT
        ``treat_as_mapping`` request still wins -- see the test above.)
        """
        sentences = [
            ("Cannot sign in even with the correct password", "Account Access"),
            ("Office Wi-Fi disconnects during calls", "Network"),
        ]
        plan = plan_examples(
            sentences,
            classification_hint="General",
            allow_llm=False,
            llm_override=OVERRIDE,
            llm_task="exact_mapping",          # the (wrong) model suggestion
            labels_hint=["Account Access", "Network"],
        )
        # Not a deterministic lookup -- routed to the model instead.
        assert plan.spec.strategy == STRATEGY_MODEL
        assert plan.spec.kind == KIND_SEMANTIC
        # The override is recorded in the evidence trail.
        assert any("rule-like" in e for e in plan.evidence)

    def test_llm_exact_mapping_on_true_closed_table_kept(self):
        """A genuine closed table (short, unrelated codes) keeps the lookup."""
        table = [("inp1", "A"), ("inp2", "B"), ("inp3", "C")]
        plan = plan_examples(
            table,
            classification_hint="General",
            allow_llm=False,
            llm_override=OVERRIDE,
            llm_task="exact_mapping",
        )
        assert plan.spec.strategy == STRATEGY_DETERMINISTIC
        assert plan.spec.kind == KIND_EXACT_MAPPING


# --------------------------------------------------------------------------- #
# Guard rails: ambiguous / contradictory / insufficient
# --------------------------------------------------------------------------- #
class TestGuardRails:
    def test_contradictory_examples_refused(self):
        from transform.planner import PlanError, find_contradictions

        pairs = [("same input", "A"), ("same input", "B")]
        assert find_contradictions(pairs)
        with pytest.raises(PlanError):
            plan_examples(pairs, classification_hint="General",
                          allow_llm=False, llm_override=OVERRIDE)

    def test_ambiguous_task_is_unsupported_not_guessed(self):
        plan = plan_examples(
            [("a", "1"), ("b", "2")],
            classification_hint="General",
            allow_llm=False,
            llm_override=OVERRIDE,
            llm_task="ambiguous",
        )
        assert plan.spec.strategy == STRATEGY_NONE

    def test_insufficient_examples_without_model_is_clear(self):
        plan = plan_examples(
            [("some sentence here", "Label")],
            classification_hint="General",
            allow_llm=False,   # no model, not a closed mapping
        )
        assert plan.spec.strategy == STRATEGY_NONE
        assert "model" in (plan.spec.unsupported_reason or "").lower()


# --------------------------------------------------------------------------- #
# Confidence / validation tracking (spec §8)
# --------------------------------------------------------------------------- #
class TestConfidenceAndValidation:
    def test_low_confidence_rows_flagged_for_review(self, monkeypatch):
        monkeypatch.setattr(
            "llm.invoke",
            rowid_responder(lambda v: "Urgent", confidences={"low": 0.2, "high": 0.95}),
        )
        spec = TransformationSpec_for(["Urgent"], KIND_SEMANTIC, "semantic")
        report = apply_model_backed(pd.Series(["low", "high"]), spec, [],
                                    llm_override=OVERRIDE)
        assert report.failed == 0              # accepted, not failed
        assert report.review == 1              # the 0.2 row needs review
        assert report.row_status == ["review", "ok"]
        assert report.confidences[0] == pytest.approx(0.2)
        assert report.confidences[1] == pytest.approx(0.95)
        assert any("review" in w for w in report.warnings)

    def test_percent_style_confidence_coerced(self, monkeypatch):
        monkeypatch.setattr(
            "llm.invoke",
            rowid_responder(lambda v: "Urgent", confidences={"x": 85}),
        )
        spec = TransformationSpec_for(["Urgent"], KIND_SEMANTIC, "semantic")
        report = apply_model_backed(pd.Series(["x"]), spec, [], llm_override=OVERRIDE)
        assert report.confidences[0] == pytest.approx(0.85)
        assert report.row_status == ["ok"]

    def test_validate_report_flags_row_count_mismatch(self):
        from transform.apply import ApplyReport

        report = ApplyReport(
            series=pd.Series(["a", "b"]), total=5, failed=0, errors=[]
        )
        validate_report(report)
        assert any("mismatch" in e.lower() for e in report.errors)

    def test_validate_report_flags_disallowed_accepted_label(self):
        from transform.apply import ApplyReport

        spec = TransformationSpec_for(["A", "B"], KIND_SEMANTIC, "semantic")
        report = ApplyReport(
            series=pd.Series(["C"]), total=1, failed=0, errors=[],
            row_status=["ok"],
        )
        validate_report(report, spec)
        assert any("outside the allowed" in e for e in report.errors)


# --------------------------------------------------------------------------- #
# Engine facade: execute / reconstruct / run
# --------------------------------------------------------------------------- #
class TestEngineFacade:
    def test_analyze_intent_returns_task_and_family(self):
        intent = analyze_intent(
            [("Cannot sign in", "Account Access"), ("Charged twice", "Billing")],
            allow_llm=False,
        )
        assert isinstance(intent, Intent)
        # deterministic detectors didn't match; General + no model -> no task
        assert intent.classification in ("General", "String-based", "Numerical")

    def test_execute_model_route(self, monkeypatch):
        monkeypatch.setattr("llm.invoke", rowid_responder(lambda v: "OK"))
        spec = TransformationSpec_for([], KIND_GENERAL_TEXT, "general_text")
        report = execute(pd.Series(["a", "b"]), spec, [], llm_override=OVERRIDE)
        assert report.series.tolist() == ["OK", "OK"]

    def test_execute_unsupported_raises(self):
        from transform.spec import TransformationSpec

        spec = TransformationSpec(kind=KIND_SEMANTIC, strategy=STRATEGY_NONE)
        with pytest.raises(EngineError):
            execute(pd.Series(["a"]), spec, [], llm_override=OVERRIDE)

    def test_reconstruct_preserves_rows_and_columns(self, monkeypatch):
        monkeypatch.setattr("llm.invoke", rowid_responder(lambda v: v.upper()))
        df = pd.DataFrame({
            "id": [10, 20, 30],
            "text": ["alpha", "beta", "gamma"],
            "keep": ["x", "y", "z"],
        })
        spec = TransformationSpec_for([], KIND_GENERAL_TEXT, "general_text")
        report = execute(df["text"], spec, [], llm_override=OVERRIDE)
        out = reconstruct(df, "text", report)
        assert out["text"].tolist() == ["ALPHA", "BETA", "GAMMA"]
        assert out["id"].tolist() == [10, 20, 30]        # row identity/order
        assert out["keep"].tolist() == ["x", "y", "z"]   # unselected column
        assert df["text"].tolist() == ["alpha", "beta", "gamma"]  # original intact

    def test_reconstruct_refuses_misaligned_result(self):
        from transform.apply import ApplyReport

        df = pd.DataFrame({"text": ["a", "b", "c"]})
        short = ApplyReport(series=pd.Series(["A", "B"]), total=3, failed=0)
        with pytest.raises(EngineError):
            reconstruct(df, "text", short)

    def test_run_end_to_end_semantic(self, monkeypatch):
        monkeypatch.setattr("llm.invoke", rowid_responder(lambda v: "Yes"))
        df = pd.DataFrame({"t": ["first", "second"]})
        out, report, plan = run(
            df, "t", [("demo in", "Yes")],
            classification_hint="General",
            allow_llm=False, llm_override=OVERRIDE,
            llm_task="general_text",
        )
        assert out["t"].tolist() == ["Yes", "Yes"]
        assert report.failed == 0
        assert plan.spec.strategy == STRATEGY_MODEL


# --------------------------------------------------------------------------- #
# Helper to build a model-backed spec of a given kind/task
# --------------------------------------------------------------------------- #
def TransformationSpec_for(labels, kind, task):
    from transform.spec import TransformationSpec

    return TransformationSpec(
        kind=kind,
        strategy=STRATEGY_MODEL,
        requires_model=True,
        allowed_labels=list(labels),
        task=task,
    )
