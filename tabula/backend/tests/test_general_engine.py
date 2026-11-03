"""General transformation engine regression tests (spec test groups A-G).

Layers (kept deliberately separate):

* deterministic unit tests -- planner / generators, no LLM involved;
* mocked model-backed tests -- ``llm.invoke`` is monkeypatched, so running the
  suite consumes **no live API quota**;
* endpoint tests through the FastAPI app -- also with a mocked LLM.

Fixture data comes from the repo fixture packs:

* ``tabulax_test_formats_v2/``            -- numeric, string, inventory, ...
* ``tabulax_algorithmic_general_tests/``  -- roman numerals, ticket categories

The ticket dataset is ONE regression case among many, never a definition of
the engine: nothing in ``transform/`` is written for these specific values.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import pytest

from transform import (
    GenerationError,
    apply_model_backed,
    apply_to_series,
    classify,
    generate,
    parse_examples,
    plan_examples,
)
from transform.classify import _affix_swap
from transform.generate import _exact_lookup_code, validate_code_against_examples
from transform.planner import PlanError, looks_like_closed_mapping
from transform.spec import (
    KIND_EXACT_MAPPING,
    KIND_GENERAL_TEXT,
    KIND_SEMANTIC,
    STRATEGY_DETERMINISTIC,
    STRATEGY_MODEL,
    STRATEGY_NONE,
    TransformationSpec,
)

REPO = Path(__file__).resolve().parents[3]          # tabulax-project/
PACK_V2 = REPO / "tabulax_test_formats_v2"
PACK_GEN = REPO / "tabulax_algorithmic_general_tests" / "tabulax_algorithmic_general_tests"

# Transient personal key routes every code path through (mocked) llm.invoke.
OVERRIDE = {"api_key": "test-personal-key"}


def _require(path: Path) -> Path:
    if not path.exists():  # pragma: no cover - fixture pack moved/renamed
        pytest.skip(f"fixture missing: {path}")
    return path


# --------------------------------------------------------------------------- #
# Mocked LLM (no live API calls)
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Scripted stand-in for ``llm.invoke`` with a call recorder."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.classify_response: Optional[dict] = None
        self.batch_fn: Optional[Callable[[str], object]] = None
        self.responses: List[str] = []      # raw scripted responses (FIFO)
        self.raise_exc: Optional[Exception] = None

    @property
    def batch_calls(self) -> List[str]:
        return [p for p in self.calls if "Inputs (JSON array, in order):" in p]

    @staticmethod
    def batch_inputs(prompt: str) -> List[str]:
        m = re.search(r"Inputs \(JSON array, in order\):\n(\[.*\])", prompt, re.DOTALL)
        assert m, "batch prompt did not carry its inputs"
        return json.loads(m.group(1))

    def respond(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.responses:
            return self.responses.pop(0)
        if "Source -> Target pairs" in prompt and self.classify_response is not None:
            return json.dumps(self.classify_response)
        if self.batch_fn is not None and "Inputs (JSON array, in order):" in prompt:
            inputs = self.batch_inputs(prompt)
            return json.dumps({"outputs": [self.batch_fn(x) for x in inputs]})
        raise AssertionError("unexpected LLM call: " + prompt[:80])


@pytest.fixture()
def fake_llm(monkeypatch):
    import llm as llm_mod

    fake = FakeLLM()

    def _invoke(prompt, max_tokens=400, override=None, budget=None):
        llm_mod._spend_budget(budget)   # exercise the real budget accounting
        return fake.respond(prompt)

    monkeypatch.setattr(llm_mod, "invoke", _invoke)
    return fake


# --------------------------------------------------------------------------- #
# Shared data
# --------------------------------------------------------------------------- #
TICKET_LABELS = ["Account Access", "Billing", "Hardware", "Network"]
TICKET_EXAMPLES = [
    ("Cannot sign in even with the correct password", "Account Access"),
    ("Subscription charged twice this month", "Billing"),
    ("Laptop shuts down whenever I open it", "Hardware"),
    ("Office Wi-Fi disconnects during calls", "Network"),
]
TICKET_EXPECTED = {
    "GEN-001": "Account Access",
    "GEN-002": "Account Access",
    "GEN-003": "Billing",
    "GEN-004": "Billing",
    "GEN-005": "Hardware",
    "GEN-006": "Hardware",
    "GEN-007": "Network",
    "GEN-008": "Network",
}


def ticket_frame() -> pd.DataFrame:
    path = _require(PACK_GEN / "csv" / "general_ticket_categories.csv")
    return pd.read_csv(path)


def keyword_classifier(text: str) -> str:
    """Meaning-based mock (the model's job at apply time), not string matching
    against the examples -- unseen wording still lands in the right bucket."""
    t = text.lower()
    if any(k in t for k in ("wi-fi", "wifi", "vpn", "network", "disconnect")):
        return "Network"
    if any(k in t for k in ("sign in", "password", "login", "reset link", "locked")):
        return "Account Access"
    if any(k in t for k in ("bill", "charge", "refund", "subscription", "cancel")):
        return "Billing"
    if any(k in t for k in ("laptop", "screen", "monitor", "keyboard", "flicker", "shuts down")):
        return "Hardware"
    return "Account Access"  # pragma: no cover - fixture fully covered


SEMANTIC_CLASSIFY = {
    "class": "General",
    "task": "semantic",
    "labels": TICKET_LABELS,
    "reason": "outputs are closed category labels decided by meaning",
}


# =========================================================================== #
# Test group A -- semantic classification
# =========================================================================== #
class TestGroupA_SemanticClassification:
    def test_planner_routes_semantic_examples_to_model_strategy(self, fake_llm):
        fake_llm.classify_response = SEMANTIC_CLASSIFY
        plan = plan_examples(
            TICKET_EXAMPLES,
            classification_hint="General",
            allow_llm=False,
            llm_override=OVERRIDE,
            source_column="ticket_description",
        )
        assert plan.spec.strategy == STRATEGY_MODEL
        assert plan.spec.kind == KIND_SEMANTIC
        assert plan.spec.allowed_labels == TICKET_LABELS
        assert plan.classification == "General"

    def test_ticket_fixture_unseen_rows_classified_by_meaning(self, fake_llm):
        df = ticket_frame()
        fake_llm.classify_response = SEMANTIC_CLASSIFY
        plan = plan_examples(
            TICKET_EXAMPLES, classification_hint="General",
            allow_llm=False, llm_override=OVERRIDE,
        )
        fake_llm.batch_fn = keyword_classifier

        col = df["ticket_description"]
        report = apply_model_backed(col, plan.spec, TICKET_EXAMPLES, llm_override=OVERRIDE)
        assert report.failed == 0
        got = dict(zip(df["ticket_id"], report.series))
        assert got == TICKET_EXPECTED
        # Rows arrive at the model exactly once, in original order.
        assert list(df["ticket_id"]) == list(got)

    def test_generalizes_to_unseen_wording_not_exact_sentences(self, fake_llm):
        """The 4 holdout descriptions share no exact string with the examples."""
        df = ticket_frame()
        example_inputs = {e for e, _ in TICKET_EXAMPLES}
        unseen = [d for d in df["ticket_description"] if d not in example_inputs]
        assert len(unseen) == 4

        fake_llm.classify_response = SEMANTIC_CLASSIFY
        plan = plan_examples(
            TICKET_EXAMPLES, classification_hint="General",
            allow_llm=False, llm_override=OVERRIDE,
        )
        fake_llm.batch_fn = keyword_classifier
        report = apply_model_backed(pd.Series(unseen), plan.spec, TICKET_EXAMPLES,
                                    llm_override=OVERRIDE)
        assert report.failed == 0
        # A lookup table could not answer these; the model-backed path does.
        assert all(v in TICKET_LABELS for v in report.series)
        assert report.series.tolist() == [
            keyword_classifier(u) for u in unseen
        ]

    def test_sentiment_domain_routes_to_model_and_generalizes(self, fake_llm):
        pairs = parse_examples(
            "Absolutely wonderful experience -> Positive\n"
            "This broke immediately, terrible -> Negative"
        )
        fake_llm.classify_response = {
            "class": "General", "task": "semantic",
            "labels": ["Positive", "Negative"],
            "reason": "judgement labels",
        }
        plan = plan_examples(pairs, classification_hint="General",
                             allow_llm=False, llm_override=OVERRIDE)
        assert plan.spec.strategy == STRATEGY_MODEL
        assert plan.spec.allowed_labels == ["Positive", "Negative"]

        fake_llm.batch_fn = lambda t: (
            "Negative" if any(w in t.lower() for w in ("hate", "awful", "slow"))
            else "Positive"
        )
        report = apply_model_backed(
            pd.Series(["I hate how slow it is", "Best purchase this year"]),
            plan.spec, pairs, llm_override=OVERRIDE,
        )
        assert report.series.tolist() == ["Negative", "Positive"]
        assert report.failed == 0

    def test_intent_domain_routes_to_model(self, fake_llm):
        pairs = parse_examples(
            "Where is my order -> OrderStatus\nCancel my subscription -> Cancellation"
        )
        fake_llm.classify_response = {
            "class": "General", "task": "semantic",
            "labels": ["OrderStatus", "Cancellation"], "reason": "intent labels",
        }
        plan = plan_examples(pairs, classification_hint="General",
                             allow_llm=False, llm_override=OVERRIDE)
        assert plan.spec.strategy == STRATEGY_MODEL
        fake_llm.batch_fn = lambda t: (
            "Cancellation" if "cancel" in t.lower() else "OrderStatus"
        )
        report = apply_model_backed(pd.Series(["Please cancel this order"]),
                                    plan.spec, pairs, llm_override=OVERRIDE)
        assert report.series.tolist() == ["Cancellation"]

    def test_rule_like_examples_without_model_report_requirement(self):
        """ROOT-CAUSE regression: never silently emit a lookup returning UNKNOWN."""
        with pytest.raises(GenerationError) as exc:
            generate(TICKET_EXAMPLES, "General", allow_llm=False)
        msg = str(exc.value).lower()
        assert "model" in msg or "llm" in msg or "openai_api_key" in msg

        plan = plan_examples(TICKET_EXAMPLES, classification_hint="General",
                             allow_llm=False)
        assert plan.spec.strategy == STRATEGY_NONE
        assert plan.spec.model_required_flag is True
        assert plan.spec.unsupported_reason

    def test_old_lookup_behaviour_is_the_documented_failure_not_the_fix(self):
        """Direct proof the old code path mislabels unseen rows."""
        df = ticket_frame()
        old_code = _exact_lookup_code(TICKET_EXAMPLES)
        out = apply_to_series(old_code, df["ticket_description"]).series.tolist()
        unseen_wrong = [
            (d, v) for d, v in zip(df["ticket_description"], out)
            if d not in {e for e, _ in TICKET_EXAMPLES}
        ]
        assert unseen_wrong, "the legacy lookup must fail unseen rows"
        assert all(v == "UNKNOWN" for _, v in unseen_wrong)


# =========================================================================== #
# Test group B -- information extraction
# =========================================================================== #
class TestGroupB_InformationExtraction:
    def test_extract_order_id_from_unseen_text(self, fake_llm):
        pairs = [
            ("Order 55322 has shipped", "55322"),
            ("Your order 88913 is delayed", "88913"),
        ]
        fake_llm.classify_response = {
            "class": "General", "task": "extraction", "labels": [],
            "reason": "output is a piece of information from the input",
        }
        plan = plan_examples(pairs, classification_hint="General",
                             allow_llm=False, llm_override=OVERRIDE)
        assert plan.spec.strategy == STRATEGY_MODEL
        assert plan.spec.kind == KIND_GENERAL_TEXT

        fake_llm.batch_fn = lambda t: (
            re.search(r"\d{4,}", t).group(0) if re.search(r"\d{4,}", t) else ""
        )
        report = apply_model_backed(
            pd.Series(["Package 77401 was returned", "Ref for 66250 accepted"]),
            plan.spec, pairs, llm_override=OVERRIDE,
        )
        assert report.series.tolist() == ["77401", "66250"]
        assert report.failed == 0

    def test_extract_date_from_unseen_wording(self, fake_llm):
        pairs = [
            ("Meeting on 2024-05-01 with Ana", "2024-05-01"),
            ("Deadline 2024-06-15 confirmed", "2024-06-15"),
        ]
        fake_llm.classify_response = {
            "class": "General", "task": "extraction", "labels": [],
            "reason": "extract the embedded date",
        }
        plan = plan_examples(pairs, classification_hint="General",
                             allow_llm=False, llm_override=OVERRIDE)
        assert plan.spec.strategy == STRATEGY_MODEL

        fake_llm.batch_fn = lambda t: (
            re.search(r"\d{4}-\d{2}-\d{2}", t).group(0)
            if re.search(r"\d{4}-\d{2}-\d{2}", t) else "NONE"
        )
        report = apply_model_backed(
            pd.Series(["Invoice dated 2025-11-30 arrived", "No date here"]),
            plan.spec, pairs, llm_override=OVERRIDE,
        )
        assert report.series.tolist() == ["2025-11-30", "NONE"]


# =========================================================================== #
# Test group C -- text normalization
# =========================================================================== #
class TestGroupC_TextNormalization:
    def test_uppercase_names_reusable_on_unseen_rows(self):
        path = _require(PACK_V2 / "csv" / "string_transform.csv")
        df = pd.read_csv(path)
        pairs = parse_examples("ananya -> ANANYA\nkiran -> KIRAN")
        g = generate(pairs, "String-based", allow_llm=False)
        assert g.used_llm is False
        out = apply_to_series(g.code, df["first_name"])
        assert out.series.tolist() == [
            "ANANYA", "KIRAN", "MEERA", "ARJUN", "DIVYA"
        ]

    def test_inventory_prefix_swap_is_deterministic_and_reusable(self):
        """The SKU->ITEM fixture: inferred from shape, not memorised rows."""
        path = _require(PACK_V2 / "csv" / "inventory.csv")
        df = pd.read_csv(path)
        pairs = parse_examples("SKU-001 -> ITEM-001\nSKU-002 -> ITEM-002")
        assert _affix_swap(pairs) == ("prefix", "SKU-", "ITEM-")

        r = classify(pairs, allow_llm=False)
        assert r.classification == "String-based"
        assert r.used_llm is False

        g = generate(pairs, "String-based", allow_llm=False)
        assert g.used_llm is False
        out = apply_to_series(g.code, df["sku"])
        assert out.series.tolist() == [f"ITEM-{i:03d}" for i in range(1, 6)]
        # Unseen SKUs (SKU-003..005) generalise; unrelated columns untouched.
        assert df["sku"].tolist()[2:] == ["SKU-003", "SKU-004", "SKU-005"]

    def test_suffix_swap_rule(self):
        pairs = [("readme.txt", "readme.md"), ("notes.txt", "notes.md")]
        assert _affix_swap(pairs) == ("suffix", ".txt", ".md")
        g = generate(pairs, "String-based", allow_llm=False)
        assert g.used_llm is False
        out = apply_to_series(g.code, pd.Series(["license.txt", "plain.md"]))
        # only values ending in '.txt' change; others pass through
        assert out.series.tolist() == ["license.md", "plain.md"]

    def test_open_ended_string_code_is_validated_against_examples(self, fake_llm):
        pairs = [("alpha", "ALPHA!"), ("beta", "BETA!")]
        fake_llm.responses = [
            "```python\ndef transform(x):\n    return str(x).upper() + '!'\n```",
        ]
        g = generate(pairs, "String-based", allow_llm=False, llm_override=OVERRIDE)
        assert g.used_llm is True
        # accepted only because it reproduces EVERY example
        assert validate_code_against_examples(g.code, pairs) == []

    def test_wrong_llm_code_is_retried_then_rejected(self, fake_llm):
        pairs = [(">>alpha<<", "A"), (">>beta<<", "B")]
        bad = "```python\ndef transform(x):\n    return str(x)\n```"
        fake_llm.responses = [bad, bad]  # first attempt + corrective retry
        with pytest.raises(GenerationError) as exc:
            generate(pairs, "String-based", allow_llm=False, llm_override=OVERRIDE)
        assert "reproduce" in str(exc.value) or "examples" in str(exc.value)

    def test_validate_code_against_examples_reports_mismatch(self):
        code = "def transform(x):\n    return str(x).upper()"
        pairs = [("a", "A"), ("b", "c")]  # second expectation is wrong
        msgs = validate_code_against_examples(code, pairs)
        assert len(msgs) == 1
        assert "'b'" in msgs[0]


# =========================================================================== #
# Test group D -- structured algorithmic transformations
# =========================================================================== #
class TestGroupD_Algorithmic:
    ROMAN_EXPECTED = ["1", "4", "9", "12", "14", "40", "64", "90", "120", "400"]

    def test_roman_fixture_routes_to_builtin_template(self):
        path = _require(PACK_GEN / "csv" / "algorithmic_roman_numerals.csv")
        df = pd.read_csv(path)
        pairs = parse_examples("I -> 1\nIV -> 4\nIX -> 9\nXII -> 12")

        r = classify(pairs, allow_llm=False)
        assert r.classification == "Algorithmic"
        assert r.template == "roman_to_decimal"

        g = generate(pairs, "Algorithmic", template=r.template, allow_llm=False)
        assert g.used_llm is False
        assert g.spec is not None and g.spec.strategy == STRATEGY_DETERMINISTIC
        out = apply_to_series(g.code, df["roman_numeral"])
        # holdout rows ALG-005..ALG-010 are beyond the supplied examples
        assert out.series.tolist() == self.ROMAN_EXPECTED

    def test_decimal_to_binary_generalizes_beyond_examples(self):
        pairs = parse_examples("10 -> 1010\n5 -> 101")
        g = generate(pairs, "Algorithmic", allow_llm=False)
        assert g.template == "decimal_to_binary"
        out = apply_to_series(g.code, pd.Series(["8", "2", "13"]))
        assert out.series.tolist() == ["1000", "10", "1101"]

    def test_date_to_weekday_generalizes_beyond_examples(self):
        pairs = parse_examples("2023-04-12 -> Wednesday\n2023-04-13 -> Thursday")
        g = generate(pairs, "Algorithmic", allow_llm=False)
        out = apply_to_series(g.code, pd.Series(["2023-04-14", "2023-04-17"]))
        assert out.series.tolist() == ["Friday", "Monday"]


# =========================================================================== #
# Test group E -- numerical and string regression
# =========================================================================== #
class TestGroupE_NumericStringRegression:
    def test_numeric_times_ten_full_column(self):
        path = _require(PACK_V2 / "csv" / "numeric_transform.csv")
        df = pd.read_csv(path)
        pairs = parse_examples("7 -> 70\n8 -> 80")
        g = generate(pairs, "Numerical", allow_llm=False)
        assert g.used_llm is False
        out = apply_to_series(g.code, df["score"])
        assert [int(v) for v in out.series] == [70, 80, 60, 50, 70]

    def test_exact_mapping_still_supported_without_llm(self):
        pairs = parse_examples("France -> Paris\nJapan -> Tokyo")
        plan = plan_examples(pairs, classification_hint="General", allow_llm=False)
        assert plan.spec.strategy == STRATEGY_DETERMINISTIC
        assert plan.spec.kind == KIND_EXACT_MAPPING

        g = generate(pairs, "General", allow_llm=False)
        assert g.used_llm is False
        out = apply_to_series(g.code, pd.Series(["France", "Canada"]))
        assert out.series.tolist() == ["Paris", "UNKNOWN"]

    def test_user_forced_exact_mapping(self):
        pairs = parse_examples("Password reset link has expired -> Account Access")
        g = generate(pairs, "General", allow_llm=False, treat_as_mapping=True)
        assert g.spec is not None and g.spec.exhaustive_mapping is True
        out = apply_to_series(g.code, pd.Series(["Password reset link has expired", "x"]))
        assert out.series.tolist() == ["Account Access", "UNKNOWN"]

    def test_deterministic_string_rules_unchanged(self):
        pairs = parse_examples("john doe -> John Doe\njane smith -> Jane Smith")
        g = generate(pairs, "String-based", allow_llm=False)
        assert g.used_llm is False
        out = apply_to_series(g.code, pd.Series(["alice johnson"]))
        assert out.series.tolist() == ["Alice Johnson"]

    def test_deterministic_string_rules_normalise_input_whitespace(self):
        """Surrounding spaces are normalised, like the examples themselves.

        Examples are trimmed when parsed, so a rule can never learn "strip"
        from them.  The code must therefore apply the inferred rule to the
        trimmed value, otherwise it would not reproduce its own examples on
        real data ('  carol king  ' -> '  Carol King  ' instead of
        'Carol King').
        """
        pairs = parse_examples("alice smith -> Alice Smith\nbob jones -> Bob Jones")
        for classification in ("String-based",):
            g = generate(pairs, classification, allow_llm=False)
            assert g.used_llm is False
            out = apply_to_series(
                g.code, pd.Series(["  carol king  ", "dave lee\t"])
            )
            assert out.series.tolist() == ["Carol King", "Dave Lee"]

        # The same holds for the other deterministic string rules.
        up = generate(
            parse_examples("anna -> ANNA\nben -> BEN"), "String-based",
            allow_llm=False,
        )
        assert apply_to_series(up.code, pd.Series(["  cy  "])).series.tolist() == [
            "CY"
        ]
        swap = generate(
            parse_examples("SKU-1 -> ITEM-1\nSKU-2 -> ITEM-2"), "String-based",
            allow_llm=False,
        )
        assert apply_to_series(
            swap.code, pd.Series(["  SKU-9  ", "ABC-9"])
        ).series.tolist() == ["ITEM-9", "ABC-9"]


# =========================================================================== #
# Test group F -- ambiguity and failure handling
# =========================================================================== #
class TestGroupF_AmbiguityAndFailures:
    def test_contradictory_examples_rejected_not_guessed(self):
        with pytest.raises(PlanError) as exc:
            plan_examples([("x", "1"), ("x", "2")], classification_hint="General",
                          allow_llm=False)
        assert "contradict" in str(exc.value)

    def test_ambiguous_model_decision_reports_instead_of_fabricating(self, fake_llm):
        pairs = [("abc", "1"), ("def", "2")]
        fake_llm.classify_response = {
            "class": "General", "task": "ambiguous", "labels": [],
            "reason": "too few examples",
        }
        with pytest.raises(GenerationError) as exc:
            generate(pairs, "General", allow_llm=False, llm_override=OVERRIDE)
        msg = str(exc.value)
        assert "examples" in msg.lower()
        assert "add more" in msg.lower() or "single transformation" in msg.lower()

    def test_closed_mapping_heuristic_shape_only(self):
        assert looks_like_closed_mapping([("Microsoft", "Satya Nadella"),
                                          ("Apple", "Tim Cook")])
        # sentence-like inputs are rule-like, not a closed table
        assert not looks_like_closed_mapping(TICKET_EXAMPLES)

    def test_malformed_model_json_fails_batch_without_corrupting_others(self, fake_llm):
        spec = TransformationSpec(
            kind=KIND_SEMANTIC, allowed_labels=TICKET_LABELS,
            strategy=STRATEGY_MODEL, requires_model=True,
        )
        values = pd.Series(["a1", "a2", "b1", "b2", "c1", "c2"])
        seen = {"n": 0}

        def fn(_text: str) -> str:
            seen["n"] += 1
            return "Billing"

        fake_llm.batch_fn = fn
        # batch 1 (rows 0-4, batch_size=5) good; batch 2 malformed twice.
        real_respond = fake_llm.respond
        calls = {"batch": 0}

        def respond(prompt: str) -> str:
            if "Inputs (JSON array, in order):" in prompt:
                inputs = FakeLLM.batch_inputs(prompt)
                calls["batch"] += 1
                if calls["batch"] == 1:
                    fake_llm.calls.append(prompt)
                    return json.dumps({"outputs": [fn(x) for x in inputs]})
                fake_llm.calls.append(prompt)
                return "sorry, not json at all"
            return real_respond(prompt)

        fake_llm.respond = respond
        report = apply_model_backed(values, spec, [], llm_override=OVERRIDE,
                                    batch_size=5)
        assert report.total == 6
        # first batch (5 rows) succeeded, second batch (1 row) failed cleanly
        assert report.failed == 1
        assert report.series[:5].tolist() == ["Billing"] * 5
        assert report.errors

    def test_wrong_output_count_rejected_then_reported(self, fake_llm):
        spec = TransformationSpec(kind=KIND_SEMANTIC, allowed_labels=["A", "B"],
                                  strategy=STRATEGY_MODEL)
        # two scripted responses: both short (wrong count) -> retried then failed
        fake_llm.responses = ['["A"]', '["A"]']
        report = apply_model_backed(pd.Series(["x", "y"]), spec, [],
                                    llm_override=OVERRIDE)
        assert report.failed == 2
        assert report.series.tolist() == [None, None]
        assert report.errors

    def test_label_outside_allowed_set_not_accepted(self, fake_llm):
        spec = TransformationSpec(kind=KIND_SEMANTIC, allowed_labels=["A", "B"],
                                  strategy=STRATEGY_MODEL)
        fake_llm.responses = ['["A", "C"]', '["A", "C"]']  # C is not allowed
        report = apply_model_backed(pd.Series(["x", "y"]), spec, [],
                                    llm_override=OVERRIDE)
        # the disallowed row fails; the allowed row succeeds
        assert report.failed == 1
        assert report.series.tolist()[0] == "A"
        assert report.series.tolist()[1] is None
        assert any("allowed" in e for e in report.errors)

    def test_provider_error_reports_failure_without_raising(self, fake_llm):
        spec = TransformationSpec(kind=KIND_SEMANTIC, strategy=STRATEGY_MODEL)
        fake_llm.raise_exc = RuntimeError("network down")
        report = apply_model_backed(pd.Series(["a", "b"]), spec, [],
                                    llm_override=OVERRIDE)
        assert report.failed == 2
        assert any("request failed" in e for e in report.errors)

    def test_provider_failure_at_plan_time_reports_the_real_error(self, fake_llm):
        """A configured-but-failing model must NOT produce the
        'configure OPENAI_API_KEY' message (the key IS configured)."""
        fake_llm.raise_exc = RuntimeError("rate limited")
        with pytest.raises(GenerationError) as exc:
            generate(TICKET_EXAMPLES, "General",
                     allow_llm=True, llm_override=OVERRIDE)
        msg = str(exc.value)
        assert "Configure OPENAI_API_KEY" not in msg
        assert "model request failed" in msg
        assert "rate limit" in msg.lower() or "retry" in msg.lower()

    def test_unconfigured_model_still_gets_configure_message(self):
        """The genuinely-unconfigured branch keeps its original guidance."""
        plan = plan_examples(TICKET_EXAMPLES, classification_hint="General",
                             allow_llm=False)
        reason = plan.spec.unsupported_reason or ""
        assert "Configure OPENAI_API_KEY" in reason
        assert "model request failed" not in reason

    def test_classify_exposes_llm_error_when_model_fails(self, fake_llm):
        fake_llm.raise_exc = RuntimeError("rate limited")
        r = classify(TICKET_EXAMPLES, allow_llm=True, llm_override=OVERRIDE)
        assert r.classification == "General"
        assert r.task is None
        assert r.llm_error and "rate limit" in r.llm_error.lower()

    def test_classify_has_no_llm_error_when_model_answers(self, fake_llm):
        fake_llm.classify_response = SEMANTIC_CLASSIFY
        r = classify(TICKET_EXAMPLES, allow_llm=True, llm_override=OVERRIDE)
        assert r.used_llm is True
        assert r.llm_error is None
        assert r.task == "semantic"

    def test_classify_downgrades_llm_algorithmic_without_template(self, fake_llm):
        """A model answer of 'Algorithmic' with NO deterministic template match
        (steps 1-5 fell through) must not be trusted: there is no real
        algorithm behind it, so it is downgraded to the rule-based 'General'
        family.  Otherwise generate() would author a fake transform(x) that
        fabricates (or nulls out) unseen inputs -- the exact bug the spec bars.
        """
        # Sentence-like pairs: no date/base/encoding/numeric template matches,
        # yet the (mis)classified model insists on 'Algorithmic'.
        fake_llm.classify_response = {
            "class": "Algorithmic",
            "task": "semantic",
            "labels": TICKET_LABELS,
            "reason": "the model wrongly thinks this is an algorithm",
        }
        r = classify(TICKET_EXAMPLES, allow_llm=True, llm_override=OVERRIDE)
        # Deterministic detectors did not match -> family is downgraded.
        assert r.classification == "General"
        # The task survives (it is the meaningful part of the answer).
        assert r.task == "semantic"
        assert any("no deterministic" in e for e in r.evidence)

    def test_classify_endpoint_surfaces_llm_error(self, client, auth_headers,
                                                  fake_llm):
        fake_llm.raise_exc = RuntimeError("rate limited")
        ex = [{"input": s, "output": t} for s, t in TICKET_EXAMPLES]
        r = client.post("/classify/", headers=auth_headers,
                        json={"column": "ticket_description", "examples": ex,
                              "llm_config": {"api_key": "test-personal-key"}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["classification"] == "General"
        assert body["task"] is None
        assert body["llm_error"] and "rate limit" in body["llm_error"].lower()

    def test_generate_endpoint_reports_model_failure_not_missing_key(
        self, client, auth_headers, fake_llm
    ):
        """The exact reported bug: rate-limited model must yield the REAL
        reason, not 'Configure OPENAI_API_KEY'."""
        fake_llm.raise_exc = RuntimeError("rate limited")
        ex = [{"input": s, "output": t} for s, t in TICKET_EXAMPLES]
        r = client.post("/generate_function/", headers=auth_headers,
                        json={"column": "ticket_description", "examples": ex,
                              "classification": "General",
                              "llm_config": {"api_key": "test-personal-key"}})
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert "Configure OPENAI_API_KEY" not in detail
        assert "model request failed" in detail
        assert "rate limit" in detail.lower()

    def test_unavailable_provider_stops_with_clear_reason(self, fake_llm):
        import llm as llm_mod

        spec = TransformationSpec(kind=KIND_SEMANTIC, strategy=STRATEGY_MODEL)
        fake_llm.raise_exc = llm_mod.LLMUnavailableError("no provider configured")
        report = apply_model_backed(pd.Series(["a", "b"]), spec, [],
                                    llm_override=OVERRIDE)
        assert report.failed == 2
        assert any("provider" in e or "LLM" in e for e in report.errors)

    def test_nulls_and_blank_strings_pass_through(self, fake_llm):
        spec = TransformationSpec(kind=KIND_SEMANTIC, allowed_labels=["A"],
                                  strategy=STRATEGY_MODEL)
        fake_llm.batch_fn = lambda t: "A"
        report = apply_model_backed(
            pd.Series([None, "", "   ", "real"]), spec, [], llm_override=OVERRIDE
        )
        assert report.series.tolist()[0] is None
        assert report.series.tolist()[1] == ""
        assert report.series.tolist()[2] == "   "
        assert report.series.tolist()[3] == "A"
        assert report.failed == 0
        # blanks/nulls never reach the model: exactly one batch, one input
        assert len(fake_llm.batch_calls) == 1
        assert FakeLLM.batch_inputs(fake_llm.batch_calls[0]) == ["real"]


# =========================================================================== #
# Test group G -- data integrity (order, ids, columns, preview/apply/revert)
# =========================================================================== #
class TestGroupG_DataIntegrity:
    def test_model_backed_preserves_index_and_row_order(self, fake_llm):
        spec = TransformationSpec(kind=KIND_SEMANTIC, allowed_labels=["A", "B"],
                                  strategy=STRATEGY_MODEL)
        fake_llm.batch_fn = lambda t: "A" if t != "zzz" else "B"
        idx = [42, 7, 99, 3]
        report = apply_model_backed(pd.Series(["x", "y", "zzz", "w"], index=idx),
                                    spec, [], llm_override=OVERRIDE)
        assert report.series.index.tolist() == idx
        assert report.series.tolist() == ["A", "A", "B", "A"]

    # ---------------------------------------------------------------- API -- #
    def test_ticket_end_to_end_preview_apply_revert(self, client, auth_headers,
                                                    fake_llm):
        df = ticket_frame()
        csv_bytes = df.to_csv(index=False).encode()

        up = client.post("/upload_file/", headers=auth_headers,
                         files={"file": ("tickets.csv", csv_bytes, "text/csv")})
        assert up.status_code == 200, up.text
        session_id = up.json()["session_id"]
        original_preview = up.json()["preview"]

        # 1. classify (mocked model decides family + task + labels)
        fake_llm.classify_response = SEMANTIC_CLASSIFY
        ex = [{"input": s, "output": t} for s, t in TICKET_EXAMPLES]
        r = client.post("/classify/", headers=auth_headers,
                        json={"column": "ticket_description", "examples": ex,
                              "llm_config": {"api_key": "test-personal-key"}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["classification"] == "General"
        assert body["task"] == "semantic"
        assert body["labels"] == TICKET_LABELS

        # 2. generate (task/labels reused -> planner picks the model strategy)
        r = client.post(
            "/generate_function/", headers=auth_headers,
            json={
                "column": "ticket_description", "examples": ex,
                "classification": body["classification"],
                "task": body["task"], "labels": body["labels"],
                "llm_config": {"api_key": "test-personal-key"},
            },
        )
        assert r.status_code == 200, r.text
        gen = r.json()
        assert gen["strategy"] == STRATEGY_MODEL
        assert gen["spec"]["allowed_labels"] == TICKET_LABELS
        # Model-backed code is HONEST and runnable (spec §11): it is the real
        # batching / row-id / validation routine the engine runs, not an empty
        # string or a comment that pretends a deterministic rule exists.
        code = gen["code"]
        assert code and "def transform" in code
        assert "row_id" in code and "results" in code   # row-id contract
        assert "model" in code.lower()                   # states it is model-backed
        compile(code, "<model_backed>", "exec")          # it is valid Python
        # ...and it must NOT pretend to be a deterministic lookup table.
        assert "lookup_table.get" not in code
        assert gen["family"] == "Semantic Classification"
        assert gen["transformation_id"]

        # 3. apply (mocked model runs over EVERY row)
        fake_llm.batch_fn = keyword_classifier
        r = client.post(
            "/apply/", headers=auth_headers,
            json={
                "transformation_id": gen["transformation_id"],
                "selected_column": "ticket_description",
                "source_type": "file", "session_id": session_id,
                "llm_config": {"api_key": "test-personal-key"},
            },
        )
        assert r.status_code == 200, r.text
        applied = r.json()
        assert applied["processed_rows"] == 8
        assert applied["transformed_rows"] == 8
        assert applied["failed_rows"] == 0
        assert applied["success_rate"] == 1.0
        # Enriched reporting (spec §15): family + provider + clean status.
        assert applied["family"] == "Semantic Classification"
        assert applied["status"] == "success"
        assert applied["review_rows"] == 0

        got = {row["ticket_id"]: row["ticket_description"]
               for row in applied["preview"]}
        assert got == TICKET_EXPECTED
        # unrelated columns unchanged
        before = {row["ticket_id"]: row for row in original_preview}
        for row in applied["preview"]:
            assert row["ticket_id"] == before[row["ticket_id"]]["ticket_id"]

        # 4. revert restores the original descriptions for every row
        r = client.post("/revert_session/", headers=auth_headers,
                        json={"session_id": session_id})
        assert r.status_code == 200, r.text
        reverted = {row["ticket_description"] for row in r.json()["preview"]}
        assert reverted == set(df["ticket_description"])

    def test_partial_model_failure_reported_not_marked_success(self, client,
                                                               auth_headers,
                                                               fake_llm):
        df = ticket_frame()
        csv_bytes = df.to_csv(index=False).encode()
        up = client.post("/upload_file/", headers=auth_headers,
                         files={"file": ("tickets.csv", csv_bytes, "text/csv")})
        session_id = up.json()["session_id"]

        fake_llm.classify_response = SEMANTIC_CLASSIFY
        ex = [{"input": s, "output": t} for s, t in TICKET_EXAMPLES]
        gen = client.post(
            "/generate_function/", headers=auth_headers,
            json={"column": "ticket_description", "examples": ex,
                  "classification": "General", "task": "semantic",
                  "labels": TICKET_LABELS,
                  "llm_config": {"api_key": "test-personal-key"}},
        ).json()

        # model returns null for one row -> that row fails, the rest succeed
        fake_llm.batch_fn = lambda t: None if "VPN" in t else keyword_classifier(t)
        r = client.post(
            "/apply/", headers=auth_headers,
            json={"transformation_id": gen["transformation_id"],
                  "selected_column": "ticket_description",
                  "source_type": "file", "session_id": session_id,
                  "llm_config": {"api_key": "test-personal-key"}},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["failed_rows"] == 1
        assert body["processed_rows"] == 8
        assert body["transformed_rows"] == 7
        assert 0 < body["success_rate"] < 1
        assert body["row_errors"]

    def test_deterministic_transformation_through_api(self, client, auth_headers):
        csv = "id,name\n1,ananya\n2,kiran\n3,meera\n"
        up = client.post("/upload_file/", headers=auth_headers,
                         files={"file": ("d.csv", csv.encode(), "text/csv")})
        assert up.status_code == 200
        session_id = up.json()["session_id"]

        ex = [{"input": "ananya", "output": "ANANYA"},
              {"input": "kiran", "output": "KIRAN"}]
        gen = client.post("/generate_function/", headers=auth_headers,
                          json={"column": "name", "examples": ex,
                                "classification": "String-based"}).json()
        assert gen["strategy"] == STRATEGY_DETERMINISTIC
        assert "def transform" in gen["code"]

        r = client.post("/apply/", headers=auth_headers,
                        json={"transformation_id": gen["transformation_id"],
                              "selected_column": "name",
                              "source_type": "file", "session_id": session_id})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["failed_rows"] == 0
        names = [row["name"] for row in body["preview"]]
        assert names == ["ANANYA", "KIRAN", "MEERA"]
        ids = [row["id"] for row in body["preview"]]
        assert ids == [1, 2, 3]   # row identity untouched

    def test_apply_rejects_unknown_transformation_id(self, client, auth_headers):
        csv = "id,name\n1,a\n"
        up = client.post("/upload_file/", headers=auth_headers,
                         files={"file": ("d.csv", csv.encode(), "text/csv")})
        session_id = up.json()["session_id"]
        r = client.post("/apply/", headers=auth_headers,
                        json={"transformation_id": "deadbeefdeadbeef",
                              "selected_column": "name",
                              "source_type": "file", "session_id": session_id})
        assert r.status_code == 404
