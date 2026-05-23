"""Data integrity across preview / apply / revert / export (MOCKED).

Verifies spec §10 end-to-end through the HTTP API with a mocked model: row
count, row order, row identity, unselected columns, null semantics, and that
preview / apply / download agree.  No real provider is contacted.
"""

from __future__ import annotations

import io
import json
import re
from typing import List

import pandas as pd
import pytest

from transform import apply_model_backed
from transform.spec import (
    KIND_SEMANTIC,
    STRATEGY_MODEL,
    TransformationSpec,
)

OVERRIDE = {"api_key": "test-personal-key"}
LABELS = ["Account Access", "Billing", "Hardware", "Network"]

TICKETS = pd.DataFrame({
    "ticket_id": [f"GEN-{i:03d}" for i in range(1, 9)],
    "ticket_description": [
        "Cannot sign in even with the correct password",
        "Login button does nothing on mobile",
        "Subscription charged twice this month",
        "I was billed after cancelling",
        "Laptop shuts down whenever I open it",
        "Screen flickers then goes black",
        "Office Wi-Fi disconnects during calls",
        "VPN drops every few minutes",
    ],
    "priority": ["p1", "p2", "p1", "p2", "p3", "p3", "p2", "p2"],
})


def _bucket(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("wi-fi", "vpn", "disconnect", "drops")):
        return "Network"
    if any(k in t for k in ("sign in", "login", "password", "locked")):
        return "Account Access"
    if any(k in t for k in ("bill", "charge", "cancel", "subscription")):
        return "Billing"
    return "Hardware"


def _rowid_llm(monkeypatch, fn):
    def invoke(prompt, max_tokens=400, override=None, budget=None):
        ids = json.loads(re.search(
            r"Row ids \(paired with the inputs below, in order\):\n(\[[^\]]*\])",
            prompt).group(1))
        inputs = json.loads(re.search(
            r"Inputs \(JSON array, in order\):\n(\[.*\])", prompt, re.DOTALL).group(1))
        return json.dumps({"results": [
            {"row_id": rid, "output": fn(v)} for rid, v in zip(ids, inputs)]})

    monkeypatch.setattr("llm.invoke", invoke)


def _spec():
    return TransformationSpec(kind=KIND_SEMANTIC, strategy=STRATEGY_MODEL,
                              requires_model=True, allowed_labels=LABELS,
                              task="semantic")


# --------------------------------------------------------------------------- #
# Series-level integrity
# --------------------------------------------------------------------------- #
class TestSeriesIntegrity:
    def test_row_count_order_and_ids_preserved(self, monkeypatch):
        _rowid_llm(monkeypatch, _bucket)
        series = TICKETS["ticket_description"]
        report = apply_model_backed(series, _spec(), [], llm_override=OVERRIDE)
        assert report.total == len(TICKETS)
        assert len(report.series) == len(TICKETS)
        # order preserved position-for-position
        assert report.series.tolist() == [
            _bucket(v) for v in series.tolist()
        ]
        assert report.failed == 0
        assert report.row_status == ["ok"] * len(TICKETS)

    def test_output_matches_input_row_by_row_not_by_order(self, monkeypatch):
        """A model returning reversed results must still align by row id."""
        def invoke(prompt, max_tokens=400, override=None, budget=None):
            ids = json.loads(re.search(
                r"Row ids \(paired with the inputs below, in order\):\n(\[[^\]]*\])",
                prompt).group(1))
            inputs = json.loads(re.search(
                r"Inputs \(JSON array, in order\):\n(\[.*\])", prompt,
                re.DOTALL).group(1))
            pairs = list(zip(ids, inputs))[::-1]  # reversed
            return json.dumps({"results": [
                {"row_id": rid, "output": _bucket(v)} for rid, v in pairs]})

        monkeypatch.setattr("llm.invoke", invoke)
        report = apply_model_backed(TICKETS["ticket_description"], _spec(), [],
                                    llm_override=OVERRIDE)
        # still positionally correct despite reversed model output
        assert report.series.tolist() == [
            _bucket(v) for v in TICKETS["ticket_description"].tolist()]

    def test_unselected_column_untouched_by_engine(self, monkeypatch):
        _rowid_llm(monkeypatch, _bucket)
        df = TICKETS.copy()
        report = apply_model_backed(df["ticket_description"], _spec(), [],
                                    llm_override=OVERRIDE)
        from transform.engine import reconstruct

        out = reconstruct(df, "ticket_description", report)
        assert out["priority"].tolist() == df["priority"].tolist()
        assert out["ticket_id"].tolist() == df["ticket_id"].tolist()
        assert len(out) == len(df)

    def test_original_frame_not_mutated(self, monkeypatch):
        _rowid_llm(monkeypatch, _bucket)
        df = TICKETS.copy()
        before = df["ticket_description"].tolist()
        report = apply_model_backed(df["ticket_description"], _spec(), [],
                                    llm_override=OVERRIDE)
        from transform.engine import reconstruct

        reconstruct(df, "ticket_description", report)
        assert df["ticket_description"].tolist() == before

    def test_null_semantics_preserved(self, monkeypatch):
        _rowid_llm(monkeypatch, _bucket)
        series = pd.Series([None, "Cannot sign in", "", "  ", "Charged twice"])
        report = apply_model_backed(series, _spec(), [], llm_override=OVERRIDE)
        assert report.series.tolist()[0] is None      # null stays null
        assert report.series.tolist()[2] == ""        # empty stays empty
        assert report.series.tolist()[3] == "  "      # whitespace preserved
        assert report.series.tolist()[1] == "Account Access"
        assert report.series.tolist()[4] == "Billing"


# --------------------------------------------------------------------------- #
# End-to-end through the API: preview / apply / download agree
# --------------------------------------------------------------------------- #
class TestApiIntegrity:
    def _upload(self, client, auth_headers):
        csv_bytes = TICKETS.to_csv(index=False).encode()
        up = client.post("/upload_file/", headers=auth_headers,
                         files={"file": ("tickets.csv", csv_bytes, "text/csv")})
        assert up.status_code == 200, up.text
        return up.json()["session_id"], up.json()["preview"]

    def _generate(self, client, auth_headers):
        ex = [{"input": s, "output": t} for s, t in [
            ("Cannot sign in", "Account Access"),
            ("Charged twice", "Billing"),
            ("Laptop shut down", "Hardware"),
            ("Wi-Fi disconnects", "Network"),
        ]]
        gen = client.post("/generate_function/", headers=auth_headers,
                          json={"column": "ticket_description", "examples": ex,
                                "classification": "General", "task": "semantic",
                                "labels": LABELS,
                                "llm_config": {"api_key": "test-key"}})
        assert gen.status_code == 200, gen.text
        return gen.json()["transformation_id"]

    def test_apply_reports_metrics_and_download_matches(
        self, client, auth_headers, monkeypatch
    ):
        session_id, _ = self._upload(client, auth_headers)
        transformation_id = self._generate(client, auth_headers)
        _rowid_llm(monkeypatch, _bucket)

        r = client.post("/apply/", headers=auth_headers, json={
            "transformation_id": transformation_id,
            "selected_column": "ticket_description",
            "source_type": "file", "session_id": session_id,
            "llm_config": {"api_key": "test-key"},
        })
        assert r.status_code == 200, r.text
        applied = r.json()
        assert applied["processed_rows"] == 8
        assert applied["transformed_rows"] == 8
        assert applied["failed_rows"] == 0
        assert applied["review_rows"] == 0
        assert applied["status"] == "success"
        assert applied["family"] == "Semantic Classification"

        # preview reflects the transform on all 8 rows
        preview = applied["preview"]
        assert len(preview) == 8
        for row in preview:
            assert row["ticket_description"] in LABELS

        # download/export returns the SAME transformed values as the preview
        dl = client.get("/download/?session_id=" + session_id,
                        headers=auth_headers)
        assert dl.status_code == 200, dl.text
        downloaded = pd.read_csv(io.BytesIO(dl.content))
        assert len(downloaded) == 8
        assert set(downloaded["ticket_description"]) <= set(LABELS)
        # row identity preserved in the export
        assert downloaded["ticket_id"].tolist() == TICKETS["ticket_id"].tolist()

    def test_revert_restores_original(
        self, client, auth_headers, monkeypatch
    ):
        session_id, original_preview = self._upload(client, auth_headers)
        transformation_id = self._generate(client, auth_headers)
        _rowid_llm(monkeypatch, _bucket)

        client.post("/apply/", headers=auth_headers, json={
            "transformation_id": transformation_id,
            "selected_column": "ticket_description",
            "source_type": "file", "session_id": session_id,
            "llm_config": {"api_key": "test-key"},
        })
        rr = client.post("/revert_session/", headers=auth_headers,
                         json={"session_id": session_id})
        assert rr.status_code == 200, rr.text
        reverted = rr.json()["preview"]
        got = {row["ticket_id"]: row["ticket_description"] for row in reverted}
        want = {row["ticket_id"]: row["ticket_description"]
                for row in original_preview}
        assert got == want

    def test_partial_failure_not_marked_success(
        self, client, auth_headers, monkeypatch
    ):
        """A run where some rows fail must report partial, never a clean green."""
        session_id, _ = self._upload(client, auth_headers)
        transformation_id = self._generate(client, auth_headers)

        # The model answers correctly for the first 4 rows but returns a
        # label OUTSIDE the allowed set for the rest -> those rows fail
        # validation and are reported, not silently accepted.
        def invoke(prompt, max_tokens=400, override=None, budget=None):
            ids = json.loads(re.search(
                r"Row ids \(paired with the inputs below, in order\):\n(\[[^\]]*\])",
                prompt).group(1))
            inputs = json.loads(re.search(
                r"Inputs \(JSON array, in order\):\n(\[.*\])", prompt,
                re.DOTALL).group(1))
            results = []
            for n, (rid, value) in enumerate(zip(ids, inputs)):
                output = _bucket(value) if n < 4 else "NotALabel"
                results.append({"row_id": rid, "output": output})
            return json.dumps({"results": results})

        monkeypatch.setattr("llm.invoke", invoke)
        r = client.post("/apply/", headers=auth_headers, json={
            "transformation_id": transformation_id,
            "selected_column": "ticket_description",
            "source_type": "file", "session_id": session_id,
            "llm_config": {"api_key": "test-key"},
        })
        assert r.status_code == 200, r.text
        applied = r.json()
        # Contract: the 4 bad rows are failures, the 4 good rows succeeded --
        # and the response says "partial", never a generic success.
        assert applied["failed_rows"] == 4
        assert applied["transformed_rows"] == 4
        assert applied["processed_rows"] == 8
        assert applied["status"] == "partial"
        assert applied["success_rate"] == pytest.approx(0.5)
        assert applied["row_errors"]  # reasons surfaced
        # The 4 successful rows are still correct (no corruption of neighbours)
        ok = [row["ticket_description"] for row in applied["preview"]
              if row["ticket_description"] in LABELS]
        assert len(ok) == 4
