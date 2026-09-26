"""Hardening tests for the jev router: calibration, auditability, adversarial inputs.

All availability probes are stubbed — these tests never touch the network,
and they fail loudly (no swallowed exceptions) if routing is broken.
"""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jev.router as router  # noqa: E402
from jev.router import (  # noqa: E402
    COMPLEXITY_KEYWORDS,
    ESCALATE_THRESHOLD,
    estimate_complexity,
    route,
    route_with_decision,
)


@pytest.fixture
def local_only(monkeypatch):
    monkeypatch.setattr(router, "_ollama_alive", lambda url, timeout=1.5: True)
    monkeypatch.delenv("JEV_MODEL_TOKEN", raising=False)


@pytest.fixture
def both_up(monkeypatch):
    monkeypatch.setattr(router, "_ollama_alive", lambda url, timeout=1.5: True)
    monkeypatch.setenv("JEV_MODEL_TOKEN", "test-token")


@pytest.fixture
def nothing_up(monkeypatch):
    monkeypatch.setattr(router, "_ollama_alive", lambda url, timeout=1.5: False)
    monkeypatch.delenv("JEV_MODEL_TOKEN", raising=False)
    monkeypatch.delenv("JEV_DECISION_LOG", raising=False)


def test_complexity_orders_tasks():
    assert estimate_complexity("hi") < estimate_complexity(
        "prove the distributed refactor is free of concurrency bugs")


def test_threshold_is_a_named_constant_not_a_literal():
    # The escalation boundary must be reviewable in one place (ADR-0009:
    # a mis-tuned router silently degrades quality).
    assert isinstance(ESCALATE_THRESHOLD, float)
    assert 0.0 < ESCALATE_THRESHOLD < 1.0
    assert isinstance(COMPLEXITY_KEYWORDS, tuple) and COMPLEXITY_KEYWORDS


def test_decision_records_the_threshold_applied(both_up):
    _, d = route_with_decision("summarize this log", budget="balanced")
    assert d.threshold == ESCALATE_THRESHOLD
    assert d.chosen == "ollama-local"
    assert any("kept local" in r for r in d.reasons)


def test_keyword_stuffing_escalates_and_is_visible_in_reasons(both_up):
    # Adversarial: a trivial task padded with escalation keywords games the
    # heuristic into choosing the costlier route. The behavior is pinned here
    # so it cannot change silently; the decision record exposes *why* it
    # escalated, which is what an operator needs to spot abuse.
    task = "remind me to buy milk " + " ".join(COMPLEXITY_KEYWORDS)
    _, d = route_with_decision(task, budget="balanced")
    assert d.complexity >= d.threshold
    assert d.chosen == "github-models"
    assert any("escalated" in r for r in d.reasons)


def test_free_budget_never_escalates(both_up):
    hard = "prove the distributed refactor is free of concurrency bugs " * 20
    assert estimate_complexity(hard) >= ESCALATE_THRESHOLD
    r, d = route_with_decision(hard, budget="free")
    assert r.name == "ollama-local"
    assert any("budget=free" in reason for reason in d.reasons)


def test_kept_local_decision_logged(local_only, tmp_path):
    log = tmp_path / "decisions.jsonl"
    r, d = route_with_decision("summarize this log", log_path=str(log))
    assert r.name == "ollama-local"
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["chosen"] == "ollama-local"
    assert logged["complexity"] < logged["threshold"]
    assert logged["task_sha256"] == d.task_sha256


def test_task_text_never_written_to_ledger(local_only, tmp_path):
    task = "hunter2 my secret password reset request"
    log = tmp_path / "decisions.jsonl"
    route_with_decision(task, log_path=str(log))
    raw = log.read_text(encoding="utf-8")
    assert task not in raw
    assert "hunter2" not in raw
    logged = json.loads(raw.strip())
    assert logged["task_sha256"] == hashlib.sha256(task.encode()).hexdigest()
    assert logged["task_len"] == len(task)


def test_no_route_raises_loudly_and_still_logs(nothing_up, tmp_path):
    # Regression: the old smoke test swallowed RuntimeError and returned,
    # passing vacuously with a dead router. This must fail loudly.
    log = tmp_path / "decisions.jsonl"
    with pytest.raises(RuntimeError, match="no model route available"):
        route("summarize this log", log_path=str(log))
    logged = json.loads(log.read_text(encoding="utf-8").strip())
    assert logged["chosen"] == "none"
    assert logged["local_available"] is False
