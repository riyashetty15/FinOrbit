import os

from backend.server import _should_advisor_handover


def test_advisor_handover_disabled(monkeypatch):
    monkeypatch.setenv("ADVISOR_HANDOVER_ENABLED", "false")
    monkeypatch.setenv("ADVISOR_HANDOVER_CONFIDENCE_THRESHOLD", "0.90")
    assert _should_advisor_handover(overall_score=0.1, recommended_action="serve") is False


def test_advisor_handover_triggers_below_threshold(monkeypatch):
    monkeypatch.setenv("ADVISOR_HANDOVER_ENABLED", "true")
    monkeypatch.setenv("ADVISOR_HANDOVER_CONFIDENCE_THRESHOLD", "0.80")
    assert _should_advisor_handover(overall_score=0.79, recommended_action="serve") is True
    assert _should_advisor_handover(overall_score=0.80, recommended_action="serve") is False


def test_advisor_handover_triggers_on_partial(monkeypatch):
    monkeypatch.setenv("ADVISOR_HANDOVER_ENABLED", "true")
    monkeypatch.setenv("ADVISOR_HANDOVER_CONFIDENCE_THRESHOLD", "0.10")
    assert _should_advisor_handover(overall_score=0.99, recommended_action="partial") is True


def test_advisor_handover_triggers_on_refuse(monkeypatch):
    monkeypatch.setenv("ADVISOR_HANDOVER_ENABLED", "true")
    monkeypatch.setenv("ADVISOR_HANDOVER_CONFIDENCE_THRESHOLD", "0.10")
    assert _should_advisor_handover(overall_score=0.99, recommended_action="refuse") is True


def test_advisor_handover_bad_threshold_falls_back(monkeypatch):
    monkeypatch.setenv("ADVISOR_HANDOVER_ENABLED", "true")
    monkeypatch.setenv("ADVISOR_HANDOVER_CONFIDENCE_THRESHOLD", "not-a-float")
    assert _should_advisor_handover(overall_score=0.49, recommended_action="serve") is True
    assert _should_advisor_handover(overall_score=0.50, recommended_action="serve") is False
