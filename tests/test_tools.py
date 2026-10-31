"""Regression tests for the HR tools. No API key needed.

Run from the project folder:  python -m pytest -v
"""
import pytest

from hr_tools import get_claim_status, search_policies


@pytest.mark.parametrize("claim_input, expected_status", [
    ("CLM-10023", "approved"),
    ("clm-10789", "denied"),
    ("claim 10456", "pending"),
    ("CLM 10023", "approved"),
])
def test_claim_status_accepts_common_formats(claim_input, expected_status):
    result = get_claim_status(claim_input)
    assert result["status"] == expected_status


def test_unknown_claim_returns_error_not_crash():
    result = get_claim_status("CLM-99999")
    assert "error" in result


@pytest.mark.parametrize("question, expected_policy", [
    ("How many PTO days do full-time employees get?", "P1"),
    ("What is the deadline to file a dental claim?", "P6"),
    ("What is the daily meal limit when traveling?", "P7"),
    ("How long do I have to appeal a denied claim?", "P8"),
])
def test_search_ranks_right_policy_first(question, expected_policy):
    results = search_policies(question)
    assert results[0]["id"] == expected_policy


def test_search_returns_nothing_for_unrelated_question():
    assert search_policies("Who won the Super Bowl last year?") == []
