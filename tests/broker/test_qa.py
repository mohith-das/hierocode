"""Tests for hierocode.broker.qa."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hierocode.broker.plan_schema import PlanParseError, QAVerdict, TaskUnit
from hierocode.exceptions import ProviderConnectionError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _unit(**kwargs) -> TaskUnit:
    defaults = dict(id="u1", goal="Add null check", target_files=["src/foo.py"])
    defaults.update(kwargs)
    return TaskUnit(**defaults)


def _provider(return_value: str) -> MagicMock:
    """Return a mock BaseProvider whose generate() returns return_value."""
    provider = MagicMock()
    provider.generate.return_value = return_value
    return provider


def _accept_json() -> str:
    return json.dumps({"action": "accept"})


def _revise_json(feedback: str = "fix the null check") -> str:
    return json.dumps({"action": "revise", "feedback": feedback})


def _escalate_json(reason: str = "too complex") -> str:
    return json.dumps({"action": "escalate", "reason": reason})


def _split_json() -> str:
    sub = {"id": "s1", "goal": "sub-task", "target_files": ["src/foo.py"]}
    return json.dumps({"action": "split", "sub_units": [sub]})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReviewDraft:
    """Unit tests for review_draft()."""

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_calls_provider_with_json_mode(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = _provider(_accept_json())
        review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        provider.generate.assert_called_once()
        call_kwargs = provider.generate.call_args
        assert call_kwargs.kwargs.get("json_mode") is True or (
            len(call_kwargs.args) > 0 and False  # json_mode always passed as kwarg
        )
        # Verify json_mode=True was passed
        assert provider.generate.call_args.kwargs["json_mode"] is True
        # Verify model was passed correctly
        assert provider.generate.call_args.kwargs["model"] == "claude-sonnet-4-6"

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_returns_parsed_verdict_accept(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = _provider(_accept_json())
        verdict = review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        assert isinstance(verdict, QAVerdict)
        assert verdict.action == "accept"

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_returns_parsed_verdict_revise(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = _provider(_revise_json("fix the null check"))
        verdict = review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        assert verdict.action == "revise"
        assert verdict.feedback == "fix the null check"

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_returns_parsed_verdict_split(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = _provider(_split_json())
        verdict = review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        assert verdict.action == "split"
        assert verdict.sub_units is not None
        assert len(verdict.sub_units) == 1
        assert verdict.sub_units[0].id == "s1"

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_returns_parsed_verdict_escalate(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = _provider(_escalate_json("too complex"))
        verdict = review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        assert verdict.action == "escalate"
        assert verdict.reason == "too complex"

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_parse_error_wrapped_with_raw(self, mock_build):
        from hierocode.broker.qa import review_draft

        bad_raw = "not json at all"
        provider = _provider(bad_raw)

        with pytest.raises(PlanParseError) as exc_info:
            review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        assert "not json at all" in str(exc_info.value)

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_provider_error_propagates(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = MagicMock()
        provider.generate.side_effect = ProviderConnectionError("connection refused")

        with pytest.raises(ProviderConnectionError, match="connection refused"):
            review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

    def test_build_qa_prompt_called_with_expected_args(self):
        from hierocode.broker.qa import review_draft

        unit = _unit()
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y"
        test_out = "PASSED"
        original = "Add null check to foo"

        mock_prompt = MagicMock(return_value="built-prompt")

        with patch("hierocode.broker.qa.build_qa_prompt", mock_prompt):
            provider = _provider(_accept_json())
            review_draft(
                provider,
                "claude-sonnet-4-6",
                unit,
                diff,
                test_output=test_out,
                original_task=original,
            )

        mock_prompt.assert_called_once_with(unit, diff, test_out, original)

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_system_prompt_passed(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = _provider(_accept_json())
        review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        system_arg = provider.generate.call_args.kwargs.get("system", "")
        assert "code reviewer" in system_arg or "JSON" in system_arg

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_max_tokens_forwarded_to_provider(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = _provider(_accept_json())
        review_draft(
            provider, "claude-sonnet-4-6", _unit(), "diff text", max_tokens=512
        )

        assert provider.generate.call_args.kwargs["max_tokens"] == 512

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_parse_error_truncates_raw_at_1000_chars(self, mock_build):
        from hierocode.broker.qa import review_draft

        # Provider returns a very long invalid response
        long_raw = "X" * 2000
        provider = _provider(long_raw)

        with pytest.raises(PlanParseError) as exc_info:
            review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        msg = str(exc_info.value)
        # The raw embedded in the message must not exceed 1000 chars from the original
        assert "X" * 1000 in msg
        assert "X" * 1001 not in msg

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_parse_error_chains_original_exception(self, mock_build):
        from hierocode.broker.qa import review_draft

        provider = _provider("not json at all")

        with pytest.raises(PlanParseError) as exc_info:
            review_draft(provider, "claude-sonnet-4-6", _unit(), "diff text")

        # Exception must be chained (raise ... from exc)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, PlanParseError)

    @patch("hierocode.broker.qa.build_qa_prompt", return_value="qa-prompt-text")
    def test_review_draft_forwards_exploration_kwargs(self, mock_build):
        """review_draft must forward exploration and allowed_tools to provider.generate."""
        from hierocode.broker.qa import review_draft

        provider = _provider(_accept_json())
        review_draft(
            provider,
            "claude-sonnet-4-6",
            _unit(),
            "diff text",
            exploration="active",
            allowed_tools=["Read"],
        )

        kwargs = provider.generate.call_args.kwargs
        assert kwargs.get("exploration") == "active"
        assert kwargs.get("allowed_tools") == ["Read"]
