"""Tests for AgentJudgeBench alignment scorer."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from inspect_ai.scorer import Score

from agentjudgebench._scorer import _extract_judgment, agentjudgebench_scorer


class TestExtractJudgment:
    """Unit tests for the CORRECT/INCORRECT parser."""

    def test_correct_first_word(self):
        assert _extract_judgment("CORRECT — all tools called in sequence.") is True

    def test_incorrect_first_word(self):
        assert _extract_judgment("INCORRECT — step 2 used wrong parameter.") is False

    def test_incorrect_takes_priority_over_correct(self):
        # "INCORRECT" contains "CORRECT" as substring; INCORRECT must win.
        assert _extract_judgment("INCORRECT — the agent was not correct.") is False

    def test_lowercase_normalized(self):
        assert _extract_judgment("correct, the agent completed the task.") is True
        assert _extract_judgment("incorrect — a tool was skipped.") is False

    def test_unparseable_returns_none(self):
        assert _extract_judgment("I cannot determine this.") is None
        assert _extract_judgment("") is None
        assert _extract_judgment("The evaluation is...") is None

    def test_correct_in_preamble_fallback(self):
        # Model adds preamble before the verdict word.
        assert (
            _extract_judgment("Based on my analysis: CORRECT. The agent followed all steps.")
            is True
        )


def _make_state(completion: str, label: bool, difficulty: str = "easy") -> MagicMock:
    """Build a minimal TaskState mock."""
    state = MagicMock()
    output = MagicMock()
    output.completion = completion
    state.output = output
    state.metadata = {
        "label": label,
        "difficulty": difficulty,
        "dag_topology": "linear",
        "with_ground_truth": False,
    }
    return state


@pytest.fixture
def scorer_fn():
    return agentjudgebench_scorer()


class TestAgentJudgeBenchScorer:
    @pytest.mark.asyncio
    async def test_correct_judgment_correct_label_scores_one(self, scorer_fn):
        state = _make_state("CORRECT — all steps matched.", label=True)
        result = await scorer_fn(state, MagicMock())
        assert isinstance(result, Score)
        assert result.value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_incorrect_judgment_incorrect_label_scores_one(self, scorer_fn):
        state = _make_state("INCORRECT — agent skipped step 2.", label=False)
        result = await scorer_fn(state, MagicMock())
        assert result.value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_correct_judgment_incorrect_label_scores_zero(self, scorer_fn):
        state = _make_state("CORRECT — everything looks fine.", label=False)
        result = await scorer_fn(state, MagicMock())
        assert result.value == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_incorrect_judgment_correct_label_scores_zero(self, scorer_fn):
        state = _make_state("INCORRECT — wrong tool order.", label=True)
        result = await scorer_fn(state, MagicMock())
        assert result.value == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_unparseable_output_is_unscored(self, scorer_fn):
        state = _make_state("I'm not sure about this trajectory.", label=True)
        result = await scorer_fn(state, MagicMock())
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_metadata_included_in_result(self, scorer_fn):
        state = _make_state("CORRECT — good.", label=True, difficulty="hard")
        result = await scorer_fn(state, MagicMock())
        assert result.metadata["difficulty"] == "hard"
        assert result.metadata["aligned"] is True
        assert result.metadata["judgment"] is True
        assert result.metadata["ground_truth"] is True

    @pytest.mark.asyncio
    async def test_score_value_is_binary(self, scorer_fn):
        for completion, label in [
            ("CORRECT", True),
            ("CORRECT", False),
            ("INCORRECT", True),
            ("INCORRECT", False),
        ]:
            state = _make_state(completion, label)
            result = await scorer_fn(state, MagicMock())
            assert result.value in (0.0, 1.0)
