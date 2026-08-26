"""Tests for the AgenticDataBench scorer."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.scorer import Score

from agenticdatabench._scorer import agenticdatabench_scorer


def _make_state(
    sample_id: str,
    eval_funcs: list[str],
    output_names: list[str],
    gold_names: list[str],
):
    """Build a minimal TaskState mock."""
    state = MagicMock()
    state.sample_id = sample_id
    state.metadata = {
        "eval_func": eval_funcs,
        "output_file_names": output_names,
        "gold_file_names": gold_names,
    }
    return state


@pytest.fixture
def scorer_fn():
    return agenticdatabench_scorer()


class TestAgenticDataBenchScorer:
    @pytest.mark.asyncio
    async def test_perfect_score(self, scorer_fn, csv_pair):
        output, gold = csv_pair
        sb = AsyncMock()
        sb.read_file = AsyncMock(
            side_effect=lambda path, text=True: (
                output.read_text() if "workspace" in path else gold.read_text()
            )
        )

        state = _make_state(
            "agriculture_02",
            [
                "compare_csv(output_file_name='output.csv', gold_file_name='result.csv', "
                "ignore_order=True, specified_columns=['median_n'])"
            ],
            ["output.csv"],
            ["result.csv"],
        )

        with patch("agenticdatabench._scorer.sandbox", return_value=sb):
            result = await scorer_fn(state, MagicMock())

        assert isinstance(result, Score)
        assert result.value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_missing_output_scores_zero(self, scorer_fn, tmp_path):
        gold = tmp_path / "result.csv"
        gold.write_text("median_n\n0.25\n")

        sb = AsyncMock()

        async def fake_read(path, text=True):
            if "workspace" in path:
                raise FileNotFoundError(path)
            return gold.read_text()

        sb.read_file = fake_read

        state = _make_state(
            "agriculture_02",
            [
                "compare_csv(output_file_name='output.csv', gold_file_name='result.csv', "
                "ignore_order=True)"
            ],
            ["output.csv"],
            ["result.csv"],
        )

        with patch("agenticdatabench._scorer.sandbox", return_value=sb):
            result = await scorer_fn(state, MagicMock())

        assert isinstance(result, Score)
        assert result.value == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_no_eval_func_returns_unscored(self, scorer_fn):
        state = _make_state("test_id", [], [], [])
        result = await scorer_fn(state, MagicMock())
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_score_in_valid_range(self, scorer_fn, csv_pair_mismatch):
        output, gold = csv_pair_mismatch
        sb = AsyncMock()
        sb.read_file = AsyncMock(
            side_effect=lambda path, text=True: (
                output.read_text() if "workspace" in path else gold.read_text()
            )
        )

        state = _make_state(
            "agriculture_02",
            [
                "compare_csv(output_file_name='output.csv', gold_file_name='result.csv', "
                "ignore_order=True)"
            ],
            ["output.csv"],
            ["result.csv"],
        )

        with patch("agenticdatabench._scorer.sandbox", return_value=sb):
            result = await scorer_fn(state, MagicMock())

        assert isinstance(result, Score)
        assert 0.0 <= result.value <= 1.0
