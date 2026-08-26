"""Unit tests for the ported metric functions."""

from __future__ import annotations

import json

import pytest

from agenticdatabench.metrics import (
    compare_csv,
    compare_json,
    compare_json_normalized,
    compare_text,
)


class TestCompareCsv:
    def test_exact_match(self, csv_pair):
        output, gold = csv_pair
        result = compare_csv(
            str(output),
            str(gold),
            ignore_order=True,
            specified_columns=["median_n"],
        )
        assert result["score"] == pytest.approx(1.0)
        assert result["errors"] == []

    def test_mismatch(self, csv_pair_mismatch):
        output, gold = csv_pair_mismatch
        result = compare_csv(
            str(output),
            str(gold),
            ignore_order=True,
            specified_columns=["median_n"],
        )
        assert result["score"] == pytest.approx(0.0)
        assert len(result["errors"]) > 0

    def test_missing_output(self, tmp_path):
        gold = tmp_path / "result.csv"
        gold.write_text("col\n1\n")
        result = compare_csv(str(tmp_path / "missing.csv"), str(gold))
        assert result["score"] == 0
        assert "does not exist" in result["errors"][0]

    def test_tolerance(self, tmp_path):
        gold = tmp_path / "g.csv"
        output = tmp_path / "o.csv"
        gold.write_text("val\n1.0\n")
        output.write_text("val\n1.005\n")  # within 1% absolute tolerance
        result = compare_csv(str(output), str(gold), ignore_order=True)
        assert result["score"] == pytest.approx(1.0)


class TestCompareText:
    def test_exact_match(self, tmp_path):
        gold = tmp_path / "gold.txt"
        output = tmp_path / "out.txt"
        gold.write_text("Hello World")
        output.write_text("hello world")  # case insensitive
        result = compare_text(str(output), str(gold))
        assert result["score"] == pytest.approx(1.0)

    def test_mismatch(self, tmp_path):
        gold = tmp_path / "gold.txt"
        output = tmp_path / "out.txt"
        gold.write_text("expected answer")
        output.write_text("wrong answer")
        result = compare_text(str(output), str(gold))
        assert result["score"] == pytest.approx(0.0)

    def test_missing_output(self, tmp_path):
        gold = tmp_path / "gold.txt"
        gold.write_text("something")
        result = compare_text(str(tmp_path / "missing.txt"), str(gold))
        assert result["score"] == pytest.approx(0.0)


class TestCompareJson:
    def test_exact_match(self, tmp_path):
        data = {"mean": 0.5, "std": 0.1}
        gold = tmp_path / "gold.json"
        output = tmp_path / "out.json"
        gold.write_text(json.dumps(data))
        output.write_text(json.dumps(data))
        result = compare_json(str(output), str(gold), thresholds={"mean": 0.01, "std": 0.01})
        assert result["score"] == pytest.approx(1.0)

    def test_within_tolerance(self, tmp_path):
        gold = tmp_path / "gold.json"
        output = tmp_path / "out.json"
        gold.write_text(json.dumps({"mean": 1.0}))
        output.write_text(json.dumps({"mean": 1.005}))  # 0.5% error, within 1%
        result = compare_json(str(output), str(gold), thresholds={"mean": None})
        assert result["score"] == pytest.approx(1.0)

    def test_missing_file(self, tmp_path):
        gold = tmp_path / "gold.json"
        gold.write_text("{}")
        result = compare_json(str(tmp_path / "missing.json"), str(gold))
        assert result["score"] == pytest.approx(0.0)


class TestCompareJsonNormalized:
    def test_range_higher_better(self, tmp_path):
        output = tmp_path / "out.json"
        output.write_text(json.dumps({"auc": 0.9}))
        result = compare_json_normalized(str(output), thresholds={"auc": [0.5, 1.0]})
        assert result["score"] == pytest.approx(0.8, abs=0.01)

    def test_range_lower_better(self, tmp_path):
        output = tmp_path / "out.json"
        output.write_text(json.dumps({"error": 0.1}))
        result = compare_json_normalized(str(output), thresholds={"error": [1.0, 0.0]})
        assert result["score"] == pytest.approx(0.9, abs=0.01)
