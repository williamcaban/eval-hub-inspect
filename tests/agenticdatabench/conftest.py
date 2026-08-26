"""Shared fixtures for AgenticDataBench tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_task() -> dict:
    return {
        "id": "agriculture_02",
        "question": "Calculate the median nitrogen value across wheat samples.",
        "data_sources": ["Crop Recommendation.csv"],
        "skills": ["Data Loading with Pandas", "Statistical Calculations"],
        "domain": "agriculture",
        "output_file_name": ["output.csv"],
        "gold_file_name": ["result.csv"],
        "eval_func": [
            "compare_csv(output_file_name='output.csv', gold_file_name='result.csv', "
            "ignore_order=True, specified_columns=['median_n'])"
        ],
    }


@pytest.fixture
def sample_tasks(sample_task) -> list[dict]:
    task2 = dict(sample_task)
    task2["id"] = "finance_01"
    task2["domain"] = "finance"
    return [sample_task, task2]


@pytest.fixture
def tasks_jsonl(tmp_path, sample_tasks) -> Path:
    """Write sample tasks to a temp jsonl file in the expected directory layout."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    p = tasks_dir / "dev.jsonl"
    p.write_text("\n".join(json.dumps(t) for t in sample_tasks))
    return p


@pytest.fixture
def csv_pair(tmp_path):
    """Write matching gold and output CSV files."""
    gold = tmp_path / "result.csv"
    output = tmp_path / "output.csv"
    gold.write_text("median_n\n0.25\n0.30\n")
    output.write_text("median_n\n0.25\n0.30\n")
    return output, gold


@pytest.fixture
def csv_pair_mismatch(tmp_path):
    """Write mismatching gold and output CSV files."""
    gold = tmp_path / "result.csv"
    output = tmp_path / "output.csv"
    gold.write_text("median_n\n0.25\n0.30\n")
    output.write_text("median_n\n0.99\n0.99\n")
    return output, gold
