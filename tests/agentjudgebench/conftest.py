"""Shared fixtures for AgentJudgeBench tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def correct_record() -> dict:
    """A record where the agent trajectory is correct (label=True)."""
    return {
        "id": "ajb_linear_easy_001",
        "workflow": "Retrieve user profile, then fetch their orders, then compute total spend.",
        "trajectory": [
            {
                "tool": "crm_get_user",
                "args": {"user_id": "U42"},
                "result": {"name": "Alice", "tier": "premium"},
            },
            {
                "tool": "orders_get_recent",
                "args": {"user_id": "U42", "limit": 10},
                "result": {"orders": [{"amount": 150.0}, {"amount": 300.0}]},
            },
            {
                "tool": "calculate_total",
                "args": {"amounts": [150.0, 300.0]},
                "result": {"total": 450.0},
            },
        ],
        "difficulty": "easy",
        "dag_topology": "linear",
        "label": True,
        "ground_truth": "All three tools were called in the correct order with correct arguments.",
        "generator_model": "test-model-3b",
    }


@pytest.fixture
def incorrect_record() -> dict:
    """A record where the agent trajectory is incorrect (label=False)."""
    return {
        "id": "ajb_diamond_hard_099",
        "workflow": "Fetch product details and inventory simultaneously, then compute reorder qty.",
        "trajectory": [
            {
                "tool": "product_lookup",
                "args": {"sku": "P9"},
                "result": {"name": "Widget", "reorder_point": 50},
            },
            # Missing: inventory_check step — agent skipped a required branch
            {
                "tool": "compute_reorder",
                "args": {"current_stock": 0, "reorder_point": 50},
                "result": {"reorder_qty": 50},
            },
        ],
        "difficulty": "hard",
        "dag_topology": "diamond",
        "label": False,
        "ground_truth": "The agent skipped the required inventory_check step.",
        "generator_model": "test-model-70b",
    }


@pytest.fixture
def sample_records(correct_record, incorrect_record) -> list[dict]:
    return [correct_record, incorrect_record]


@pytest.fixture
def data_dir(tmp_path, sample_records) -> Path:
    """Write sample records to a temp directory in the expected layout."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    jsonl = data / "test.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in sample_records), encoding="utf-8")
    return tmp_path
