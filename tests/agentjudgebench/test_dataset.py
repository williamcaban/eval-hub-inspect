"""Tests for AgentJudgeBench dataset loading."""

from __future__ import annotations

import pytest

from agentjudgebench._dataset import load_samples


class TestLoadSamples:
    def test_loads_all_samples(self, data_dir):
        samples = load_samples(split="test", data_dir=str(data_dir))
        assert len(samples) == 2

    def test_sample_has_required_fields(self, data_dir):
        samples = load_samples(split="test", data_dir=str(data_dir))
        s = samples[0]
        assert s.id == "ajb_linear_easy_001"
        assert "Workflow Task" in s.input
        assert "Agent Trajectory" in s.input
        assert "difficulty" in s.metadata
        assert "dag_topology" in s.metadata
        assert "label" in s.metadata
        assert "with_ground_truth" in s.metadata

    def test_target_reflects_label(self, data_dir, correct_record, incorrect_record):
        samples = load_samples(split="test", data_dir=str(data_dir))
        by_id = {str(s.id): s for s in samples}
        assert by_id[correct_record["id"]].target == "CORRECT"
        assert by_id[incorrect_record["id"]].target == "INCORRECT"

    def test_difficulty_filter(self, data_dir):
        samples = load_samples(split="test", difficulty="easy", data_dir=str(data_dir))
        assert len(samples) == 1
        assert samples[0].metadata["difficulty"] == "easy"

    def test_dag_topology_filter(self, data_dir):
        samples = load_samples(split="test", dag_topology="diamond", data_dir=str(data_dir))
        assert len(samples) == 1
        assert samples[0].metadata["dag_topology"] == "diamond"

    def test_max_samples(self, data_dir):
        samples = load_samples(split="test", max_samples=1, data_dir=str(data_dir))
        assert len(samples) == 1

    def test_without_ground_truth_omits_reference(self, data_dir):
        samples = load_samples(split="test", with_ground_truth=False, data_dir=str(data_dir))
        assert "Reference Answer" not in samples[0].input
        assert samples[0].metadata["with_ground_truth"] is False

    def test_with_ground_truth_includes_reference(self, data_dir):
        samples = load_samples(split="test", with_ground_truth=True, data_dir=str(data_dir))
        assert "Reference Answer" in samples[0].input
        assert samples[0].metadata["with_ground_truth"] is True

    def test_trajectory_steps_appear_in_prompt(self, data_dir):
        samples = load_samples(split="test", data_dir=str(data_dir))
        # Each trajectory step should render its tool name in the prompt
        assert "crm_get_user" in samples[0].input
        assert "Step 1" in samples[0].input

    def test_invalid_difficulty_raises(self, data_dir):
        with pytest.raises(ValueError, match="difficulty must be one of"):
            load_samples(split="test", difficulty="extreme", data_dir=str(data_dir))

    def test_unknown_difficulty_returns_empty(self, data_dir):
        samples = load_samples(split="test", difficulty="medium", data_dir=str(data_dir))
        assert samples == []

    def test_unknown_topology_returns_empty(self, data_dir):
        samples = load_samples(split="test", dag_topology="grid", data_dir=str(data_dir))
        assert samples == []
