"""Tests for the AgentJudgeBench @task function."""

from __future__ import annotations

from inspect_ai import Task

from agentjudgebench import agentjudgebench


class TestAgentJudgeBenchTask:
    def test_returns_task(self, data_dir):
        t = agentjudgebench(data_dir=str(data_dir))
        assert isinstance(t, Task)

    def test_task_has_dataset(self, data_dir):
        t = agentjudgebench(data_dir=str(data_dir))
        assert len(t.dataset) == 2

    def test_task_has_scorer(self, data_dir):
        t = agentjudgebench(data_dir=str(data_dir))
        assert t.scorer is not None

    def test_no_sandbox(self, data_dir):
        t = agentjudgebench(data_dir=str(data_dir))
        assert t.sandbox is None

    def test_difficulty_filter(self, data_dir):
        t = agentjudgebench(difficulty="easy", data_dir=str(data_dir))
        assert len(t.dataset) == 1

    def test_dag_topology_filter(self, data_dir):
        t = agentjudgebench(dag_topology="diamond", data_dir=str(data_dir))
        assert len(t.dataset) == 1

    def test_max_samples(self, data_dir):
        t = agentjudgebench(max_samples=1, data_dir=str(data_dir))
        assert len(t.dataset) == 1

    def test_with_ground_truth_sets_metadata(self, data_dir):
        t = agentjudgebench(with_ground_truth=True, data_dir=str(data_dir))
        assert all(s.metadata["with_ground_truth"] is True for s in t.dataset)

    def test_without_ground_truth_is_default(self, data_dir):
        t = agentjudgebench(data_dir=str(data_dir))
        assert all(s.metadata["with_ground_truth"] is False for s in t.dataset)
