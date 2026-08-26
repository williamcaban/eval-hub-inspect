"""Tests for the AgenticDataBench @task function."""

from __future__ import annotations

from unittest.mock import patch

from inspect_ai import Task

from agenticdatabench import agenticdatabench
from agenticdatabench._dataset import load_samples


class TestAgenticDataBenchTask:
    def test_returns_task(self, tasks_jsonl):
        data_dir = str(tasks_jsonl.parent.parent)
        with patch("agenticdatabench.task.load_samples") as mock_load:
            mock_load.return_value = load_samples(split="dev", data_dir=data_dir)
            t = agenticdatabench(data_dir=data_dir)
        assert isinstance(t, Task)

    def test_task_has_dataset(self, tasks_jsonl):
        t = agenticdatabench(data_dir=str(tasks_jsonl.parent.parent))
        assert len(t.dataset) == 2

    def test_task_has_sandbox(self, tasks_jsonl):
        t = agenticdatabench(data_dir=str(tasks_jsonl.parent.parent))
        assert t.sandbox is not None

    def test_task_has_scorer(self, tasks_jsonl):
        t = agenticdatabench(data_dir=str(tasks_jsonl.parent.parent))
        assert t.scorer is not None

    def test_domain_filter(self, tasks_jsonl):
        t = agenticdatabench(domain="agriculture", data_dir=str(tasks_jsonl.parent.parent))
        assert len(t.dataset) == 1

    def test_max_messages_set(self, tasks_jsonl):
        t = agenticdatabench(data_dir=str(tasks_jsonl.parent.parent))
        # inspect_ai >=0.3.x stores max_messages as message_limit on the Task object
        assert t.message_limit == 30
