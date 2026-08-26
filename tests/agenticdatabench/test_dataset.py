"""Tests for AgenticDataBench dataset loading."""

from __future__ import annotations

from agenticdatabench._dataset import load_samples


class TestLoadSamples:
    def test_loads_all_samples(self, tasks_jsonl):
        samples = load_samples(split="dev", data_dir=str(tasks_jsonl.parent.parent))
        assert len(samples) == 2

    def test_domain_filter(self, tasks_jsonl):
        samples = load_samples(
            split="dev",
            domain="agriculture",
            data_dir=str(tasks_jsonl.parent.parent),
        )
        assert len(samples) == 1
        assert samples[0].id == "agriculture_02"

    def test_sample_has_required_fields(self, tasks_jsonl):
        samples = load_samples(split="dev", data_dir=str(tasks_jsonl.parent.parent))
        s = samples[0]
        assert s.id == "agriculture_02"
        assert "Task:" in s.input
        assert "Available datasets" in s.input
        assert "eval_func" in s.metadata
        assert "output_file_names" in s.metadata
        assert "gold_file_names" in s.metadata
        assert "domain" in s.metadata
        assert isinstance(s.metadata["eval_func"], list)

    def test_sample_prompt_includes_data_sources(self, tasks_jsonl):
        samples = load_samples(split="dev", data_dir=str(tasks_jsonl.parent.parent))
        assert "Crop Recommendation.csv" in samples[0].input

    def test_unknown_domain_returns_empty(self, tasks_jsonl):
        samples = load_samples(
            split="dev",
            domain="nonexistent",
            data_dir=str(tasks_jsonl.parent.parent),
        )
        assert samples == []
