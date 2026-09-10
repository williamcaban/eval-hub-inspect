"""Dataset loader for AgentJudgeBench.

Dataset schema (each JSONL record):
  id:              str         unique identifier, e.g. "ajb_linear_hard_042"
  workflow:        str         natural-language description of the multi-step task
  trajectory:      list[dict]  sequence of tool calls:
                               [{"tool": str, "args": dict, "result": dict}, ...]
  difficulty:      str         "easy" | "medium" | "hard"
  dag_topology:    str         one of the 6 DAG topologies used in the paper
  label:           bool        True = agent trajectory is CORRECT, False = INCORRECT
  ground_truth:    str | None  reference explanation; present in both label classes
  generator_model: str         model that produced the trajectory

NOTE: The dataset (arXiv:2608.26623) may not yet have a public HuggingFace release.
      Set REPO_ID below once it is published, or supply data_dir for local data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Sample

from agentjudgebench._prompts import format_user_prompt

# Update once the authors publish the dataset.
REPO_ID = "AgentJudgeBench/agentjudgebench"
HF_REVISION = "main"

_SPLIT_FILES = {
    "test": "data/test.jsonl",
    "dev": "data/dev.jsonl",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _format_trajectory(trajectory: list[dict[str, Any]]) -> str:
    """Render a list of tool-call dicts as a numbered, readable sequence."""
    lines: list[str] = []
    for i, step in enumerate(trajectory, 1):
        tool = step.get("tool", "unknown_tool")
        args = json.dumps(step.get("args", {}), indent=2)
        result = json.dumps(step.get("result", {}), indent=2)
        lines.append(f"Step {i}: {tool}(\n  args={args},\n  result={result}\n)")
    return "\n\n".join(lines) if lines else "(no trajectory recorded)"


def load_samples(
    split: str = "test",
    difficulty: str | None = None,
    dag_topology: str | None = None,
    with_ground_truth: bool = False,
    max_samples: int | None = None,
    data_dir: str | None = None,
) -> list[Sample]:
    """Load AgentJudgeBench records as inspect_ai Sample objects.

    Args:
        split:             Dataset split to load ("test" or "dev").
        difficulty:        If set, restrict to "easy", "medium", or "hard".
        dag_topology:      If set, restrict to records with this dag_topology value.
        with_ground_truth: Include the reference answer in each prompt.
        max_samples:       Cap the number of samples (useful for smoke tests).
        data_dir:          Path to a local directory containing data/<split>.jsonl.
                           If None, downloads from HuggingFace.
    """
    if difficulty and difficulty not in VALID_DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {VALID_DIFFICULTIES!r}, got {difficulty!r}")

    filename = _SPLIT_FILES.get(split, f"data/{split}.jsonl")

    if data_dir:
        records_path = Path(data_dir) / filename
        raw = records_path.read_text(encoding="utf-8")
    else:
        local = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=HF_REVISION,
            )
        )
        raw = local.read_text(encoding="utf-8")

    samples: list[Sample] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)

        if difficulty and record.get("difficulty") != difficulty:
            continue
        if dag_topology and record.get("dag_topology") != dag_topology:
            continue

        label: bool = bool(record.get("label", False))
        gt_text = record.get("ground_truth") if with_ground_truth else None
        trajectory_text = _format_trajectory(record.get("trajectory", []))

        samples.append(
            Sample(
                id=record["id"],
                input=format_user_prompt(
                    workflow=record["workflow"],
                    trajectory_text=trajectory_text,
                    ground_truth=gt_text,
                ),
                target="CORRECT" if label else "INCORRECT",
                metadata={
                    "difficulty": record.get("difficulty", "unknown"),
                    "dag_topology": record.get("dag_topology", "unknown"),
                    "generator_model": record.get("generator_model", "unknown"),
                    "with_ground_truth": with_ground_truth,
                    "label": label,
                },
            )
        )

        if max_samples and len(samples) >= max_samples:
            break

    return samples
