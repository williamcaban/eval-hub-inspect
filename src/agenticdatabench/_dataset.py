"""Dataset loader for AgenticDataBench — loads from HuggingFace with pinned revision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Sample

REPO_ID = "shawnzzzh/AgenticDataBench"
# Pin to a known-good commit. Update this when the dataset changes.
# To find the current SHA: huggingface-cli lfs-files shawnzzzh/AgenticDataBench
HF_REVISION = "main"  # TODO: replace with pinned SHA once dataset is stable

TASK_FILE = "testbed/tasks/dev.jsonl"


def _format_prompt(task: dict[str, Any]) -> str:
    """Build the user-facing task prompt."""
    sources = "\n".join(f"  - {src}" for src in task["data_sources"])
    outputs = ", ".join(
        task["output_file_name"]
        if isinstance(task["output_file_name"], list)
        else [task["output_file_name"]]
    )
    return (
        f"Task: {task['question']}\n\n"
        f"Available datasets (in /datasets/{task['id']}/):\n{sources}\n\n"
        f"Required output file(s): {outputs}\n"
        f"Save your output to /workspace/{outputs} using the exact filename(s) shown."
    )


def load_samples(
    split: str = "dev",
    domain: str | None = None,
    data_dir: str | None = None,
) -> list[Sample]:
    """
    Load AgenticDataBench tasks as inspect_ai Sample objects.

    Args:
        split: Dataset split to load (default: "dev").
        domain: If set, filter to tasks from this domain only.
        data_dir: Path to pre-downloaded data. If None, downloads from HuggingFace.
    """
    if data_dir:
        tasks_path = Path(data_dir) / "tasks" / f"{split}.jsonl"
        lines = tasks_path.read_text(encoding="utf-8").splitlines()
    else:
        tasks_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"testbed/tasks/{split}.jsonl",
                repo_type="dataset",
                revision=HF_REVISION,
            )
        )
        lines = tasks_path.read_text(encoding="utf-8").splitlines()

    samples: list[Sample] = []
    for line in lines:
        if not line.strip():
            continue
        task: dict[str, Any] = json.loads(line)
        task_domain = task.get("domain", "").split("/")[0]
        if domain and task_domain != domain:
            continue

        output_files = (
            task["output_file_name"]
            if isinstance(task["output_file_name"], list)
            else [task["output_file_name"]]
        )
        gold_files = (
            task["gold_file_name"]
            if isinstance(task["gold_file_name"], list)
            else [task["gold_file_name"]]
        )
        eval_func = (
            task["eval_func"] if isinstance(task["eval_func"], list) else [task["eval_func"]]
        )

        samples.append(
            Sample(
                id=task["id"],
                input=_format_prompt(task),
                target="",  # scoring is file-based; not text comparison
                metadata={
                    "domain": task_domain,
                    "skills": task.get("skills", []),
                    "eval_func": eval_func,
                    "output_file_names": output_files,
                    "gold_file_names": gold_files,
                    "data_sources": task.get("data_sources", []),
                },
            )
        )

    return samples
