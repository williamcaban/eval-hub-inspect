"""inspect_ai @task for AgenticDataBench."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import bash

# Absolute imports so this file works both when loaded directly by `inspect eval`
# (which adds src/ to sys.path via the editable install .pth) and when imported
# as part of the installed agenticdatabench package.
from agenticdatabench._dataset import load_samples
from agenticdatabench._scorer import agenticdatabench_scorer

_SYSTEM_PROMPT = """You are an expert data scientist with access to a bash terminal.

Environment:
- Input datasets are at /datasets/{task_id}/ (read-only)
- Write all output files to /workspace/ using the exact filenames requested
- Python is available with: pandas, numpy, scikit-learn, scipy, matplotlib, statsmodels

Instructions:
1. Read the task carefully and identify the required output file(s) and their exact names.
2. Write Python code that processes the datasets and produces the required output.
3. Execute your code with bash and verify the output file exists with reasonable content.
4. Do not ask for clarification — complete the task fully.
5. If your code fails, debug and retry; do not give up.

When referencing dataset files, use the full path: /datasets/<task_id>/<filename>
"""


@task
def agenticdatabench(
    split: str = "dev",
    domain: str | None = None,
    data_dir: str | None = None,
) -> Task:
    """
    AgenticDataBench: 246 agentic data science tasks across 15 domains.

    Args:
        split: Dataset split to use (default: "dev").
        domain: Restrict to a single domain (e.g. "agriculture", "finance").
        data_dir: Path to pre-downloaded data directory. If None, downloads from HuggingFace.
    """
    samples = load_samples(split=split, domain=domain, data_dir=data_dir)
    return Task(
        dataset=samples,
        solver=[
            system_message(_SYSTEM_PROMPT),
            use_tools(bash()),
            generate(),
        ],
        scorer=agenticdatabench_scorer(),
        # Path is relative to this file's directory (src/agenticdatabench/)
        sandbox=("docker", "docker/compose.yaml"),
        max_messages=30,
    )
