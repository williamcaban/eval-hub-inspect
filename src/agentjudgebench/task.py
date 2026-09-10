"""inspect_ai @task for AgentJudgeBench."""

from __future__ import annotations

from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message

from agentjudgebench._dataset import load_samples
from agentjudgebench._prompts import SYSTEM_PROMPT
from agentjudgebench._scorer import agentjudgebench_scorer


@task
def agentjudgebench(
    split: str = "test",
    difficulty: Literal["easy", "medium", "hard"] | None = None,
    with_ground_truth: bool = False,
    dag_topology: str | None = None,
    max_samples: int | None = None,
    data_dir: str | None = None,
) -> Task:
    """AgentJudgeBench: meta-evaluation of LLM judges on agentic tool-calling.

    The model under evaluation acts as a judge: given a tool-calling trajectory
    produced by a generator agent, it must assess whether the agent correctly
    completed the workflow. The scorer measures alignment against ground-truth
    correctness labels across 3,808 instances and six DAG topologies.

    Key finding from the paper (arXiv:2608.26623): judge alignment degrades
    monotonically with task difficulty, 1.5x faster without ground truth. On
    hard tasks without GT, all judges converge to a 77-82% accuracy band
    regardless of model scale.

    Args:
        split:             Dataset split ("test" or "dev").
        difficulty:        Filter to "easy", "medium", or "hard" instances.
        with_ground_truth: Include the reference answer in each prompt.
                           Note: the paper shows GT can *reduce* alignment for
                           frontier models; omit when evaluating blind judgment.
        dag_topology:      Filter to a specific DAG topology string.
        max_samples:       Limit instances for smoke tests or cost control.
        data_dir:          Local directory with data/<split>.jsonl.
                           If None, downloads from HuggingFace.
    """
    samples = load_samples(
        split=split,
        difficulty=difficulty,
        dag_topology=dag_topology,
        with_ground_truth=with_ground_truth,
        max_samples=max_samples,
        data_dir=data_dir,
    )
    return Task(
        dataset=samples,
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=agentjudgebench_scorer(),
        # No sandbox — the judge model only reads text and returns text.
    )
