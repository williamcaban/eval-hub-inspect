"""Scorer for AgentJudgeBench.

Flow:
  1. Extract CORRECT / INCORRECT from the judge model's completion text.
  2. Compare to the ground-truth label stored in sample metadata.
  3. Return Score(value=1.0) on match, Score(value=0.0) on mismatch.
  4. Return Score.unscored() when the output cannot be parsed.

Primary metric: mean alignment — fraction of judgments that match ground truth.
"""

from __future__ import annotations

import math

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState


def _extract_judgment(completion: str) -> bool | None:
    """Return True for CORRECT, False for INCORRECT, None if unparseable.

    INCORRECT is checked first because it contains the substring CORRECT.
    """
    text = completion.strip().upper()
    # Look at the first 60 chars where the mandatory first word must appear.
    head = text[:60]
    if "INCORRECT" in head:
        return False
    if "CORRECT" in head:
        return True
    # Fall back to scanning the full completion in case the model added preamble.
    if "INCORRECT" in text:
        return False
    if "CORRECT" in text:
        return True
    return None


@scorer(metrics=[mean(), stderr()])
def agentjudgebench_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        completion = state.output.completion if state.output else ""
        judgment = _extract_judgment(completion)

        if judgment is None:
            return Score(
                value=math.nan,
                explanation=f"Could not parse CORRECT/INCORRECT from: {completion[:120]!r}",
            )

        ground_truth: bool = bool(state.metadata.get("label", False))
        aligned = judgment == ground_truth
        difficulty = state.metadata.get("difficulty", "unknown")
        with_gt = state.metadata.get("with_ground_truth", False)

        return Score(
            value=1.0 if aligned else 0.0,
            explanation=(
                f"Judge said {'CORRECT' if judgment else 'INCORRECT'}; "
                f"ground truth is {'CORRECT' if ground_truth else 'INCORRECT'} "
                f"[difficulty={difficulty}, with_gt={with_gt}]"
            ),
            metadata={
                "judgment": judgment,
                "ground_truth": ground_truth,
                "aligned": aligned,
                "difficulty": difficulty,
                "with_ground_truth": with_gt,
                "dag_topology": state.metadata.get("dag_topology", "unknown"),
            },
        )

    return score
