"""Prompt templates for AgentJudgeBench judge evaluation."""

# Base instruction for the judge model.
# The paper shows CoT and temperature have negligible effect; structured rubrics
# give the most consistent uplift (+6.5 pp) even if inconsistent across pairs.
SYSTEM_PROMPT = """\
You are an expert AI evaluator. Your task is to assess whether an AI agent correctly \
completed a structured tool-calling workflow.

Evaluate the agent's trajectory against these criteria:
1. Tool Selection — were the right tools called for each step?
2. Dependency Order — were tools called in the order required by the workflow's dependencies?
3. Parameter Correctness — were tool arguments correct given prior context?
4. Output Utilization — did the agent use tool results correctly in subsequent steps?
5. Task Completion — was the overall workflow goal achieved?

Respond with EXACTLY ONE of the following as your FIRST word, then give a 2-3 sentence explanation:
  CORRECT   — all criteria pass; the agent completed the workflow correctly
  INCORRECT — one or more criteria fail; the agent made an error or produced a wrong result\
"""


def format_user_prompt(
    workflow: str,
    trajectory_text: str,
    ground_truth: str | None = None,
) -> str:
    """Build the per-sample user prompt."""
    parts = [
        f"## Workflow Task\n{workflow}",
        f"## Agent Trajectory\n{trajectory_text}",
    ]
    if ground_truth:
        parts.append(f"## Reference Answer\n{ground_truth}")
    parts.append("## Your Assessment\nDid the agent correctly complete the workflow?")
    return "\n\n".join(parts)
