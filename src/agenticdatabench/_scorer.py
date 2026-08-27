"""
Scorer for AgenticDataBench.

Flow:
  1. Read agent output files from Docker sandbox (/workspace/<file>).
  2. Read gold files from Docker sandbox (/gold/<task_id>/<file>).
  3. Write both to a temporary directory on the host.
  4. Execute each eval_func string using the ported metric functions.
  5. Return Score(value=mean_score).

The eval_func strings use eval() on trusted, statically-defined strings from
the benchmark dataset — not user input. See noqa annotations at each call site.
"""

from __future__ import annotations

import logging
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from .metrics import (
    compare_csv,
    compare_image,
    compare_json,
    compare_json_normalized,
    compare_model,
    compare_sqlite,
    compare_text,
)

# Namespace passed to eval() — only the metric functions, no builtins beyond what's needed.
_METRIC_NAMESPACE: dict[str, Any] = {
    "compare_csv": compare_csv,
    "compare_sqlite": compare_sqlite,
    "compare_text": compare_text,
    "compare_json": compare_json,
    "compare_json_normalized": compare_json_normalized,
    "compare_image": compare_image,
    "compare_model": compare_model,
}

# File extensions that should be read as binary
_BINARY_EXTENSIONS = {
    ".npy",
    ".pkl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".db",
    ".sqlite",
}


def _is_binary(filename: str) -> bool:
    return Path(filename).suffix.lower() in _BINARY_EXTENSIONS


@scorer(metrics=[mean(), stderr()])
def agenticdatabench_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        task_id = str(state.sample_id)
        metadata = state.metadata
        eval_funcs: list[str] = metadata.get("eval_func", [])
        output_names: list[str] = metadata.get("output_file_names", [])
        gold_names: list[str] = metadata.get("gold_file_names", [])

        if not eval_funcs:
            return Score.unscored()

        sb = sandbox()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_map: dict[str, str] = {}
            gold_map: dict[str, str] = {}

            # Read output files from sandbox
            for fname in output_names:
                local = tmp / "output" / fname
                local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if _is_binary(fname):
                        local.write_bytes(await sb.read_file(f"/workspace/{fname}", text=False))
                    else:
                        local.write_text(
                            await sb.read_file(f"/workspace/{fname}", text=True),
                            encoding="utf-8",
                        )
                    output_map[fname] = str(local)
                except Exception as exc:
                    logging.debug(f"Output file {fname} not produced: {exc}")

            # Read gold files from sandbox
            for fname in gold_names:
                base = Path(fname).name
                local = tmp / "gold" / base
                local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if _is_binary(fname):
                        local.write_bytes(await sb.read_file(f"/gold/{task_id}/{base}", text=False))
                    else:
                        local.write_text(
                            await sb.read_file(f"/gold/{task_id}/{base}", text=True),
                            encoding="utf-8",
                        )
                    gold_map[fname] = str(local)
                    gold_map[base] = str(local)
                except Exception as exc:
                    logging.debug(f"Gold file {fname} not available: {exc}")

            # Resolve file paths in each eval_func and execute
            func_scores: list[float] = []
            explanations: list[str] = []

            for func_str in eval_funcs:
                resolved = func_str
                for fname, local_path in output_map.items():
                    resolved = re.sub(
                        rf"(['\"]){re.escape(fname)}\1",
                        f"'{local_path}'",
                        resolved,
                    )
                for fname, local_path in gold_map.items():
                    resolved = re.sub(
                        rf"(['\"]){re.escape(fname)}\1",
                        f"'{local_path}'",
                        resolved,
                    )

                try:
                    # S307: eval_func is a static string from the curated benchmark
                    # dataset, not user input. The namespace is restricted to metric
                    # functions only — no builtins, no os, no subprocess.
                    result = eval(  # noqa: S307
                        resolved, {"__builtins__": {}}, _METRIC_NAMESPACE
                    )
                    if isinstance(result, dict):
                        s = float(result.get("score", 0.0))
                        errs = result.get("errors", [])
                        if errs:
                            explanations.extend(str(e) for e in errs[:3])
                    else:
                        s = float(result)
                except FileNotFoundError:
                    # Output file not produced — model failed the task
                    s = 0.0
                    explanations.append(f"Output file missing for: {func_str[:60]}")
                except Exception:
                    # Instrument failure: eval_func itself errored — do not blame model
                    return Score.unscored()

                func_scores.append(s)

            if not func_scores:
                return Score.unscored()

            final = sum(func_scores) / len(func_scores)
            explanation = "; ".join(explanations) if explanations else "All checks passed"
            return Score(
                value=final,
                explanation=explanation,
                metadata={"per_func_scores": func_scores, "task_id": task_id},
            )

    return score


def _unscored_nan() -> float:
    """Return nan for use in Score.unscored() check."""
    return math.nan
