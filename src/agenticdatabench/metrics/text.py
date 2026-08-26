"""
Text Metrics Module - Text comparison metric for evaluation.

This module compares text output:
- Case-insensitive, whitespace-normalized comparison
- Provides detailed error messages for mismatches
- Returns score (0.0 or 1.0) and comparison details

Ported from DA-Code (MIT) — https://github.com/yiyihum/da-code
"""

from __future__ import annotations

import logging
import os
from typing import Any


def compare_text(
    output_file_name: str,
    gold_file_name: str,
) -> dict[str, Any]:
    """
    Evaluate LLM output by comparing text content.

    Args:
        output_file_name: Path to the output/predicted text file
        gold_file_name: Path to the gold/expected text file

    Returns:
        dict: {'score': float, 'errors': list[str], 'meaning': str}
            - score: normalized score in range [0, 1]
            - errors: detailed error messages explaining mismatches
            - meaning: description of what the comparison did
    """
    try:
        with open(output_file_name, encoding="utf-8") as f:
            output_text = f.read().strip()
    except (FileNotFoundError, UnicodeDecodeError) as e:
        logging.warning(f"Failed to read output file {output_file_name}: {e}")
        errors = [f"Failed to read output file '{os.path.basename(output_file_name)}': {e}"]
        meaning = (
            f"Failed to compare text files: cannot read output file "
            f"'{os.path.basename(output_file_name)}'"
        )
        return {
            "score": 0.0,
            "errors": errors,
            "meaning": meaning,
            "output_data": None,
            "gold_data": None,
        }

    try:
        with open(gold_file_name, encoding="utf-8") as f:
            reference_text = f.read().strip()
    except (FileNotFoundError, UnicodeDecodeError) as e:
        logging.warning(f"Failed to read gold file {gold_file_name}: {e}")
        errors = [f"Failed to read gold file '{os.path.basename(gold_file_name)}': {e}"]
        meaning = (
            f"Failed to compare text files: cannot read gold file "
            f"'{os.path.basename(gold_file_name)}'"
        )
        return {
            "score": 0.0,
            "errors": errors,
            "meaning": meaning,
            "output_data": None,
            "gold_data": None,
        }

    # Case-insensitive, whitespace-normalized comparison
    output_normalized = " ".join(output_text.lower().split())
    reference_normalized = " ".join(reference_text.lower().split())

    errors: list[str] = []

    if len(output_text) < 1000:
        output_data: str | None = output_text
    else:
        output_data = output_text[:1000] + f"\n... ({len(output_text)} characters total)"

    if len(reference_text) < 1000:
        gold_data: str | None = reference_text
    else:
        gold_data = reference_text[:1000] + f"\n... ({len(reference_text)} characters total)"

    if output_normalized != reference_normalized:
        if len(output_text) == 0:
            errors.append("Output is empty but expected text is not")
        elif len(reference_text) == 0:
            errors.append("Output contains text but expected empty")
        else:
            min_len = min(len(output_normalized), len(reference_normalized))
            diff_pos = None
            for i in range(min_len):
                if output_normalized[i] != reference_normalized[i]:
                    diff_pos = i
                    break

            if diff_pos is not None:
                start = max(0, diff_pos - 20)
                end = min(min_len, diff_pos + 20)
                context_output = output_normalized[start:end]
                context_ref = reference_normalized[start:end]
                errors.append(
                    f"Text mismatch at position {diff_pos}: output has '{context_output}' "
                    f"but expected '{context_ref}'"
                )
            else:
                if len(output_normalized) > len(reference_normalized):
                    extra = output_normalized[min_len : min_len + 50]
                    errors.append(
                        f"Output has extra text: '{extra}...' "
                        f"(length {len(output_text)} vs {len(reference_text)})"
                    )
                else:
                    missing = reference_normalized[min_len : min_len + 50]
                    errors.append(
                        f"Output is missing text: expected '{missing}...' "
                        f"(length {len(output_text)} vs {len(reference_text)})"
                    )
    else:
        meaning = (
            f"Text comparison successful: output matches gold exactly after normalization "
            f"(case-insensitive, whitespace-normalized). "
            f"Both texts contain {len(output_text)} characters."
        )
        return {
            "score": 1.0,
            "errors": [],
            "meaning": meaning,
            "output_data": output_data,
            "gold_data": gold_data,
        }

    meaning = (
        f"Text comparison: case-insensitive, whitespace-normalized comparison between "
        f"output ({len(output_text)} chars) and gold ({len(reference_text)} chars). Score: 0.0"
    )

    return {
        "score": 0.0,
        "errors": errors,
        "meaning": meaning,
        "output_data": output_data,
        "gold_data": gold_data,
    }
