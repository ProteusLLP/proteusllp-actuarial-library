"""Summarize manually or externally assessed PAL coding-assistant results.

This command does not execute model-generated code. Execute and inspect such
code only in a suitable isolated environment before recording the verdicts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CRITERIA = ("runs", "public_api", "numerical", "semantics", "coupling")


def main() -> None:
    """Read per-task verdicts and report strict pass rate and failure types."""
    parser = argparse.ArgumentParser(description="Score PAL assistant-generated code.")
    parser.add_argument("results", type=Path, help="JSONL with one verdict per evaluation task")
    args = parser.parse_args()
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    total = 0

    for line_number, line in enumerate(args.results.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        result = json.loads(line)
        task_id = result["task_id"]
        if task_id in seen:
            raise ValueError(f"Repeated task_id {task_id!r} on line {line_number}")
        seen.add(task_id)
        for criterion in CRITERIA:
            if type(result.get(criterion)) is not bool:
                raise ValueError(f"{task_id}: {criterion} must be a boolean")
            counts[criterion] += result[criterion]
        counts["pass"] += all(result[key] for key in CRITERIA)
        total += 1

    if not total:
        raise ValueError("No evaluation verdicts provided")

    print(f"Tasks assessed: {total}")
    for criterion in (*CRITERIA, "pass"):
        print(f"{criterion}: {counts[criterion]}/{total} ({counts[criterion] / total:.1%})")


if __name__ == "__main__":
    main()
