#!/usr/bin/env python3
"""Parse experiment log file and write results JSON.

Usage: python scripts/collect_result.py <exp_id> <task>

Parses experiments/logs/<exp_id>_<task>.log for EVALUATION RESULTS section.
Writes experiments/results/<exp_id>_<task>.json.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse_log(log_path: Path) -> dict:
    """Extract evaluation metrics from log file."""
    text = log_path.read_text()

    result = {"status": "ok", "original": {}, "translated": {}, "difference": {}}

    # Find EVALUATION RESULTS block (between two === lines)
    # Pattern: === line, EVALUATION RESULTS, === line, content, === line
    match = re.search(
        r"EVALUATION RESULTS\s*\n.*?={40,}\s*\n(.*?)={40,}",
        text,
        re.DOTALL,
    )
    if not match:
        result["status"] = "no_results"
        return result

    block = match.group(1)

    # Parse Original
    orig_section = re.search(r"Original Test Data:(.*?)Translated Test Data:", block, re.DOTALL)
    if orig_section:
        for m in re.finditer(r"(\w+):\s+([\d.]+)", orig_section.group(1)):
            result["original"][m.group(1)] = float(m.group(2))

    # Parse Translated
    trans_section = re.search(r"Translated Test Data:(.*?)Difference:", block, re.DOTALL)
    if trans_section:
        for m in re.finditer(r"(\w+):\s+([\d.]+)", trans_section.group(1)):
            result["translated"][m.group(1)] = float(m.group(2))

    # Parse Difference
    diff_section = re.search(r"Difference:(.*)", block, re.DOTALL)
    if diff_section:
        for m in re.finditer(r"(\w+):\s+([+-]?[\d.]+)", diff_section.group(1)):
            result["difference"][m.group(1)] = float(m.group(2))

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/collect_result.py <exp_id> <task>")
        sys.exit(1)

    exp_id = sys.argv[1]
    task = sys.argv[2]

    log_path = REPO / "experiments" / "logs" / f"{exp_id}_{task}.log"
    out_path = REPO / "experiments" / "results" / f"{exp_id}_{task}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        result = {"status": "missing_log", "exp_id": exp_id, "task": task}
    else:
        result = parse_log(log_path)

    result["exp_id"] = exp_id
    result["task"] = task

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Collected {exp_id}/{task}: status={result['status']}")
    if result.get("difference"):
        for k, v in result["difference"].items():
            print(f"  {k}: {v:+.4f}")


if __name__ == "__main__":
    main()
