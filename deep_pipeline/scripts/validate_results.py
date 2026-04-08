#!/usr/bin/env python3
"""Lightweight experiment result integrity check.

Importable by gpu_scheduler.py and runnable standalone:
    python scripts/validate_results.py           # all results
    python scripts/validate_results.py NAME      # filter by name substring
"""
import json
import math
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"


def validate_experiment_result(name: str, task: str | None = None) -> list[str]:
    """Return list of issue strings for one experiment. Empty list = OK.

    Args:
        name: Experiment name (e.g. 'aki_v5_cross3')
        task: Task suffix used in result filename (e.g. 'aki'). If None, inferred.
    """
    if task is None:
        for t in ("mortality", "mort", "aki", "sepsis", "los", "kf", "kidney_function"):
            if t in name.lower():
                task = {"mort": "mortality"}.get(t, t)
                break
        else:
            task = ""

    candidates = [
        RESULTS_DIR / f"{name}_{task}.json",
        RESULTS_DIR / f"{name}.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)

    if json_path is None:
        return [f"MISSING_RESULT_JSON {name}"]

    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        return [f"PARSE_ERROR {name}: {e}"]

    if data.get("status") == "missing_log":
        return [f"MISSING_LOG {name}"]

    issues = []
    for key in ("auroc", "auprc", "mae", "mse", "rmse", "r2"):
        val = data.get(key)
        if val is None:
            # Also check nested dict format (status=ok, original/translated/difference)
            if data.get("status") == "ok":
                translated = data.get("translated", {})
                val = translated.get(key.upper())
            if val is None:
                continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            issues.append(f"NON_NUMERIC {name}.{key}={val!r}")
            continue
        if math.isnan(fval):
            issues.append(f"NAN {name}.{key}")
        elif math.isinf(fval):
            issues.append(f"INF {name}.{key}={fval}")

    # Sanity: AUROC in classification results should be in [0.5, 1.0]
    auroc = data.get("auroc")
    if auroc is None and data.get("status") == "ok":
        auroc = data.get("translated", {}).get("AUROC")
    if auroc is not None:
        try:
            if 0 < float(auroc) < 0.5:
                issues.append(f"LOW_AUROC {name}={float(auroc):.4f} (below chance — likely invalid)")
        except (TypeError, ValueError):
            pass

    return issues


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("-")]
    name_filter = args[0] if args else None

    all_files = sorted(RESULTS_DIR.glob("*.json"))
    # Exclude meta files and diagnosis logs
    all_files = [f for f in all_files if not f.stem.startswith("_")]
    if name_filter:
        all_files = [f for f in all_files if name_filter.lower() in f.stem.lower()]

    if not all_files:
        print(f"[validate] No result files found" + (f" matching '{name_filter}'" if name_filter else ""))
        return 0

    all_issues, ok_count = [], 0
    for p in all_files:
        issues = validate_experiment_result(p.stem)
        if issues:
            all_issues.extend(issues)
        else:
            ok_count += 1

    if all_issues:
        print(f"[validate] {len(all_issues)} issue(s), {ok_count} OK:")
        for issue in all_issues[-20:]:
            print(f"  {issue}")
        return 1
    else:
        print(f"[validate] All {ok_count} result files OK")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
