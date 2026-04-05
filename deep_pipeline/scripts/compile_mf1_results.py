#!/usr/bin/env python3
"""Compile Macro-F1 (MF1) results from all 30 AdaTime scenarios.

Reads results.json from each scenario and outputs:
  1. experiments/results/adatime_mf1_results.json  (machine-readable)
  2. experiments/results/adatime_mf1_comparison.md  (human-readable comparison table)
"""
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "adatime"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The 30 standard AdaTime scenarios (matching the paper)
SCENARIOS = {
    "HAR": [
        "2_to_11", "6_to_23", "7_to_13", "9_to_18", "12_to_16",
        "18_to_27", "20_to_5", "24_to_8", "28_to_27", "30_to_20",
    ],
    "HHAR": [
        "0_to_2", "0_to_6", "1_to_6", "2_to_7", "3_to_8",
        "4_to_5", "5_to_0", "6_to_1", "7_to_4", "8_to_3",
    ],
    "WISDM": [
        "2_to_11", "5_to_26", "6_to_19", "7_to_18", "17_to_23",
        "20_to_30", "23_to_32", "28_to_4", "33_to_12", "35_to_31",
    ],
}

# AdaTime published MF1 (Table 4, TGT risk, 1D-CNN backbone)
ADATIME_PUBLISHED = {
    "HAR": {
        "Source-only": 72.09,
        "DANN": 80.49,
        "CDAN": 82.74,
        "CoDATS": 78.92,
        "CORAL": 73.08,
        "AdvSKM": 73.87,
        "DIRT-T": 83.27,
        "CoTMix": 84.93,
    },
    "HHAR": {
        "Source-only": 70.17,
        "DANN": 77.88,
        "CDAN": 75.81,
        "CoDATS": 73.65,
        "CORAL": 73.39,
        "AdvSKM": 72.75,
        "DIRT-T": 81.09,
    },
    "WISDM": {
        "Source-only": 48.53,
        "DANN": 56.21,
        "CDAN": 53.93,
        "CoDATS": 47.04,
        "CORAL": 50.34,
        "AdvSKM": 48.30,
        "DIRT-T": 63.17,
    },
}


def load_results():
    """Load all scenario results and extract MF1 scores."""
    all_results = {}

    for dataset, scenarios in SCENARIOS.items():
        all_results[dataset] = {}
        for scenario in scenarios:
            results_path = RUNS_DIR / dataset / scenario / "results.json"
            if not results_path.exists():
                print(f"WARNING: Missing {results_path}")
                continue

            with open(results_path) as f:
                data = json.load(f)

            entry = {}

            # Source-only MF1
            so = data.get("source_only", {})
            entry["source_only_mf1"] = so.get("f1", None)
            entry["source_only_acc"] = so.get("accuracy", None)

            # Translator MF1
            tr = data.get("translator", {})
            entry["translator_mf1"] = tr.get("f1", None)
            entry["translator_acc"] = tr.get("accuracy", None)

            # DANN (frozen) MF1
            dann = data.get("dann_frozen", {})
            entry["dann_mf1"] = dann.get("f1", None)
            entry["dann_acc"] = dann.get("accuracy", None)

            # Target-only (upper bound) if available
            to = data.get("target_only", {})
            if to:
                entry["target_only_mf1"] = to.get("f1", None)
                entry["target_only_acc"] = to.get("accuracy", None)

            all_results[dataset][scenario] = entry

    return all_results


def compute_summary(all_results):
    """Compute per-dataset mean MF1 and standard deviation."""
    summary = {}
    for dataset, scenarios in all_results.items():
        so_mf1s = [s["source_only_mf1"] for s in scenarios.values() if s.get("source_only_mf1") is not None]
        tr_mf1s = [s["translator_mf1"] for s in scenarios.values() if s.get("translator_mf1") is not None]
        dann_mf1s = [s["dann_mf1"] for s in scenarios.values() if s.get("dann_mf1") is not None]
        to_mf1s = [s.get("target_only_mf1") for s in scenarios.values() if s.get("target_only_mf1") is not None]

        summary[dataset] = {
            "n_scenarios": len(scenarios),
            "source_only_mean_mf1": mean(so_mf1s) if so_mf1s else None,
            "source_only_std_mf1": stdev(so_mf1s) if len(so_mf1s) > 1 else None,
            "translator_mean_mf1": mean(tr_mf1s) if tr_mf1s else None,
            "translator_std_mf1": stdev(tr_mf1s) if len(tr_mf1s) > 1 else None,
            "dann_mean_mf1": mean(dann_mf1s) if dann_mf1s else None,
            "dann_std_mf1": stdev(dann_mf1s) if len(dann_mf1s) > 1 else None,
        }
        if to_mf1s:
            summary[dataset]["target_only_mean_mf1"] = mean(to_mf1s)
            summary[dataset]["target_only_std_mf1"] = stdev(to_mf1s) if len(to_mf1s) > 1 else None

    return summary


def generate_comparison_md(all_results, summary):
    """Generate the comparison markdown table."""
    lines = []
    lines.append("# AdaTime MF1 Comparison: Our Method vs. Published Baselines")
    lines.append("")
    lines.append("Macro-F1 (MF1) scores averaged over 10 scenarios per dataset.")
    lines.append("Our method uses a frozen target LSTM with retrieval-based translation.")
    lines.append("AdaTime baselines use 1D-CNN backbone (TGT risk, from Table 4 of Ragab et al., TKDD 2023).")
    lines.append("")

    # Summary comparison table
    lines.append("## Summary: Mean MF1 (x100)")
    lines.append("")
    lines.append("| Method | HAR | HHAR | WISDM | Avg |")
    lines.append("|--------|-----|------|-------|-----|")

    # Our methods
    for method_key, method_label in [
        ("source_only", "Source-only (ours, LSTM)"),
        ("translator", "Translator (ours, LSTM)"),
        ("dann", "DANN-frozen (ours, LSTM)"),
    ]:
        vals = []
        for ds in ["HAR", "HHAR", "WISDM"]:
            v = summary[ds].get(f"{method_key}_mean_mf1")
            if v is not None:
                vals.append(v * 100)
            else:
                vals.append(None)
        val_strs = [f"{v:.2f}" if v is not None else "-" for v in vals]
        avg = mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
        avg_str = f"{avg:.2f}" if avg is not None else "-"
        lines.append(f"| {method_label} | {val_strs[0]} | {val_strs[1]} | {val_strs[2]} | {avg_str} |")

    lines.append("|--------|-----|------|-------|-----|")

    # AdaTime published baselines
    adatime_methods = ["Source-only", "DANN", "CDAN", "CoDATS", "CORAL", "AdvSKM", "DIRT-T", "CoTMix"]
    for method in adatime_methods:
        vals = []
        for ds in ["HAR", "HHAR", "WISDM"]:
            v = ADATIME_PUBLISHED.get(ds, {}).get(method)
            vals.append(v)
        val_strs = [f"{v:.2f}" if v is not None else "-" for v in vals]
        valid_vals = [v for v in vals if v is not None]
        avg = mean(valid_vals) if valid_vals else None
        avg_str = f"{avg:.2f}" if avg is not None else "-"
        lines.append(f"| {method} (AdaTime, CNN) | {val_strs[0]} | {val_strs[1]} | {val_strs[2]} | {avg_str} |")

    lines.append("")

    # Gain analysis
    lines.append("## Translator Gain Analysis")
    lines.append("")
    lines.append("| Dataset | Our Source-only | Our Translator | Our Gain | AdaTime Source-only | Best AdaTime DA | AdaTime Best Gain |")
    lines.append("|---------|----------------|----------------|----------|---------------------|-----------------|-------------------|")
    for ds in ["HAR", "HHAR", "WISDM"]:
        our_so = summary[ds]["source_only_mean_mf1"] * 100 if summary[ds]["source_only_mean_mf1"] else 0
        our_tr = summary[ds]["translator_mean_mf1"] * 100 if summary[ds]["translator_mean_mf1"] else 0
        our_gain = our_tr - our_so

        adatime_so = ADATIME_PUBLISHED[ds]["Source-only"]
        # Best AdaTime DA method (excluding source-only)
        adatime_da_vals = {k: v for k, v in ADATIME_PUBLISHED[ds].items() if k != "Source-only" and v is not None}
        best_adatime_method = max(adatime_da_vals, key=adatime_da_vals.get)
        best_adatime_val = adatime_da_vals[best_adatime_method]
        adatime_best_gain = best_adatime_val - adatime_so

        lines.append(
            f"| {ds} | {our_so:.2f} | {our_tr:.2f} | "
            f"+{our_gain:.2f} | {adatime_so:.2f} | "
            f"{best_adatime_val:.2f} ({best_adatime_method}) | +{adatime_best_gain:.2f} |"
        )

    lines.append("")

    # Per-scenario details
    for ds in ["HAR", "HHAR", "WISDM"]:
        lines.append(f"## {ds}: Per-Scenario MF1 (x100)")
        lines.append("")
        lines.append("| Scenario | Source-only | Translator | DANN (frozen) | Gain (Tr-SO) |")
        lines.append("|----------|------------|------------|---------------|-------------|")
        for scenario in SCENARIOS[ds]:
            entry = all_results[ds].get(scenario, {})
            so = entry.get("source_only_mf1")
            tr = entry.get("translator_mf1")
            dann = entry.get("dann_mf1")
            so_str = f"{so*100:.2f}" if so is not None else "-"
            tr_str = f"{tr*100:.2f}" if tr is not None else "-"
            dann_str = f"{dann*100:.2f}" if dann is not None else "-"
            gain = (tr - so) * 100 if (tr is not None and so is not None) else None
            gain_str = f"{gain:+.2f}" if gain is not None else "-"
            lines.append(f"| {scenario} | {so_str} | {tr_str} | {dann_str} | {gain_str} |")

        # Mean row
        so_vals = [all_results[ds][s]["source_only_mf1"]*100 for s in SCENARIOS[ds] if all_results[ds].get(s, {}).get("source_only_mf1") is not None]
        tr_vals = [all_results[ds][s]["translator_mf1"]*100 for s in SCENARIOS[ds] if all_results[ds].get(s, {}).get("translator_mf1") is not None]
        dann_vals = [all_results[ds][s]["dann_mf1"]*100 for s in SCENARIOS[ds] if all_results[ds].get(s, {}).get("dann_mf1") is not None]
        so_mean = mean(so_vals) if so_vals else 0
        tr_mean = mean(tr_vals) if tr_vals else 0
        dann_mean = mean(dann_vals) if dann_vals else 0
        gain_mean = tr_mean - so_mean
        lines.append(f"| **Mean** | **{so_mean:.2f}** | **{tr_mean:.2f}** | **{dann_mean:.2f}** | **{gain_mean:+.2f}** |")
        lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("")
    lines.append("- MF1 = Macro-F1 = average of per-class F1 scores (sklearn `f1_score(average='macro')`).")
    lines.append("- Our F1 metric in results.json was already computed as macro-F1 (verified in `src/benchmarks/adatime/evaluate.py` line 75).")
    lines.append("- AdaTime baselines use a 1D-CNN backbone; our method uses an LSTM. Backbones differ, so direct comparison requires noting this caveat.")
    lines.append("- 'Translator' = frozen target LSTM + retrieval-based translator (no target model fine-tuning).")
    lines.append("- 'DANN (frozen)' = our own frozen-model DANN baseline using the same frozen LSTM + a small adapter MLP.")
    lines.append("- 'Source-only' = frozen target LSTM evaluated on raw source data (no adaptation).")
    lines.append("")

    return "\n".join(lines)


def main():
    all_results = load_results()
    summary = compute_summary(all_results)

    # Build JSON output
    output = {
        "metadata": {
            "description": "AdaTime Macro-F1 (MF1) results for all 30 scenarios",
            "date": "2026-04-04",
            "note": "MF1 = sklearn f1_score(average='macro'). Already computed in evaluate.py.",
            "methods": {
                "source_only": "Frozen target LSTM on raw source data (no adaptation)",
                "translator": "Frozen target LSTM + retrieval translator",
                "dann_frozen": "Frozen target LSTM + DANN adapter (MLP)",
            },
        },
    }

    # Per-dataset results
    for ds in ["HAR", "HHAR", "WISDM"]:
        output[ds] = {}
        for scenario in SCENARIOS[ds]:
            entry = all_results[ds].get(scenario, {})
            output[ds][scenario] = {
                "source_only_mf1": round(entry.get("source_only_mf1", 0), 6),
                "translator_mf1": round(entry.get("translator_mf1", 0), 6),
                "dann_mf1": round(entry.get("dann_mf1", 0) if entry.get("dann_mf1") is not None else 0, 6),
            }

    # Summary
    output["summary"] = {}
    for ds in ["HAR", "HHAR", "WISDM"]:
        s = summary[ds]
        output["summary"][ds] = {
            "n_scenarios": s["n_scenarios"],
            "source_only_mean_mf1": round(s["source_only_mean_mf1"], 6) if s["source_only_mean_mf1"] is not None else None,
            "source_only_std_mf1": round(s["source_only_std_mf1"], 6) if s["source_only_std_mf1"] is not None else None,
            "translator_mean_mf1": round(s["translator_mean_mf1"], 6) if s["translator_mean_mf1"] is not None else None,
            "translator_std_mf1": round(s["translator_std_mf1"], 6) if s["translator_std_mf1"] is not None else None,
            "dann_mean_mf1": round(s["dann_mean_mf1"], 6) if s["dann_mean_mf1"] is not None else None,
            "dann_std_mf1": round(s["dann_std_mf1"], 6) if s["dann_std_mf1"] is not None else None,
        }

    # Save JSON
    json_path = OUTPUT_DIR / "adatime_mf1_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {json_path}")

    # Save MD comparison
    md_content = generate_comparison_md(all_results, summary)
    md_path = OUTPUT_DIR / "adatime_mf1_comparison.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved: {md_path}")

    # Print summary to console
    print("\n=== MF1 Summary (x100) ===")
    for ds in ["HAR", "HHAR", "WISDM"]:
        s = summary[ds]
        so = s["source_only_mean_mf1"] * 100 if s["source_only_mean_mf1"] else 0
        tr = s["translator_mean_mf1"] * 100 if s["translator_mean_mf1"] else 0
        dann = s["dann_mean_mf1"] * 100 if s["dann_mean_mf1"] else 0
        print(f"  {ds:6s}: Source-only={so:6.2f}  Translator={tr:6.2f}  DANN={dann:6.2f}  Gain(Tr-SO)={tr-so:+.2f}")


if __name__ == "__main__":
    main()
