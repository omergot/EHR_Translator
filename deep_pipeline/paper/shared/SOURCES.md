# Paper Source Index

Maps each paper section to its primary data source. Use this to quickly find authoritative numbers.

| Paper Section               | Primary Source                                       | Notes |
|-----------------------------|------------------------------------------------------|-------|
| Abstract (numbers)          | MEMORY.md "Current Best Results"                     | Best deltas per task |
| Introduction (framing)      | `memory/project_neurips_framing_research.md`         | Novelty claims, hooks |
| Related work                | `docs/neurips/related_work.md`                       | 50+ papers, 6 categories |
| Related work (polished TeX) | `docs/neurips/positioning_paper/part1_related_work.tex` | Publication-ready prose, Table 1 |
| Theory (gradient magnitude) | `docs/neurips/gradient_magnitude_theory.md`          | Replaces falsified cosine theory |
| Method (polished TeX)       | `docs/neurips/positioning_paper/part2_method.tex`    | Eq. 1-36, all 3 architectures |
| Method (architecture ref)   | `docs/retrieval_translator_architecture.md`          | Retrieval translator details |
| Main results + CIs          | `docs/neurips/bootstrap_ci_results.md`               | 500-replicate, all tasks |
| DA baselines table          | `docs/neurips/da_baselines_results.md`               | 8 methods x 3 tasks |
| Baseline strategy/narrative | `docs/neurips/baseline_strategy_final.md`            | Padding discovery, two-tier narrative |
| E2E audit (fairness)        | `docs/neurips/e2e_baselines_audit.md`                | LSTM leakage, architectural honesty |
| Multi-source (HiRID)        | MEMORY.md "HiRID Results" + `bootstrap_ci_results.md`| All 5 HiRID tasks |
| MAS (arch-agnostic)         | MEMORY.md "MAS" section                              | GRU/TCN transfer percentages |
| AdaTime benchmark           | `docs/adatime_experiments_summary.md`                | 5 datasets x 5 seeds |
| Ablation table              | `docs/comprehensive_results_summary.md` Section 20   | V3 ablation matrix |
| Seed variance               | MEMORY.md "Seed Variance"                            | 3-5 seed stats |
| Computational cost          | `docs/neurips/computational_cost.md`                 | Params, GPU hours, VRAM |
| Calibration                 | `docs/neurips/calibration_analysis.md`               | ECE/Brier, reliability diagrams |
| Visualization/divergence    | `docs/neurips/visualization_analysis.md`             | Input-space + hidden-space divergence |
| Reviewer gap checklist      | `docs/neurips/gap_analysis_generality_claim.md`      | P0-P2 gaps from reviewer perspective |
| Reviewer Q&A                | `docs/neurips/reviewer_strategy_analysis.md`         | Anticipated questions + rebuttals |
| Statistical completeness    | `docs/neurips/statistical_completeness.md`           | What stats are needed for paper |
| NeurIPS strategy            | `docs/neurips/neurips_strategy.md`                   | Acceptance model, paper structure |
| YAIB references             | `docs/yaib_reference_baselines.md`                   | In-domain LSTM scores |
| Differentiation (TeX)       | `docs/neurips/positioning_paper/part3_differentiation.tex` | Comparison vs 15+ methods |
| Figure generation scripts   | `paper/shared/figures/*.py`                          | Data paths in each script |
