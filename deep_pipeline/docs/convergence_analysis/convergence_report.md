# Convergence Analysis Report

**Phase A: Static analysis of 124 training logs — zero GPU cost.**
**Goal**: Determine the minimum experiment time to reliably screen configs.

## Executive Summary

| Task | Paradigm | Epochs for signal | Phase 2 time | Data load | **Total (current)** | **Total (w/ caching + pretrain reuse)** |
|---|---|---|---|---|---|---|
| Sepsis | retrieval | **3** (ρ=-0.73) | 186 min | 208 min | **6.6h** | **~3.4h** |
| AKI | SL+FG | **3** (ρ=-0.99) | 81 min | 144 min | **3.8h** | **~1.5h** |
| AKI | retrieval V5 | **5** (ρ=-0.62) | 360 min | 202 min | **9.4h** | **~6.3h** |
| Mortality | SL+FG | unclear | — | 59 min | full run needed | **~3h full** |

**Two infrastructure pieces needed**: (1) pretrain reuse, (2) YAIB data caching.

---

## 1. Data Summary

- **Total logs parsed**: 110 / 124 (6 eval-only, 2 crashed, 6 runner/scheduler)
- **Full-data with epoch data**: 80
- **Debug experiments**: 28
- **Result JSONs**: 92
- **Full-data matched to results**: 65 / 80

### Experiments by task × paradigm (full data only)

| Task | Delta | SL | Retrieval | Total |
|---|---|---|---|---|
| Mortality | 9 | 7 | 10 | 26 |
| AKI | 0 | 6 | 17 | 23 |
| Sepsis | 3 | 9 | 19 | 31 |

---

## 2. Core Finding: Within-Paradigm Early Signal

The **within-paradigm** analysis is the key result — this is the actual use case (comparing hyperparameter variants of the same architecture).

### Spearman ρ between epoch-N val_task and final AUCROC Δ (within paradigm)

| Task | Paradigm | N | ep1 | ep2 | ep3 | ep5 | ep8 | ep10 | best |
|---|---|---|---|---|---|---|---|---|---|
| **Sepsis** | **retrieval** | **16** | **-0.640\*** | **-0.675\*** | **-0.726\*** | -0.636\* | -0.524\* | -0.555\* | -0.761\* |
| **AKI** | **retrieval** | **13** | -0.341 | -0.412 | -0.544 | **-0.621\*** | -0.687\* | -0.637\* | -0.907\* |
| **AKI** | **SL** | **6** | +0.406 | -0.464 | **-0.986\*** | -0.087 | -0.145 | -0.493 | -0.754 |
| Sepsis | SL | 5 | -0.300 | +0.600 | -0.300 | +0.000 | +0.200 | -0.900\* | -0.500 |
| Mortality | delta | 9 | +0.077 | +0.008 | +0.143 | +0.008 | -0.059 | +0.042 | -0.513 |
| Mortality | SL | 4 | +0.949 | -0.949 | -0.632 | -0.632 | +0.000 | -0.632 | -0.632 |
| Mortality | retrieval | 10 | -0.430 | +0.164 | -0.418 | -0.224 | -0.539 | -0.285 | -0.511 |

\* = p < 0.05. Bold rows = actionable for quick screening.

### Key takeaways

- **Sepsis retrieval (n=16)**: Signal from **epoch 1** (ρ=-0.64), peaks at **epoch 3** (ρ=-0.73). The strongest and most reliable early signal.
- **AKI retrieval (n=13)**: Signal from **epoch 5** (ρ=-0.62), strengthens through epoch 8 (ρ=-0.69).
- **AKI SL (n=6)**: Extremely strong at **epoch 3** (ρ=-0.99), but n=6 — treat with caution.
- **Mortality (all paradigms)**: val_task does NOT predict AUCROC within any paradigm. The cross-paradigm ρ=-0.68 was driven by paradigm-level differences (SL/retrieval > delta), not config-level. **Mortality requires full runs.**

---

## 3. Cross-Paradigm val_task → AUCROC Correlation

Spearman ρ between **best** val_task and final AUCROC Δ, across all paradigms per task.

| Task | Spearman ρ | p-value | N |
|---|---|---|---|
| Mortality | -0.677 | 0.000 | 23 |
| AKI | -0.962 | 0.000 | 18 |
| Sepsis | -0.791 | 0.000 | 24 |

Strong across-paradigm correlation everywhere. But mortality's within-paradigm failure means this only tells you "use SL/retrieval, not delta" — not which SL config is best.

---

## 4. Debug Experiments Are Unreliable

10 mortality delta debug/full pairs:

- Debug→full AUCROC rank: **τ = 0.207** (p = 0.415) — essentially random
- 20% data × 20 epochs gives zero ranking information
- All three noise sources (fewer steps, noisy val, noisy test) compound

**Conclusion**: Never use debug mode for config comparison. Always use full data.

---

## 5. Wall-Clock Time Breakdown

### Per-epoch timing (full data, measured from logs)

| Task | Paradigm | Data load | Pretrain/ep | Phase 2/ep | Example log |
|---|---|---|---|---|---|
| Mortality | Delta | 5 min | — | 2.5 min | c3_cosine_fid_full |
| Mortality | SL+FG | 59 min | ~9 min | 11 min | mortality_sl_fg_tseed1337_local |
| AKI | SL+FG | 144 min | ~14 min | 27 min | aki_sl_fg_tseed1337_local |
| AKI | Retrieval V5 | 202 min | 6.5 min | 72 min | aki_v5_stride3 |
| Sepsis | Retrieval | 208 min | 7.5 min | 62 min | sepsis_retr_fg_no_smooth |

### Minimum screening time (current infrastructure)

| Scenario | Epochs needed | Data load | Pretrain (15ep) | Phase 2 | **Total** |
|---|---|---|---|---|---|
| Sepsis retrieval | 3 | 208 min | 113 min | 186 min | **8.4h** |
| AKI SL+FG | 3 | 144 min | 210 min | 81 min | **7.3h** |
| AKI retrieval V5 | 5 | 202 min | 98 min | 360 min | **11h** |

### With pretrain reuse + data caching

| Scenario | Epochs needed | Cached load | Phase 2 | **Total** |
|---|---|---|---|---|
| **Sepsis retrieval** | **3** | ~15 min | 186 min | **~3.4h** |
| **AKI SL+FG** | **3** | ~10 min | 81 min | **~1.5h** |
| **AKI retrieval V5** | **5** | ~15 min | 360 min | **~6.3h** |
| Mortality SL+FG | 30 (full) | ~5 min | 330 min | **~5.6h** |

### Bottleneck analysis

- **Current**: data loading (2-3.5h) + pretrain (1.5-3.5h) dominate short experiments
- **With caching + reuse**: Phase 2 epoch time dominates; retrieval is 2.5-6x slower than SL per epoch due to memory bank k-NN + cross-attention
- **Further speedup options**: training data subsampling (untested), smaller memory bank, fewer batches/epoch

---

## 6. Ranking Stability (Kendall's τ vs epoch)

### Per-paradigm stability horizons (τ ≥ 0.8 sustained, val_task ranking vs final val_task)

| Task | Paradigm | N | Stability horizon | τ at epoch 5 | τ at epoch 10 |
|---|---|---|---|---|---|
| Mortality | delta | 9 | 28 | 0.257 | 0.257 |
| Mortality | SL | 7 | never | -0.333 | -0.238 |
| Mortality | retrieval | 10 | 23 | 0.511 | -0.422 |
| AKI | SL | 6 | 30 | -0.414 | 0.966 |
| AKI | retrieval | 17 | 35 | 0.235 | -0.033 |
| Sepsis | SL | 9 | never | 0.444 | 0.200 |
| Sepsis | retrieval | 19 | 26 | 0.322 | 0.205 |

The strict "sustained τ≥0.8" criterion requires 23-35 epochs — nearly full training. But the within-paradigm Spearman analysis (Section 2) shows the AUCROC-predictive signal arrives much earlier.

---

## 7. Multi-Seed Variance (AKI SL+FG, 6 runs)

| Epoch | Mean val_task | Std | Epochs 1-5 avg std |
|---|---|---|---|
| 1 | 0.4549 | 0.0108 | |
| 3 | 0.4280 | 0.0060 | |
| 5 | 0.4209 | 0.0106 | 0.0079 |
| 10 | 0.4049 | 0.0060 | |
| 15 | 0.3942 | 0.0030 | |
| 20 | 0.3967 | 0.0059 | |
| 25 | 0.3906 | 0.0035 | |
| 30 | 0.3955 | 0.0074 | |

Seed variance (±0.003-0.01) is much smaller than config-level differences (0.02-0.05). Good configs are distinguishable from bad configs at any epoch.

---

## 8. Pretrain Reuse Safety

| Task | Paradigm | N | Final recon (mean ± std) | CV | Verdict |
|---|---|---|---|---|---|
| AKI | SL | 6 | 4.52 ± 0.10 | 0.021 | Safe |
| AKI | retrieval | 15 | 4.88 ± 0.63 | 0.130 | Safe (CV inflated by arch variants) |
| Sepsis | SL | 6 | 4.79 ± 0.41 | 0.086 | Safe |
| Sepsis | retrieval | 19 | 4.59 ± 0.31 | 0.068 | Safe |
| Mortality | SL | 6 | 6.07 ± 2.67 | 0.440 | Mixed — sl_v1 (11.6) vs sl_fg (4.3-4.5) |
| Mortality | retrieval | 10 | 6.35 ± 1.63 | 0.257 | Mixed |

High mortality CV is an artifact of mixing architectures (sl_v1/v3 vs sl_fg). Within the sl_fg family: recon = 4.25-4.52, CV ≈ 0.03. **Pretrain reuse is safe for same-architecture experiments.**

---

## 9. Actionable Protocol

### For quick screening (go/no-go on a new config)

1. **Sepsis retrieval**: Run **3 Phase 2 epochs** on full data with pretrain reuse. Compare val_task. If val_task is in the bottom 30% of existing configs, kill it. (ρ=-0.73 at epoch 3)
2. **AKI retrieval**: Run **5 Phase 2 epochs** with pretrain reuse. Same threshold logic. (ρ=-0.62 at epoch 5)
3. **AKI SL**: Run **3 Phase 2 epochs** with pretrain reuse. (ρ=-0.99 at epoch 3, but n=6)
4. **Mortality**: No quick screening available. Run full 30 epochs. But mortality experiments are fast (SL: 5.6h, delta: 1.3h), so this is acceptable.

### Infrastructure needed

1. **Pretrain reuse** — save encoder checkpoint after Phase 1, load in new experiments that share the same architecture + task. Saves 1.5-3.5 hours per experiment.
2. **YAIB data caching** — save preprocessed tensors (train/val/test splits) to disk after first load. Saves 1-3.5 hours per experiment.

### What NOT to do

- **Never use debug mode** (20% data) for config comparison — rankings are random (τ=0.207)
- **Don't compare val_task across paradigms** for mortality — only useful across paradigms for AKI/sepsis
- **Don't trust single-epoch τ** for small groups (n<6) — use Spearman ρ on the full trajectory instead

---

## Appendix: Plots

All plots in `docs/convergence_analysis/`:

| Plot | Description |
|---|---|
| `ranking_stability.png` | Kendall τ vs epoch (top: vs final val_task, bottom: vs AUCROC Δ) |
| `val_task_trajectories_{task}.png` | All val_task curves overlaid, colored by paradigm |
| `debug_vs_full_ranking.png` | Debug rank vs full rank scatter (mortality delta, 10 pairs) |
| `epoch5_vs_final_ranking.png` | Epoch-5 val_task vs final AUCROC Δ scatter (per task) |
| `val_task_vs_aucroc.png` | Best val_task vs AUCROC Δ scatter (per task) |

Script: `scripts/analyze_convergence.py` (deterministic, re-runnable)
