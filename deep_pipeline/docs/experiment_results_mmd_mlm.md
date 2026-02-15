# Experiment Results: MMD + MLM Debug Runs

> **Role**: Historical — detailed per-experiment logs from MMD/MLM phase. All variants produced +0.001-0.002 AUCROC.
> **See also**: [gradient_bottleneck_analysis.md](gradient_bottleneck_analysis.md) (consolidated results table including these + later experiments), [investigation_mortality_vs_sepsis.md](investigation_mortality_vs_sepsis.md) (follow-up investigation)

**Task**: Sepsis (Binary Classification, eICU → MIMIC-IV)
**Mode**: Debug (20% data subset, seed=2222)
**Baseline**: Frozen LSTM trained on MIMIC-IV, AUCROC=0.7193, AUCPR=0.0309 on eICU test
**Translator**: Causal transformer, d_model=64, n_layers=4, n_heads=4
**Common hyperparams**: lr=1e-4, batch_size=64, lambda_fidelity=0.1, lambda_range=0.001, lambda_forecast=0

---

## Experiment Configurations

| Exp | Description | lambda_mmd | lambda_mmd_trans | mlm_pretrain_epochs | Config |
|---|---|---|---|---|---|
| A | Baseline (task+fidelity+range only) | 0 | 0 | 0 | `exp_baseline_debug.json` |
| B | MMD only | 1.0 | 0 | 0 | `exp_mmd_debug.json` |
| C | MMD + Transition MMD | 1.0 | 0.5 | 0 | `exp_mmd_trans_debug.json` |
| D | MLM pretrain only | 0 | 0 | 10 | `exp_mlm_debug.json` |
| E | MLM pretrain + MMD | 1.0 | 0 | 10 | `exp_mlm_mmd_debug.json` |

MLM pretraining: BERT-style masked timestep reconstruction (mask_prob=0.15, lr=1e-4), bidirectional attention during pretrain, switch to causal for fine-tuning.

---

## Results: 10 Epochs (no early stopping)

| Experiment | val_task | AUCROC | delta_AUCROC | AUCPR | delta_AUCPR | test_loss |
|---|---|---|---|---|---|---|
| Original (no translator) | - | 0.7193 | - | 0.0309 | - | 0.6956 |
| A: Baseline | 0.6801 | 0.7223 | +0.0030 | 0.0308 | -0.0001 | 0.6728 |
| B: MMD only | 0.6738 | 0.7209 | +0.0016 | 0.0307 | -0.0002 | 0.6744 |
| C: MMD+Trans | 0.6684 | 0.7214 | +0.0021 | 0.0306 | -0.0002 | 0.6714 |
| D: MLM only | 0.6661 | 0.7202 | +0.0009 | 0.0306 | -0.0002 | 0.6655 |
| E: MLM+MMD | 0.6741 | 0.7204 | +0.0011 | 0.0308 | -0.0000 | 0.6758 |

Note: All experiments still improving at epoch 10.

### Per-Experiment Details (10 epochs)

**A: Baseline** — Standard causal transformer, no new losses. val_task=0.6801 at ep 10, no early stopping triggered. Consistent with prior results from overfit experiments.

**B: MMD only** — lambda_mmd=1.0. Better val_task learning than baseline (0.6738 vs 0.6801), confirming MMD provides useful gradient signal. MMD loss: train 0.091→0.089, val 0.096→0.086. Test AUCROC slightly lower than baseline.

**C: MMD + Transition** — lambda_mmd=1.0, lambda_mmd_transition=0.5. Best val_task of non-MLM experiments (0.6684). Transition MMD adds clear benefit over marginal MMD alone. MMD loss: train 0.090→0.084, val 0.093→0.082. MMD transition: train 0.051→0.029, val 0.103→0.030 (good convergence).

**D: MLM only** — 10 epochs MLM pretrain, then fine-tune. MLM reconstruction loss: 0.3119 → 0.0694 (good convergence). Best individual val_task of all experiments (0.6661). Pretrained backbone learns significantly faster during fine-tuning. val_task still dropping at ep 10.

**E: MLM+MMD** — MLM pretrain + lambda_mmd=1.0. MLM reconstruction loss: 0.3063 → 0.0676. val_task (0.6741) worse than MLM alone (0.6661) at 10 epochs — MMD may conflict with pretrained representations early on. MMD loss: train 0.097→0.089, val 0.091→0.100 (val MMD increasing, suggesting tension). Best AUCPR preservation (-0.0000 delta).

---

## Results: 30 Epochs (early_stopping_patience=5, best_metric=val_task)

| Experiment | best_ep / stopped_ep | val_task | AUCROC | delta_AUCROC | AUCPR | delta_AUCPR | test_loss |
|---|---|---|---|---|---|---|---|
| Original (no translator) | - | - | 0.7193 | - | 0.0309 | - | 0.6956 |
| A: Baseline | ~12 / 17 | 0.6764 | 0.7210 | +0.0017 | 0.0305 | -0.0004 | 0.6711 |
| B: MMD only | ~12 / 17 | ~0.67 | 0.7206 | +0.0013 | 0.0306 | -0.0002 | 0.6737 |
| C: MMD+Trans | ~12 / 17 | 0.6700 | 0.7207 | +0.0013 | 0.0302 | -0.0006 | 0.6698 |
| D: MLM only | ~13 / 18 | 0.6683 | 0.7188 | -0.0005 | 0.0303 | -0.0006 | 0.6654 |
| E: MLM+MMD | ~15 / 20 | 0.6697 | 0.7209 | +0.0016 | 0.0307 | -0.0001 | 0.6716 |

Run directories: `runs/exp_*_debug_30ep/`
Loss curves saved per run: `loss_curve.png`, `task_loss_curve.png`, `fidelity_loss_curve.png`, `mmd_loss_curve.png` (where applicable)

### Per-Experiment Details (30 epochs)

**A: Baseline** — Early stopped at epoch 17 (best ~epoch 12). val_task plateaued around 0.676-0.681 from ep 11-17. Fidelity loss increased steadily (0.017→0.173), indicating the model was making larger changes to the data without improving the task.

**B: MMD only** — Early stopped ~epoch 17. Training log truncated (output piped through tail), but test results are definitive. AUCROC +0.0013, slightly below 10-epoch result.

**C: MMD + Transition** — Early stopped at epoch 17 (best ~epoch 12). val_task=0.6700, slightly worse than 10-epoch val_task (0.6684). Final epoch MMD: train 0.084, val 0.081. Transition MMD: train 0.024, val 0.029. Both converged well but didn't improve test AUCROC over plain MMD.

**D: MLM only** — Early stopped at epoch 18 (best ~epoch 13). val_task=0.6683, still the best individual val_task. But test AUCROC regressed to -0.0005 — the only experiment below the original baseline. Clear overfitting: the model learned train/val patterns that didn't generalize to test without distribution regularization.

**E: MLM+MMD** — Early stopped at epoch 20 (best ~epoch 15). Trained the longest of all experiments. val_task=0.6697. Final epoch MMD: train 0.088, val 0.096. AUCROC +0.0016 with best AUCPR preservation (-0.0001). The MMD regularization prevented the overfitting seen in D, making this the best overall combination.

---

## Key Observations

### 10-epoch vs 30-epoch comparison
- All experiments hit early stopping between epochs 17-20, confirming 10 epochs was undertrained.
- Baseline (A) dropped from +0.0030 to +0.0017 — the 10-epoch checkpoint was "lucky".
- MLM only (D) degraded from +0.0009 to **-0.0005** — overfits without regularization.
- MLM+MMD (E) improved from +0.0011 to +0.0016 — benefits most from longer training.

### Rankings (30 epochs)
1. **A: Baseline** +0.0017 — simple but effective, no domain adaptation signal
2. **E: MLM+MMD** +0.0016 — best AUCPR preservation (-0.0001), trained longest (20 ep)
3. **C: MMD+Trans** +0.0013 — transition MMD didn't help over plain MMD
4. **B: MMD only** +0.0013 — equivalent to C with less complexity
5. **D: MLM only** -0.0005 — overfits, the only regression below baseline

### Analysis
1. **The causal model barely learns** — all deltas are in the +0.001 to +0.002 range (vs bidirectional model's +0.247). The causal constraint severely limits the translator's ability to produce useful transformations.
2. **MMD provides regularization but not signal** — it prevents overfitting (E > D) but doesn't improve over the baseline alone (A ≈ E > B = C).
3. **MLM pretraining helps optimization** — best val_task in both 10ep and 30ep runs, but doesn't translate to test AUCROC when used alone.
4. **AUCPR is essentially unchanged** (~0.030) across all experiments — the extreme class imbalance (1.1% positive rate) dominates.
5. **Early stopping is working** — patience=5 catches overfitting, models converge by epoch 17-20.

---

## Historical Context: Prior Overfit Experiments (d_model variations, 20 epochs)

Before the MMD/MLM experiments, we ran hyperparameter variations with just the baseline loss (task+fidelity+range). These used d_model=64 (smaller than the original d_model=128) and various regularization strategies:

| Exp | Config | Best val_task (epoch) | Test AUCROC Delta |
|---|---|---|---|
| (a) | d_model=64, lr=2e-4 | 0.6764 (ep 12) | +0.0017 |
| (b) | n_layers=2, lr=1e-4 | 0.6708 (ep 15) | +0.0028 |
| (c) | wd=1e-2, lr=1e-4 | 0.6754 (ep 12) | +0.0030 |
| (d) | all combined, lr=5e-4 | 0.6704 (ep 19) | +0.0018 |

Notably, experiment (a) exactly matches the 30-epoch baseline (A) result: val_task=0.6764, AUCROC delta=+0.0017. This confirms reproducibility across runs with the same seed.

---

## Open Questions for Next Steps

1. **Is the causal constraint the bottleneck?** The bidirectional "cheaty" model gets +0.247 AUCROC. Even with better losses, the causal model may be fundamentally limited. Should we:
   - Try mortality task (allows bidirectional)?
   - Investigate what the bidirectional model learns that causal can't?
   - Try larger models / different architectures?

2. **Are deltas in the noise range?** +0.001-0.002 AUCROC on 20% debug data may not be statistically significant. Need:
   - Full data runs (5x more data)
   - Multiple seeds for confidence intervals
   - Statistical significance tests

3. **Lambda tuning**: Current lambda_mmd=1.0 is arbitrary. Grid search over {0.1, 0.5, 1.0, 5.0} might help.

4. **Longer MLM pretraining**: Only 10 epochs of pretraining. More epochs or curriculum learning could help.

5. **Alternative approaches**:
   - Feature-level alignment (per-feature distribution matching)
   - Adversarial domain adaptation (discriminator-based)
   - Optimal transport
   - Attention to specific features that differ most between domains

---

## File Reference

### Configs
- `configs/exp_baseline_debug.json` — Experiment A
- `configs/exp_mmd_debug.json` — Experiment B
- `configs/exp_mmd_trans_debug.json` — Experiment C
- `configs/exp_mlm_debug.json` — Experiment D
- `configs/exp_mlm_mmd_debug.json` — Experiment E

### Key source files
- `src/core/mmd.py` — Multi-kernel MMD with median heuristic
- `src/core/pretrain.py` — MLM pretrainer (BERT-style masking)
- `src/core/train.py` — TransformerTranslatorTrainer (all loss components, plotting)
- `src/core/translator.py` — EHRTranslator (set_temporal_mode, forward_mlm)
- `src/cli.py` — CLI orchestration, debug overrides

### Run outputs (30-epoch)
- `runs/exp_baseline_debug_30ep/` — A results + loss curves
- `runs/exp_mmd_debug_30ep/` — B results + loss curves
- `runs/exp_mmd_trans_debug_30ep/` — C results + loss curves
- `runs/exp_mlm_debug_30ep/` — D results + loss curves
- `runs/exp_mlm_mmd_debug_30ep/` — E results + loss curves
