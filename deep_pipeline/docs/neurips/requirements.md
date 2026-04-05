# NeurIPS 2026 Submission — Implementation Requirements

**Abstract deadline**: May 4, 2026 | **Paper deadline**: May 6, 2026
**Notification**: Sep 24, 2026 | **Conference**: Dec 6–12, 2026

---

## Decision: Show All 3 Paradigms (Delta + SL + Retrieval)

**YES** — all 3 paradigms enrich the paper:
1. **Gradient alignment analysis** (core theoretical insight) *requires* comparing paradigms
2. **Scientific narrative**: delta → SL → retrieval is a genuine conceptual progression
3. **Pre-empts reviewer Q**: "Why not simpler approaches?" — answered with evidence
4. **FeatureGate is cross-paradigm** — validates it as a general contribution
5. **Frame**: "We systematically study three translation paradigms, discovering retrieval uniquely satisfies the gradient alignment condition across all task types."

---

## Group 1: DA Baselines Implementation (CRITICAL — Reject-level gap)

**Priority**: P0 — Without this, acceptance probability drops to 20–30%.
**Effort**: 3–5 days | **Impact**: CRITICAL
**Depends on**: Group 6 (architecture stable)

### What
Implement DANN, CORAL, and CoDATS adapted to our frozen-model setting. These are the baselines every DA reviewer will expect.

### Exact Requirements

1. **DANN (Domain-Adversarial Neural Network)**
   - Learn a feature transform `T(x)` trained with domain adversarial loss
   - Gradient reversal layer + domain discriminator on LSTM hidden states
   - The frozen LSTM serves as a fixed feature extractor; `T(x)` transforms inputs before the frozen LSTM
   - Implementation: `src/baselines/dann.py`

2. **Deep CORAL**
   - Minimize second-order statistics (covariance) difference of LSTM hidden states between transformed source and target
   - Same frozen-LSTM setting: learn `T(x)` that minimizes CORAL loss + task loss
   - Implementation: `src/baselines/coral.py`

3. **CoDATS-style**
   - 1D CNN encoder + gradient reversal adapted to our frozen-LSTM setting
   - The CNN transforms input time-series before the frozen LSTM
   - Implementation: `src/baselines/codats.py`

4. **Shared infrastructure**
   - All baselines must use identical data splits, frozen LSTM, and evaluation protocol as our translators
   - Integrate with existing `YAIBRuntime` for data loading and evaluation
   - Same multi-seed protocol (at least 2 seeds per baseline per task)
   - Implementation: `src/baselines/base.py` (shared frozen-LSTM wrapper)

5. **Experiments**
   - Run on all 3 classification tasks (Mortality, AKI, Sepsis)
   - Report AUROC, AUCPR, and calibration metrics (Brier, ECE)
   - Configs: `configs/baselines/dann_mortality.json`, etc.

### Output
Results table comparing DA baselines vs our 3 paradigms across all metrics.

### Key Design Decision
The frozen-LSTM constraint means standard DANN/CORAL (which backprop through the feature extractor) must be adapted. Our baselines learn an input transformation `T(x)` that precedes the frozen LSTM — making the comparison fair and highlighting the frozen-model setting as a genuine constraint.

---

## Group 2: HiRID Dataset Integration — DONE (queued, awaiting results)

**Priority**: P1 — Addresses "single domain pair" reviewer concern.
**Effort**: 3–4 days | **Impact**: High
**Status**: Data pipeline complete. 5 debug + 5 full experiments queued (2026-03-24). All 5 tasks: Mortality, AKI, Sepsis, LoS, KF.

### What
Add HiRID as a third source domain (HiRID → MIMIC-IV) to show generalization beyond eICU.

### Exact Requirements

1. **Data pipeline**
   - YAIB already supports HiRID — leverage existing preprocessing
   - Schema alignment: map HiRID features to the MIMIC-IV 48-feature space
   - Use `SchemaResolver` pattern from KF task (synthesize zeros for missing features)

2. **Frozen baseline**
   - Use existing MIMIC-IV trained LSTM (same frozen model as eICU experiments)
   - Evaluate HiRID data through frozen MIMIC LSTM to get "no translation" baseline

3. **Retrieval translator**
   - Run retrieval translator on HiRID → MIMIC for at least Mortality + AKI + Sepsis
   - Memory bank built from MIMIC-IV (same as eICU experiments)
   - Use task-adapted n_cross_layers (3 for AKI, 2 for mortality/sepsis)

4. **Results**
   - Report same metrics as eICU → MIMIC experiments (AUROC, AUCPR, Brier, ECE)
   - Show consistent improvement pattern across both source domains

### Output
Cross-domain results table: eICU→MIMIC vs HiRID→MIMIC, demonstrating generality.

### Data Access
HiRID requires PhysioNet credentialed access (same as MIMIC-IV/eICU). Must document in paper.

---

## Group 3: Post-Hoc Calibration (Temperature Scaling)

**Priority**: P1 — Low effort, high impact for calibration metrics.
**Effort**: 1 day | **Impact**: High

### What
Apply temperature scaling to all best models, generate reliability diagrams.

### Exact Requirements

1. **Temperature scaling implementation**
   - Platt scaling: learn single temperature parameter `T` on validation set
   - Optimize NLL on held-out validation split: `p_calibrated = sigmoid(logit / T)`
   - Script: `scripts/temperature_scaling.py`

2. **Apply to all best models**
   - Classification: Mortality (SL+FG), AKI (V5 cross3), Sepsis (V4+MMD)
   - Regression: LoS (V5 cross3), KF (V5 cross3) — use isotonic regression instead

3. **Metrics**
   - Report calibrated vs uncalibrated: Brier score, ECE (15 bins), MCE
   - Before/after translation comparison

4. **Reliability diagrams**
   - Generate calibration curves for each task
   - Show: (a) frozen baseline on source, (b) translated, (c) translated + temp scaled
   - Figure: 3×3 grid (3 tasks × 3 conditions) or 1×3 with overlaid curves

### Output
Calibration table + reliability diagram figures (PDF/PNG for LaTeX).

---

## Group 4: Statistical Completeness

**Priority**: P1 — Required for publication rigor.
**Effort**: 2–3 days (compute-bound) | **Impact**: High

### What
Bootstrap CIs for all headline results + multi-seed runs for V5 cross3.

### Exact Requirements

1. **Bootstrap CIs** (script ready: `scripts/bootstrap_ci.py`)
   - Run on best configs per task (all 5 tasks)
   - 2000 bootstrap replicates, 95% percentile CIs
   - Paired comparison (translated vs baseline)
   - Tasks: Mortality (SL+FG), AKI (V5 cross3), Sepsis (V4+MMD), LoS (V5 cross3), KF (V5 cross3)
   - Command: `python scripts/bootstrap_ci.py runs/<best_run>/ --paired`

2. **Multi-seed V5 cross3 AKI** (headline result)
   - Run 2–3 additional training seeds (current: seed 1337 only)
   - Report mean ± std across seeds
   - Seeds: 42, 7, 2024 (diversity)

3. **DeLong's test**
   - p-values for all pairwise comparisons (baseline vs translated)
   - Already implemented in bootstrap_ci.py — confirm running

4. **Cross-paradigm statistical comparison**
   - Paired bootstrap test: retrieval vs SL, retrieval vs delta
   - Show statistical significance of paradigm differences

### Output
Complete statistical tables: mean ± std, 95% CI, p-values for all headline results.

---

## Group 5: Reproducibility Package

**Priority**: P2 — Required for NeurIPS checklist but not paper acceptance.
**Effort**: 1–2 days | **Impact**: Medium

### What
NeurIPS checklist items 4–8: environment, data access, compute documentation.

### Exact Requirements

1. **Environment files**
   - `requirements.txt` with pinned versions (torch, numpy, pandas, scikit-learn, etc.)
   - `environment.yml` for conda
   - Test installation on clean environment

2. **Data access documentation**
   - PhysioNet credentialed access for eICU (v2.0), MIMIC-IV (v2.2), HiRID (v1.1.1)
   - Data Use Agreements required
   - YAIB preprocessing pipeline (reference their paper)

3. **Compute resources documentation**
   - GPU types: V100S (32GB), RTX A6000 (48GB), RTX 3090 (24GB)
   - Typical training times per experiment (50 epochs)
   - Total compute budget: ~82 experiments × average time

4. **Code anonymization**
   - Remove author names, institution references from code
   - Remove server hostnames/IPs from configs and scripts
   - Sanitize git history or create fresh anonymous repo

5. **License**
   - MIT License (permissive, standard for ML research)

6. **Reproduction guide**
   - `REPRODUCE.md` with step-by-step instructions
   - From data download → preprocessing → training → evaluation
   - Expected results (with tolerance ranges)

### Output
- `requirements.txt`, `environment.yml`
- `REPRODUCE.md`
- `LICENSE`
- Anonymized code zip

---

## Group 6: Architecture Finalization

**Priority**: P0 — Must complete before Groups 1–4.
**Effort**: Minimal (verification only) | **Impact**: Gate for other groups

### What
Confirm V5/V6 architecture is stable; lock down best config per task.

### Exact Requirements

1. **V6 features status check**
   - Self-retrieval Phase 1: implemented, committed (e368dd2)
   - LR scheduling (cosine/plateau): implemented, committed
   - Gradient clipping: implemented, committed
   - Gradient accumulation: implemented, committed
   - **Status**: All implemented. Need to decide which V6 features are used in final configs.

2. **Final n_cross_layers recommendation**
   - AKI: n_cross_layers=3 (V5 cross3: +0.0556 AUROC)
   - LoS: n_cross_layers=3 (V5 cross3: -0.0196 MAE)
   - KF: n_cross_layers=3 (V5 cross3: -0.0017 MAE)
   - Mortality: n_cross_layers=2 (V4: +0.0470 AUROC) — cross3 hurts (-0.0061)
   - Sepsis: n_cross_layers=2 (V4+MMD: +0.0512 AUROC) — cross3 hurts (-0.0064)

3. **V6 features decision**
   - Option A: Use V5 configs (proven results) as paper configs, mention V6 as extensions
   - Option B: Run V6 experiments and use those results
   - **Recommendation**: Option A — V5 results are strong and complete. V6 features mentioned in architecture section as available extensions.

4. **Lock final configs**
   - Create `configs/final/` directory with one config per task
   - These are the exact configs used for all paper results

### Output
Final architecture specification document + `configs/final/*.json` files.

---

## Group 7: Paper Writing Skeleton

**Priority**: P1 — Start early, iterate.
**Effort**: 2–3 days | **Impact**: High

### What
LaTeX paper skeleton with NeurIPS 2026 template, all sections, placeholder figures/tables.

### Exact Requirements

1. **Template setup**
   - NeurIPS 2026 LaTeX template (`neurips_2026.sty`)
   - Directory: `paper/`
   - Main file: `paper/main.tex`

2. **9-page structure**

   | Section | Pages | Content |
   |---------|-------|---------|
   | Abstract + Introduction | 1.25 | Frozen-model DA setting, contribution list |
   | Related Work | 0.75 | DA methods, clinical ML, retrieval models |
   | Problem Formulation | 0.75 | Formal definitions, loss decomposition, gradient alignment condition |
   | Method | 2.5 | Delta (0.5), SL (0.5), **Retrieval** (1.0), FeatureGate (0.25), Cross-domain norm (0.25) |
   | Experiments | 2.5 | Setup (0.5), Main results + DA baselines (0.5), Analysis (0.75), Ablations (0.75) |
   | Discussion + Limitations | 1.0 | When to use which method, calibration, limitations |
   | Conclusion | 0.25 | Summary + future work |

3. **Placeholder figures**
   - Fig 1: Architecture diagram (3 paradigms overview)
   - Fig 2: Gradient alignment plot (cos similarity across tasks)
   - Fig 3: Main results bar chart (AUROC deltas by paradigm × task)
   - Fig 4: Reliability diagrams (calibration curves)
   - Fig 5: t-SNE/UMAP of translated vs untranslated representations (optional)

4. **Placeholder tables**
   - Table 1: Main results (5 tasks × methods × metrics)
   - Table 2: DA baselines comparison
   - Table 3: Ablation study (loss components, architecture choices)
   - Table 4: Cross-domain generalization (eICU vs HiRID)
   - Table 5: Statistical significance (p-values, CIs)

5. **Special sections**
   - Related work (from `docs/neurips/related_work.md`)
   - Limitations section (single domain pair → multi, calibration, label density effects)
   - Ethics statement (frozen model auditability, de-identified data, bias risks)
   - NeurIPS checklist (16 items)

### Output
Compilable LaTeX skeleton in `paper/` directory.

---

## Group 8: Lightweight Theory

**Priority**: P2 — Differentiates from pure empirical paper.
**Effort**: 1–2 days | **Impact**: Medium-High

### What
Formalize the gradient alignment insight connecting to Ben-David et al.

### Exact Requirements

1. **Theoretical framework**
   - Define frozen-model translation setting formally
   - Source domain S, target domain T, frozen predictor f_T, translator g_θ
   - Task loss L_task(g_θ(x_S)), fidelity loss L_fid(g_θ(x_S), x_S)
   - Total gradient: ∇_θ L = ∇_θ L_task + λ · ∇_θ L_fid

2. **Gradient alignment condition**
   - Define: α(θ) = cos(∇_θ L_task, ∇_θ L_fid)
   - **Proposition**: Successful frozen-model translation requires α(θ) ≥ 0 on average
   - When α < 0: fidelity and task gradients fight → training instability, reduced performance
   - When α > 0: fidelity preserves task-relevant structure → stable convergence

3. **Connection to Ben-David et al.**
   - H-divergence d_H(S, T) bounded by fidelity loss (upper bound argument)
   - Fidelity loss → controls domain divergence of translated samples
   - When α < 0: reducing fidelity (to improve task) increases divergence → violates bound
   - This explains why lambda_fidelity=0 causes catastrophic collapse

4. **Empirical validation**
   - cos(task, fidelity) measurements: Mortality +0.84, AKI moderate, Sepsis -0.21
   - Correlates with method difficulty ordering
   - Retrieval resolves negative alignment by providing an alternative information path (cross-attention bypasses the gradient conflict)

### Output
0.5–0.75 page theory section in LaTeX, with one Proposition.

---

## NeurIPS Checklist Status

| # | Item | Status | What's Needed | Group |
|---|------|--------|---------------|-------|
| 1 | Claims match contributions | Ready | Write abstract carefully | 7 |
| 2 | Limitations section | Needs writing | Domain pair limitation, calibration, label density | 7 |
| 3 | Theory/Proofs | Partial | Lightweight Ben-David connection | 8 |
| 4 | Reproducibility | Partial | requirements.txt, clear instructions | 5 |
| 5 | Open access code | Partial | Anonymize repo, add license | 5 |
| 6 | Experimental details | Ready | Already comprehensive in configs | — |
| 7 | Statistical significance | Ready (pending Group 4) | Bootstrap CIs for all results | 4 |
| 8 | Compute resources | Needs documenting | GPU types, memory, training times | 5 |
| 9 | Code of ethics | N/A | Confirm compliance | — |
| 10 | Broader impacts | Needs writing | Clinical safety, frozen model benefits | 7 |
| 11 | Safeguards | N/A | Not high-risk | — |
| 12 | Licenses | Needs checking | YAIB, eICU, MIMIC-IV, HiRID licenses | 5 |
| 13 | New assets | Needs writing | Document code release | 5 |
| 14 | Human subjects | N/A | Retrospective de-identified data | — |
| 15 | IRB approvals | Check | PhysioNet credentialed access | 5 |
| 16 | LLM declaration | Required | Declare Claude usage for code/analysis | 7 |

---

## Execution Timeline

```
Week 1 (Mar 21–27): Group 6 (lock architecture) + Group 4 (bootstrap CIs, multi-seed)
Week 2 (Mar 28–Apr 3): Group 1 (DA baselines — start implementation)
Week 3 (Apr 4–10): Group 1 (DA baselines — run experiments) + Group 3 (temp scaling)
Week 4 (Apr 11–17): Group 2 (HiRID integration)
Week 5 (Apr 18–24): Group 7 (paper skeleton) + Group 8 (theory)
Week 6 (Apr 25–May 1): Group 5 (reproducibility) + writing
Week 7 (May 1–4): Final writing, abstract submission
Week 8 (May 4–6): Paper finalization and submission
```

---

## Acceptance Probability Estimates

| Scenario | Probability | What's Included |
|----------|-------------|-----------------|
| As-is (no DA baselines) | 20–30% | Strong results but missing standard comparisons |
| + DA baselines + temp scaling | 40–45% | Addresses biggest reviewer concern |
| + DA baselines + HiRID + multi-seed | 50–55% | Multi-domain + statistical rigor |
| + DA baselines + HiRID + multi-seed + theory | 55–65% | Full package |
