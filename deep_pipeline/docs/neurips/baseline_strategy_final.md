# FINAL Baseline Strategy for NeurIPS 2026

**Date**: March 26, 2026 (updated March 31 with complete E2E results + padding discovery)
**Status**: All baselines COMPLETE. Ready for paper writing.
**Deadline**: May 6, 2026 (5 weeks)

---

## 0. Executive Summary (Updated March 31)

**All baseline experiments are complete.** ~170 experiments across 7 DA methods, 2 evaluation protocols, multiple HP variants per method, 3 classification tasks.

**Critical discovery (Mar 29-31):** Left-padding inflates per-timestep AUROC for ALL architectures (LSTM, TCN, CNN). YAIB uses right-padding. All E2E baselines were re-run with right-padding for fair comparison. Left-padded results (including CLUDA sepsis +0.119 that appeared to beat our translator) are INVALID for comparison.

**Final story**: Under fair evaluation (right-padding), all E2E DA baselines provide marginal improvement (+0.001 to +0.023 AUROC). Our translator outperforms by 2-4x on every task.

---

## 0b. The Core Problem and Its Resolution

We face a genuine apples-to-oranges comparison problem. Every existing TS-DA method (DANN through ACON) assumes end-to-end training with a shared, learnable feature extractor. Our method freezes the predictor entirely and transforms inputs. These are structurally different optimization problems.

**Resolution: A two-tier comparison that makes the argument layered.**

The paper must tell this story in sequence:

1. **The frozen-model constraint is real and hard** (Tier 1 baselines show degradation under it)
2. **Our method solves it** (Tier 2 shows we beat even unconstrained methods)
3. **The gap widens with task difficulty** (sepsis: largest gap)
4. **E2E methods provide only marginal improvement** even without the frozen constraint

This is not a weakness -- it IS the contribution. The fact that no prior method addresses this setting is what makes the paper novel.

---

## 1. Primary Baselines (MUST HAVE in Table 1)

Table 1 should be the main results table: 5 tasks x all methods x AUROC + AUCPR (classification) or MAE + R2 (regression).

### 1a. Reference Points (no implementation needed)

| Row | What | Status |
|-----|------|--------|
| Frozen baseline (no adaptation) | MIMIC LSTM on raw eICU/HiRID | **Done** |
| eICU-native LSTM | YAIB reference, trained on eICU | **Done** (from paper) |
| MIMIC-native LSTM | Oracle upper bound | **Done** (from paper) |
| Statistics-only (target normalization, no translator) | Affine renorm only | **Done** (hurts all tasks: -0.007 to -0.011) |

### 1b. Frozen-Model DA Baselines (Option B -- same constraint as us)

These use our frozen LSTM + a learned input transformation, differing only in the training objective. This is the fair, same-constraint comparison.

| Method | Training Objective | Architecture | Status | Results (Mort/AKI/Sep AUROC Δ) |
|--------|-------------------|--------------|--------|-------------------------------|
| DANN | Task + GRL domain discriminator | EHRTranslator backbone | **Done** | +0.036/+0.025/+0.016 |
| Deep CORAL | Task + covariance alignment | EHRTranslator backbone | **Done** | +0.037/+0.031/+0.017 |
| CoDATS | Task + GRL (1D CNN backbone) | CNN backbone | **Done** | +0.035/+0.013/-0.004 |

**CDAN justification**: It is the single most-cited adversarial DA method after DANN. Every top TS-DA paper (ACON, RAINCOAT, CLUDA, AdaTime) includes CDAN. Not having it is an obvious gap. Since we already have DANN infrastructure, CDAN requires only conditioning the discriminator on classifier predictions -- a few hours of work.

**Linear probe justification**: A skeptical reviewer will ask "does this need a transformer?" A learned per-feature affine map (96 or 100 parameters for classification tasks) is the minimal learnable baseline. If it works surprisingly well, that is an interesting finding. If it does not, it validates the need for capacity.

### 1c. Our Methods (the contribution)

| Method | Paradigm | Status |
|--------|----------|--------|
| Delta translator (EHRTranslator) | Input-space delta | **Done** |
| Shared latent translator (SL) | Encode-decode | **Done** |
| Shared latent + FeatureGate (SL+FG) | Encode-decode + per-feature gating | **Done** |
| Retrieval translator (best per task) | Memory bank + cross-attention | **Done** |

**Recommended Table 1 layout**: 11 rows (4 reference + 5 frozen-model baselines + 2 best of our methods) x 5 task columns. SL and SL+FG can be merged into one row ("SL") in the main table with the FG ablation in Table 3.

---

## 2. Secondary Baselines (Appendix / Table 2)

### 2a. End-to-End DA Methods (Option A -- unconstrained) — ALL COMPLETE

All E2E methods implemented and run with right-padding (fair comparison). Multiple HP variants per method.

**CRITICAL: All E2E methods use right-padding** to match YAIB convention. Left-padded results are excluded (inflated per-timestep AUROC by +0.03 to +0.21 depending on architecture).

| Method | Backbone | Venue | Status | Best Results (Mort/AKI/Sep AUROC Δ) |
|--------|----------|-------|--------|--------------------------------------|
| Source-only | 2L LSTM | — | **Done** | +0.017/+0.013/+0.006 |
| DANN | 2L LSTM | JMLR 2016 | **Done** (3 HP variants) | +0.019/+0.012/+0.023 |
| CORAL | 2L LSTM | ECCV 2016 | **Done** (2 HP variants) | +0.020/+0.014/+0.018 |
| CoDATS | Causal CNN | KDD 2020 | **Done** | +0.010/-0.018/-0.007 |
| CLUDA | TCN | ICLR 2023 | **Done** | +0.009/-0.008/-0.018 |
| RAINCOAT | CNN+Spectral | ICML 2023 | **Done** | +0.005/+0.019/+0.021 |
| ACON | CNN temporal | NeurIPS 2024 | **Done** | +0.018/-0.114/+0.015 |

**Key finding**: All E2E methods provide marginal improvement (+0.001 to +0.023). DA alignment provides zero benefit over source-only (within noise). Our translator outperforms all by 2-4x.

### 2b. Fine-Tuned Upper Bound

| Method | What | Status |
|--------|------|--------|
| Fine-tuned LSTM | Unfreeze MIMIC LSTM, train on source with target pretrain | **Done** |

This answers: "what if you just fine-tune?"

---

## 3. What to DROP

| Current Baseline | Recommendation | Reason |
|-----------------|----------------|--------|
| CoDATS (in main table) | **Move to appendix** | CNN backbone confounds loss vs architecture. DANN and CORAL already use EHRTranslator backbone, isolating the loss difference. CoDATS adds the variable of a weaker backbone, making it hard to interpret. Keep for completeness in appendix. |
| MMD standalone / OT-DA | **Do not implement** | We already have MMD as a loss component (V4). Adding standalone MMD and OT baselines creates method proliferation without clear insight. Reference them in related work. |
| CoTMix, AdvSKM | **Do not implement** | Marginal value. Neither has been tested on clinical data. Time better spent on CLUDA. |

---

## 4. What to ADD (Critical for Acceptance)

Ordered by criticality:

### MUST ADD (reject-level risk if missing)

1. **Statistics-only baseline** (target normalization without any translator)
   - Answers: "Is a neural translator even needed?"
   - Effort: 30 minutes. Run existing pipeline with `use_target_normalization: true` and a no-op translator (identity function or zero-delta).
   - Impact: Foundational. Every reviewer will wonder this.

2. **CDAN** (conditional adversarial)
   - Answers: "Did you try better adversarial methods than DANN?"
   - Effort: 3-4 hours. Modify existing DANN discriminator to condition on classifier predictions.
   - Impact: Completes the adversarial family. Standard in every TS-DA paper.

3. **Linear probe** (per-feature affine)
   - Answers: "Does this need a deep model?"
   - Effort: 0.5 day. Single linear layer, same training loop.
   - Impact: Establishes that nonlinear capacity is necessary.

### SHOULD ADD (strengthens paper significantly)

4. **CLUDA** (contrastive TS-DA, end-to-end)
   - Answers: "How do you compare to the best clinical TS-DA method?"
   - Effort: 2-3 days. Adapt published code to YAIB pipeline.
   - Impact: Only prior method tested on clinical ICU data. Most likely reviewer ask.

5. **Fine-tuned LSTM upper bound**
   - Answers: "What does the frozen constraint cost?"
   - Effort: 1 day. Unfreeze baseline, retrain on eICU.
   - Impact: Frames the frozen-model gap. Essential for the narrative.

### NICE TO HAVE (appendix material)

6. **RAINCOAT** (frequency-aware, end-to-end)
   - Only if CLUDA implementation goes smoothly and there is time.
   - Effort: 2-3 days.

---

## 5. Implementation Priority (Execution Order)

Total estimated effort: 6-8 working days of implementation + 5-7 days of GPU time.

### Phase 1: Quick Wins (March 27-28, 2 days)

| # | Task | Effort | GPU Time | Blocking? |
|---|------|--------|----------|-----------|
| 1 | Statistics-only baseline (5 tasks) | 30 min impl + 2h runs | 2h | No |
| 2 | CDAN implementation + debug | 3-4h impl | 0 | No |
| 3 | CDAN experiments (3 classification tasks) | 0 (queue) | 12-15h | No |
| 4 | Linear probe implementation + debug | 4h impl | 0 | No |
| 5 | Linear probe experiments (3 classification tasks) | 0 (queue) | 8-10h | No |

**Deliverable**: 3 new baselines x 3 tasks = 9 new data points for Table 1.

### Phase 2: CLUDA Adaptation (March 29 - April 2, 4 days)

| # | Task | Effort | GPU Time |
|---|------|--------|----------|
| 6 | Clone CLUDA repo, study architecture | 3h | 0 |
| 7 | Write YAIB data adapter for CLUDA | 4h | 0 |
| 8 | Adapt CLUDA for our feature space (48-292 features, variable length) | 6h | 0 |
| 9 | Debug run (1 epoch, mortality) | 1h | 30 min |
| 10 | Full CLUDA experiments (3 classification tasks, 2 seeds) | 0 (queue) | 24-36h |

**Deliverable**: CLUDA numbers for 3 classification tasks. If CLUDA wins on any task, that is fine -- it has more degrees of freedom and is not frozen.

### Phase 3: Fine-Tuned Upper Bound (April 3, 1 day)

| # | Task | Effort | GPU Time |
|---|------|--------|----------|
| 11 | Implement fine-tuning mode (unfreeze LSTM, source training) | 2h | 0 |
| 12 | Run fine-tuned LSTM on 3 classification tasks | 0 (queue) | 12h |

**Deliverable**: Fine-tuning ceiling for all classification tasks.

### Phase 4: RAINCOAT (April 4-7, only if Phase 2 finishes on time)

| # | Task | Effort | GPU Time |
|---|------|--------|----------|
| 13 | Adapt RAINCOAT to YAIB pipeline | 8h | 0 |
| 14 | Run RAINCOAT on 3 classification tasks | 0 (queue) | 24h |

**Deliverable**: RAINCOAT numbers. Skip if behind schedule -- CLUDA alone is sufficient.

### Phase 5: Regression Tasks for New Baselines (April 8-10)

| # | Task | Effort | GPU Time |
|---|------|--------|----------|
| 15 | Run stats-only, linear probe, CDAN on LoS + KF | 0 (queue) | 12h |
| 16 | Run CLUDA on LoS + KF (if CLUDA works) | 0 (queue) | 12h |

**Deliverable**: Complete 5-task coverage for new baselines.

---

## 6. The Narrative

The paper should frame the comparison as a three-layer argument:

### Layer 1: The Setting Is Novel (Section 1 + Section 3)

"Deployed clinical prediction models cannot be retrained when applied to new hospital systems. We formalize frozen-model domain adaptation: given a fixed predictor trained on target-domain data, learn an input-space translator that enables the predictor to perform well on source-domain data without modifying any predictor parameters."

Cite TATO (ICLR 2026) and SHOT (ICML 2020) as the closest prior work, explain why they differ (handcrafted transforms / partial freeze).

### Layer 2: Existing DA Methods Underperform Under This Constraint (Table 1)

"We adapt five standard DA methods (DANN, CDAN, Deep CORAL, CoDATS, linear probe) to operate under the frozen-model constraint. All methods use identical data splits, frozen predictor, and evaluation protocol. Table 1 shows that while these methods improve over the no-adaptation baseline, our retrieval translator achieves 27-207% larger improvements."

This is honest: the DA baselines DO help, but our method helps MORE. The advantage scales with task difficulty (sepsis: 1.13% positive rate, hardest task, largest gap).

### Layer 3: We Even Beat Unconstrained Methods (Table 2 / Section 5.3)

"For completeness, we compare against CLUDA (ICLR 2023), a contrastive TS-DA method previously evaluated on clinical ICU data, in its original unconstrained formulation (full pipeline trained end-to-end). Despite having strictly more degrees of freedom, CLUDA achieves [X] while our frozen-model translator achieves [Y]."

If CLUDA wins on some tasks: "CLUDA achieves competitive performance on [task] but requires retraining the predictor -- a luxury unavailable in our motivating deployment scenario."

### Layer 4: The Paradigm Progression Explains Why (Section 4)

"We systematically evaluate three translation paradigms (delta, shared-latent, retrieval) and show that retrieval uniquely resolves the gradient alignment conflict between task loss and fidelity loss. The gradient alignment analysis (Proposition 1) predicts which paradigm will succeed on which task, validated empirically."

### Handling Reviewer Objections

| Likely Objection | Pre-emptive Answer |
|-----------------|-------------------|
| "Why not fine-tune?" | Table 2 shows fine-tuning upper bound. Our translator closes X% of the gap without touching the predictor. In deployment, fine-tuning violates regulatory constraints. |
| "Why not CLUDA/RAINCOAT?" | Table 2 includes CLUDA end-to-end. It solves a different problem (unconstrained DA). Under our frozen constraint, it cannot be applied as designed. |
| "Your DA baselines are old (2016-2020)" | CDAN (2018) is still the most-cited adversarial DA method. We include it alongside DANN and CORAL. The key insight is that loss-function innovation (DANN vs CORAL vs CDAN) provides diminishing returns -- architectural innovation (retrieval + cross-attention) is needed. |
| "Statistics matching might be enough" | Table 1 row 1 shows statistics-only baseline. Neural translation adds [X] beyond normalization. |
| "Linear probe might be enough" | Table 1 shows linear probe. The nonlinear translator achieves [X] more, validating the need for a deep model. |
| "Single domain pair" | Table 4 shows HiRID->MIMIC with consistent improvement pattern across both source domains. |

---

## 7. GPU Budget

### Already Spent (not counted)

- 9 DA baseline experiments (DANN/CORAL/CoDATS x 3 tasks): ~45 GPU-hours on V100S
- All retrieval/SL/delta experiments: ~400+ GPU-hours across project

### New Budget Required

| Phase | Experiments | GPU-Hours (est.) | Hardware |
|-------|------------|-----------------|----------|
| Statistics-only (5 tasks) | 5 | 2 | Any GPU |
| CDAN (3 class + 2 reg) | 5 | 15 | V100S |
| Linear probe (3 class + 2 reg) | 5 | 10 | V100S |
| CLUDA adaptation (3 class, 2 seeds) | 6 | 36 | V100S or A6000 |
| Fine-tuned LSTM (3 class) | 3 | 12 | V100S |
| RAINCOAT (3 class, if time) | 3 | 24 | V100S or A6000 |
| Multi-seed runs for stats (3 seeds x 3 tasks) | 9 | 45 | Any GPU |
| Bootstrap CIs (5 tasks) | 5 | 2 | CPU-only |
| **Total (without RAINCOAT)** | **38** | **~122** | |
| **Total (with RAINCOAT)** | **41** | **~146** | |

At ~3 GPUs available (daytime policy), this is approximately:

- Without RAINCOAT: 122 / 3 = ~41 wall-clock GPU-hours = **~5 days of continuous GPU use** (fits within schedule)
- With RAINCOAT: 146 / 3 = ~49 wall-clock GPU-hours = **~7 days** (tight but feasible)

Many experiments can run in parallel across local V100S, A6000, and 3090 servers. Real wall-clock time is limited by the longest sequential chain (CLUDA adaptation), not total GPU-hours.

---

## 8. Final Table Structure for the Paper

### Table 1: Main Results (in paper body, ~0.5 page)

```
                          | Mortality  | AKI       | Sepsis    | LoS (MAE) | KF (MAE)
                          | AUROC      | AUROC     | AUROC     | MAE       | MAE
--------------------------|-----------|-----------|-----------|-----------|----------
No adaptation (frozen)    | 80.79     | 85.58     | 71.59     | 42.5h     | 0.403
Statistics-only (norm)    | 79.80     | 84.46     | 70.87     |   —       |  —
DANN (frozen-model)       | 84.34     | 88.08     | 73.23     |   —       |  —
Deep CORAL (frozen-model) | 84.53     | 88.66     | 73.26     |   —       |  —
Delta translator (ours)   | 84.12     | 88.00     | 73.09     |   —       |  —
SL + FeatureGate (ours)   | 85.55     | 90.82     |   —       |   —       |  —
Retrieval translator (ours)| 85.49    | 91.14     | 76.71     | 39.2h     | 0.382
--------------------------|-----------|-----------|-----------|-----------|----------
eICU-native LSTM (ref.)   | 85.5      | 90.2      | 74.0      | 39.2h     | 0.28
MIMIC-native LSTM (oracle) | 86.7     | 89.7      | 82.0      | 40.6h     | 0.28
```

### Table 2: E2E DA Baselines (right-padded, in paper body or appendix)

```
                          | Mortality  | AKI       | Sepsis    | Constraint
--------------------------|-----------|-----------|-----------|------------
Source-only (no DA)       | 81.76     | 86.86     | 72.17     | None (MIMIC-trained LSTM)
DANN (E2E, paper HPs)    | 82.73     | 86.82     | 73.84     | None (full pipeline)
CORAL (E2E, λ=0.1)       | 82.78     | 86.94     | 73.41     | None (full pipeline)
RAINCOAT (E2E)            | 81.28     | 87.51     | 73.67     | None (full pipeline)
CLUDA (E2E, lr=5e-4)     | 81.66     | 85.77     | 69.77     | None (full pipeline)
ACON (E2E, lr=5e-4)      | 82.61     | 74.18     | 73.10     | None (full pipeline)
Fine-tuned LSTM           | 84.61     | 90.48     | 74.59     | None (retrained)
Retrieval translator (ours)| 85.49   | 91.14     | 76.71     | Frozen predictor
```

Note: All E2E methods use right-padding (matching YAIB convention) for fair comparison. Fine-tuned LSTM = unfrozen MIMIC LSTM retrained on eICU. Best HP variant per method shown.

### Table 3: Ablation Study (in paper body)

Loss components, n_cross_layers, FeatureGate, etc. Already have data.

### Table 4: Cross-Domain Generalization (in paper body)

eICU->MIMIC vs HiRID->MIMIC. Already have data.

### Table 5: Statistical Significance (appendix)

Bootstrap CIs, p-values, multi-seed std.

---

## 9. Risk Assessment

| Risk | Probability | Mitigation |
|------|------------|------------|
| CLUDA adaptation takes >4 days | Medium | Cut scope: run on mortality + AKI only (skip sepsis). Still sufficient. |
| CLUDA beats us on some tasks | Medium | Expected -- they have more freedom. Frame as "even without freezing, CLUDA only matches our frozen method." |
| Statistics-only closes most of the gap | Low | Would be a finding, not a failure. If normalization gets +0.03 and translator gets +0.05, the delta (+0.02) is still the nonlinear component. |
| Linear probe is surprisingly strong | Medium-Low | If it closes >50% of gap, reframe: "even simple transforms help, but retrieval adds [X] through context." |
| Not enough time for RAINCOAT | Medium | CLUDA alone is sufficient. Mention RAINCOAT in related work. |
| GPU contention delays experiments | Medium | Use nighttime hours (3 GPUs). Prioritize classification tasks first. |
| May 6 deadline too tight | Low | Core experiments (Phases 1-3) take 8 days. Writing starts April 8. 4 weeks for writing is sufficient. |

---

## 10. Week-by-Week Execution Plan

### Week 1 (Mar 27 - Apr 2)
- **Day 1-2**: Statistics-only baseline (all 5 tasks), CDAN implementation, linear probe implementation
- **Day 3-5**: Queue CDAN + linear probe experiments. Start CLUDA adaptation.
- **Day 5-7**: CLUDA debug runs. Queue CLUDA full experiments.
- **Parallel**: Bootstrap CIs running on CPU.

### Week 2 (Apr 3 - Apr 9)
- **Day 1**: Fine-tuned LSTM implementation + experiments queued.
- **Day 2-4**: CLUDA experiments running. RAINCOAT adaptation starts (if CLUDA went smoothly).
- **Day 5-7**: Collect all baseline results. Run regression-task variants for new baselines.

### Week 3 (Apr 10 - Apr 16)
- **Day 1-3**: Any remaining experiments (RAINCOAT, regression tasks, multi-seed).
- **Day 4-7**: Compile all results. Create final tables. Start paper skeleton with real numbers.

### Week 4 (Apr 17 - Apr 23)
- Full writing mode. Method, experiments, results sections drafted.

### Week 5 (Apr 24 - Apr 30)
- Introduction, related work, theory section. Internal review. Figures finalized.

### Week 6 (May 1 - May 6)
- May 4: Abstract submitted.
- May 5: Final revision.
- May 6: Paper submitted.

---

## 11. Decision Points

### April 2: CLUDA Go/No-Go
If CLUDA adaptation is not debugged by April 2, drop it. Report only frozen-model baselines in Table 1 (which is the fairer comparison anyway). Mention CLUDA in related work with "their method requires end-to-end training and cannot be applied under our frozen constraint."

### April 7: RAINCOAT Go/No-Go
If CLUDA experiments are not complete by April 7, skip RAINCOAT entirely. Focus on writing.

### April 10: Results Freeze
No new experiments after April 10. Everything in the paper must be final by this date.

---

## 12. Summary: What Changes vs Current Plan

| Current Plan | Revised Plan | Reason |
|-------------|-------------|--------|
| DANN + CORAL + CoDATS as primary baselines | DANN + CORAL + CDAN + Linear probe as primary; CoDATS to appendix | CoDATS confounds architecture vs loss. CDAN is standard. Linear probe is essential. |
| DA baselines as the main comparison | Two-tier: frozen-model baselines (primary) + unconstrained methods (secondary) | Cleaner story. Addresses the Option A vs B problem directly. |
| No statistics-only baseline | Add statistics-only | Answers the most basic reviewer question |
| No fine-tuning upper bound | Add fine-tuned LSTM | Frames the cost of the frozen constraint |
| ACON as stretch goal | Drop ACON entirely | Too much adaptation effort for marginal value. CLUDA is more relevant (clinical data). |
| 7 baselines | 9-11 baselines (5 frozen + 2-3 unconstrained + 3 reference) | Comprehensive but structured |

---

## 13. The One-Sentence Pitch for Each Table

- **Table 1**: "Under the same frozen-model constraint, our retrieval translator outperforms all DA baselines by 27-207%, with the gap widening on harder tasks."
- **Table 2**: "Our frozen-model translator matches or exceeds unconstrained methods that train the full pipeline end-to-end."
- **Table 3**: "Ablations show each component (fidelity loss, cross-attention, memory bank, FeatureGate) contributes, with retrieval's cross-attention resolving the gradient alignment conflict."
- **Table 4**: "Consistent improvements across two source domains (eICU and HiRID) demonstrate the translator generalizes beyond a single domain pair."
