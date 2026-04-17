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

## Group 1: DA Baselines Implementation — DONE

**Priority**: P0 | **Status**: COMPLETE (8 methods × 3 tasks, frozen + E2E)
**Results**: `docs/neurips/da_baselines_results.md`

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

## Group 2: HiRID Dataset Integration — DONE

**Priority**: P1 | **Status**: COMPLETE. All 5 tasks done.
**Results**: AKI +0.078, Sepsis +0.078, Mortality +0.047, LoS MAE −0.045, KF MAE −0.0001.

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
**Analysis**: `docs/neurips/calibration_analysis.md` (existing Brier/ECE analysis, temperature scaling plan)

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
**Analysis**: `docs/neurips/statistical_completeness.md` (gap analysis, NeurIPS sufficiency assessment)

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

## Group 6: Architecture Finalization — DONE

**Priority**: P0 | **Status**: COMPLETE. V5 configs locked per task.

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
**Analysis**: `docs/neurips/gradient_magnitude_theory.md` (falsification of cos α, magnitude bottleneck theory)

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

## Group 9: TTA Baselines — TENT / SHOT / T3A (CRITICAL — Gap from Apr 12 review)

**Priority**: P0 — Reviewers at ICLR 2024+ specifically asked "why not TTA?" on DA submissions.
**Effort**: 1 week | **Impact**: CRITICAL (blocks "missing baseline" rejection)
**Source**: `docs/neurips/gap_analysis_generality_claim.md`

### Why This Matters

TTA methods are the closest published comparison to our frozen-backbone constraint. SHOT keeps the classifier frozen and updates the encoder; T3A keeps the entire backbone frozen and adjusts classifier prototypes. A reviewer will argue: "Why not just use SHOT/T3A instead of training a whole translator?"

### Runtime Estimates

TTA methods are orders of magnitude cheaper than translator training. The bottleneck is implementation (~3-4 days coding), not compute (~1 day GPU).

**EHR tasks** (reference: retrieval translator = 30-60 hours per task):

| Method | Mortality | AKI | Sepsis | Notes |
|---|---|---|---|---|
| **T3A** | ~2 min | ~5 min | ~5 min | Single forward pass (prototypes) + test. Zero training. |
| **SHOT** (adapter) | ~30-60 min | ~1-2 hrs | ~1-2 hrs | ~20 epochs entropy min on target, small adapter |
| **TENT** | N/A | N/A | N/A | Inapplicable — LSTMs have no BN layers |

**AdaTime tasks** (reference: translator = 14s-2min per scenario):

| Method | Per Scenario | All 10 Scenarios | Notes |
|---|---|---|---|
| **T3A** | ~1-3 sec | ~30 sec | Prototypes + inference |
| **TENT** | ~5-10 sec | ~1-2 min | Few backward passes on test BN |
| **SHOT** (adapter) | ~30-60 sec | ~5-10 min | Lightweight adaptation loop |

**Total compute**: ~7-9 hours for all methods × all datasets. Multi-seed (3 seeds) under 1 day.

### Exact Requirements

1. **T3A** (Iwasawa & Matsuo, NeurIPS 2021) — **implement first**
   - Adjusts classifier prototypes at test time. Backbone fully frozen.
   - **This is the most direct competitor** — same frozen-backbone constraint.
   - **Action**: Implement T3A on our frozen LSTM's output features.
   - Reference impl: [github.com/matsuolab/T3A](https://github.com/matsuolab/T3A) (vision, adapt to TS)
   - Implementation: `src/baselines/t3a.py`
   - Effort: ~1 day coding, ~40 min GPU (all tasks)

2. **SHOT** (Liang et al., ICML 2020) — **implement second**
   - Source-free: freezes classifier, updates encoder via entropy minimization + pseudo-labels
   - **Problem**: In our setting the entire model is frozen (not just classifier). SHOT requires updating encoder weights.
   - **Action**: Implement SHOT where only a small adapter before the frozen LSTM is updated (fair comparison). Compare vs our translator.
   - Implementation: `src/baselines/shot.py`
   - Effort: ~2 days coding, ~5-7 hrs GPU

3. **TENT** (Wang et al., ICLR 2021) — **AdaTime only**
   - Adapts batch-norm statistics at test time via entropy minimization
   - **Problem**: LSTMs have no batch-norm layers → TENT is inapplicable for EHR.
   - **Action**: Implement for AdaTime CNNs only (they have BN). State LSTM incompatibility in paper.
   - Reference impl: [github.com/DequanWang/tent](https://github.com/DequanWang/tent) (vision, adapt to 1D-CNN)
   - If implementing for EHR: LayerNorm adaptation as proxy (optional, low priority).
   - Implementation: `src/baselines/tent.py`
   - Effort: ~1 day coding, ~1 hr GPU

4. **Experiments**
   - Run on EHR: at least Mortality, AKI, Sepsis (T3A + SHOT)
   - Run on AdaTime: at least HAR, HHAR, WISDM (T3A + SHOT + TENT)
   - Multi-seed: 3 seeds minimum
   - Expected outcome: TTA methods should underperform because they adapt output/features, not inputs. Our translator provides richer adaptation by transforming the entire input space.

### Paper narrative value

TTA methods are fast and cheap, yet our translator beats them — this becomes an argument for *when to use which*: TTA for quick adaptation with small shift, full translator for large cross-domain gaps. The cost table itself motivates the method.

### Output
TTA comparison table in paper. Discussion of why input-space adaptation > output-space adaptation.

---

## Group 10: Computational Cost Analysis (CRITICAL — Gap from Apr 12 review)

**Priority**: P0 — Reviewers routinely expect this for any method that adds parameters.
**Effort**: 1 day | **Impact**: HIGH
**Analysis**: `docs/neurips/computational_cost.md` (param counts, training times, VRAM, inference)

### Exact Requirements

1. **Parameter count table**
   - Frozen LSTM baseline: #params
   - Translator (retrieval): #params (breakdown: encoder, decoder, cross-attention, memory bank)
   - Each DA baseline (DANN, CORAL, CoDATS): #params of adaptation module
   - Ratio: translator params / baseline params

2. **Training cost**
   - Wall-clock time per epoch (from existing logs)
   - Total GPU hours per task (Phase 1 + Phase 2)
   - Peak GPU VRAM (from `nvidia-smi` during runs)

3. **Inference cost**
   - ms/batch at inference (translator forward pass + frozen LSTM)
   - Compare vs source-only (just frozen LSTM)
   - Compare vs E2E methods (full model forward pass)

4. **AdaTime cost** (simpler — smaller models)
   - Training time per scenario
   - Compare vs published AdaTime method training times

### Output
Table in paper: Method | #Params | GPU Hours | VRAM | Inference ms/batch

---

## Group 11: Visualization & Domain Divergence (HIGH — Gap from Apr 12 review)

**Priority**: P1 — Nearly universal in DA papers; missing this loses 1-2 review points.
**Effort**: 3–4 days | **Impact**: HIGH
**Analysis**: `docs/neurips/visualization_analysis.md` (domain divergence metrics, hidden state analysis, figure plans)

### Exact Requirements

1. **t-SNE / UMAP visualization**
   - Extract LSTM hidden states for: (a) source (eICU) raw, (b) translated source, (c) target (MIMIC)
   - Plot 2D projections colored by domain
   - Show alignment improvement after translation
   - Do for 2 tasks: AKI (causal, per-timestep) + Mortality (bidirectional, per-stay)
   - Script: `scripts/visualize_tsne.py`

2. **Proxy A-distance (PAD)**
   - Train linear SVM to distinguish source vs target in LSTM feature space
   - PAD = 2(1 - 2 × classification_error)
   - Compute before and after translation
   - Report for all 3 classification tasks
   - Script: `scripts/compute_pad.py`

3. **Feature-level analysis** (optional but high-value)
   - Which input features does the translator modify most? (L2 distance per feature)
   - Correlate with known domain-shift features (lab distributions, vital sign scales)
   - Shows the translator learned clinically meaningful transformations

### Output
- Fig: t-SNE before/after (2 tasks)
- Table: PAD before/after (3 tasks)
- Optional fig: per-feature translation magnitude heatmap

---

## Group 12: 2025 Related Work & Positioning (HIGH — Gap from Apr 12 review)

**Priority**: P1 — Signals awareness of current landscape; missing recent citations looks outdated.
**Effort**: 0.5–1 day | **Impact**: MEDIUM-HIGH

### Must-Cite Papers (2024-2025)

| Paper | Venue | Why Cite |
|---|---|---|
| **TransPL** | ICML 2025 | Latest TS-DA SOTA, VQ-code pseudo-labeling |
| **TemSR** | ICLR 2025 | Source-free TS-DA via temporal recovery |
| **CDA-DAPT** | ICLR 2025 | **Closest competitor**: frozen Transformer + adapters + prompts |
| **CPFM** | arXiv 2025 | Black-box TS-DA via foundation model prompts |
| **ACON** | NeurIPS 2024 | AdaTime SOTA (already in our comparison) |
| **RCD-KD** | NeurIPS 2024 | Knowledge distillation for TS-DA |
| **Fawaz et al.** | DMKD 2025 | Newer TS-DA benchmark (7 extra datasets) |
| **SEA++** | TPAMI 2024 | Multi-graph sensor alignment for MTS |
| **LCA** | TPAMI 2025 | Latent causal alignment |
| **TA4LS** | KDD 2025 | Label shift in TS-DA |

### Key Positioning vs CDA-DAPT (ICLR 2025)

CDA-DAPT is the most important to differentiate:
- **Theirs**: Frozen Transformer + parameter-efficient adapters + domain prompts → adapts intermediate representations
- **Ours**: Frozen backbone (any architecture) + input-space translator → adapts raw input
- **Theirs**: Continual DA (sequential domains) → different problem setting
- **Ours**: Single-pair UDA → deeper evaluation per pair
- **Theirs**: Transformer-specific (adapters require specific architecture) → architecture-locked
- **Ours**: Architecture-agnostic (demonstrated on LSTM, GRU, TCN, CNN) → universal

### Output
Updated related work section in paper skeleton.

---

## Group 13: Failure Mode Analysis (MEDIUM — Gap from Apr 12 review)

**Priority**: P2 — Shows intellectual honesty; reviewers appreciate self-awareness.
**Effort**: 1 day | **Impact**: MEDIUM

### Exact Requirements

1. **Identify failure scenarios**
   - AdaTime: HAR 12->16 consistently loses across all configs
   - EHR: Sepsis high variance (inherent to 1.1% label density)
   - When does translation hurt vs help?

2. **Characterize failures**
   - Is there a detectable signal? (e.g., high reconstruction error, low memory bank similarity)
   - Domain shift magnitude vs translation benefit correlation
   - Label density vs translation benefit correlation

3. **Discuss in paper**
   - 1-2 paragraphs in Discussion/Limitations
   - Honest assessment of when the method shouldn't be used

### Output
Failure analysis paragraph in Limitations section.

---

## Updated Execution Timeline

```
ALREADY DONE:
  Group 1 (DA baselines) — COMPLETE, 8 methods × 3 tasks
  Group 2 (HiRID) — COMPLETE, all 5 tasks
  Group 6 (Architecture lock) — COMPLETE, V5 configs finalized

REMAINING:
  Week of Apr 13–19:  Group 9 (TTA baselines — start impl) + Group 10 (computational cost — 1 day)
  Week of Apr 20–26:  Group 9 (TTA experiments) + Group 4 (EHR bootstrap CIs) + Group 11 (t-SNE, PAD)
  Week of Apr 27–May 1: Group 7 (paper skeleton) + Group 8 (theory) + Group 12 (related work) + Group 3 (temp scaling)
  Week of May 1–4:    Group 13 (failure modes) + Group 5 (reproducibility) + final writing
  May 4: Abstract submission
  May 4–6: Paper finalization and submission
```

---

## Updated Acceptance Probability Estimates

| Scenario | Probability | What's Included |
|----------|-------------|-----------------|
| Current (DA baselines + HiRID + AdaTime 5/5 wins) | 40–45% | Strong results but missing TTA baselines, CIs, viz |
| + EHR bootstrap CIs + TTA baselines + cost table | 55–60% | Addresses all P0 gaps |
| + t-SNE/PAD + 2025 citations + theory | 60–70% | Full P0 + P1 package |
| + failure analysis + calibration + subgroup | 65–75% | Complete package |
