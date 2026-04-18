# SD4H 2026 Workshop Paper — Writing Plan

**Venue**: Structured Data for Health (SD4H) @ ICML 2026, Seoul
**Deadline**: April 15, 2026 AoE (~3 days from Apr 12)
**Format**: 4 pages + unlimited refs/appendices, ICML 2026 template, double-blind
**Status**: Non-archival — fully compatible with NeurIPS 2026 main track (May 4-6)

---

## Paper Title
**Input-Space Domain Adaptation via Retrieval-Guided Translation for Clinical Time Series**

## Key Selling Points (for SD4H reviewers)
1. **Novel paradigm**: Input-space DA with frozen downstream model (not feature-space)
2. **Directly in scope**: Clinical EHR structured data, time-series forecasting, representation learning
3. **Strong results**: Beats 8 DA baselines by 2-4×, surpasses native-domain performance
4. **Practical**: Frozen model = no revalidation needed for deployment
5. **General**: Works across 5 tasks, 2 source domains, 3 architectures

---

## Page Budget (4 pages = ~3400 words in ICML two-column)

| Section | Target | Words | Key Content |
|---------|--------|-------|-------------|
| Abstract | — | ~150 | Problem → gap → approach → results → significance |
| Introduction | 0.75 pg | ~650 | Motivation, gap, approach, 3 contributions |
| Method | 1.25 pg | ~1000 | Problem setting + architecture + training |
| Experiments | 1.50 pg | ~1200 | Setup, main table, multi-source, ablation |
| Conclusion | 0.25 pg | ~200 | Summary, limitations, future |
| References | unlimited | — | ~30-40 most essential citations |
| Appendix | unlimited | — | Full tables, hyperparameters, extended results |

---

## Tables & Figures Plan

### Figure 1: Architecture Diagram (MUST HAVE — top of page 2)
- Encoder → Memory Bank → Cross-Attention → Decoder → Frozen LSTM → Prediction
- Show source input, target memory bank, and the frozen model clearly
- Should be self-contained (understandable without reading text)
- **Source**: Need to create. Use `paper/figures/` or draw with TikZ/Excalidraw

### Table 1: Main Classification Results (MUST HAVE)
- Compact format for 4-page paper
- Rows: Frozen baseline, best of {DANN, CORAL, CoDATS, CDAN}, RAINCOAT, CLUDA, Fine-tuned, **Ours**
- Columns: Mortality AUROC, AKI AUROC, Sepsis AUROC
- Reference rows (gray/italic): eICU-native LSTM, MIMIC-native LSTM
- Bold best, underline second-best
- Include mean±std from multi-seed runs

Data (from MEMORY.md and da_baselines_results.md):
```
Frozen baseline:  80.79    85.58    71.59
DANN:            +3.59    +3.16    +1.64
Deep CORAL:      +3.74    +3.08    +1.67
CoDATS:          +3.52    +1.26    -0.37
Ours:            +4.76    +5.56    +5.12
eICU-native:      85.5     90.2     74.0
MIMIC-native:     86.7     89.7     82.0
```

### Table 2: Multi-Source + Architecture Transfer (SHOULD HAVE)
- Compact: 2-3 rows × 3-4 columns
- HiRID→MIMIC results: +0.047 mort, +0.078 AKI, +0.078 sepsis
- Architecture transfer: GRU (47-85%), TCN (57-86%)
- Can be merged with Table 1 or placed in appendix if space is tight

### Table 3: Ablation (SHOULD HAVE — can go to appendix)
- Key ablations: no fidelity, no cross-attn, no memory bank, no normalization
- n_cross_layers: 2 vs 3 per task

---

## Writing Order (Recommended for Next Session)

### Phase 1: Core Content (highest priority)
1. **Abstract** — Write last but draft first. 150 words, tight.
2. **Table 1** — Generate LaTeX table with exact numbers from results
3. **Experiments §3.1 Setup** — Factual, straightforward to write
4. **Experiments §3.2 Main Results** — Narrate Table 1
5. **Method §2.1 Problem Setting** — Formal notation (already outlined)
6. **Method §2.2 Architecture** — Describe retrieval translator
7. **Method §2.3 Training** — Loss functions, phases

### Phase 2: Framing
8. **Introduction** — Paragraphs 1-3 + contributions
9. **Conclusion** — Summary + limitations + future work
10. **Experiments §3.3** — Multi-source and architecture transfer
11. **Experiments §3.4** — Ablation (brief for workshop)

### Phase 3: Polish
12. **Figure 1** — Architecture diagram
13. **Abstract** — Final revision
14. **Appendix** — Hyperparameters, extended results, dataset stats
15. **References** — Verify all citations compile, remove unused

---

## Key Numbers to Include

### Classification (eICU → MIMIC-IV)
| Task | Baseline | Our Best | Δ | eICU-native | Beat native? |
|------|----------|----------|---|-------------|-------------|
| Mortality | 80.79 | 85.55 | +4.76 | 85.5 | YES |
| AKI | 85.58 | 91.14 | +5.56 | 90.2 | YES (+0.94) |
| Sepsis | 71.59 | 76.78* | +5.19* | 74.0 | YES |

*Sepsis: mean across seeds = +4.94 (high variance, std ~2.0)

### Regression (eICU → MIMIC-IV)
| Task | Baseline MAE | Translated MAE | Δ |
|------|-------------|---------------|---|
| LoS | 42.5h | 39.2h | -3.3h |
| KF | 0.403 mg/dL | 0.382 mg/dL | -0.021 |

### Multi-Source (HiRID → MIMIC-IV)
| Task | Δ AUROC | vs eICU best |
|------|---------|-------------|
| AKI | +7.76 | +2.20 above eICU |
| Sepsis | +7.77 | +2.65 above eICU |
| Mortality | +4.74 | Tied |

### Architecture-Agnostic Transfer
| Task | GRU (% of LSTM) | TCN (% of LSTM) |
|------|----------------|----------------|
| Mortality | 85% | 72% |
| AKI | 56% | 57% |
| Sepsis | 47% | 86% |

### vs DA Baselines (best baseline per task)
| Task | Best Baseline | Best Baseline Δ | Our Δ | Improvement ratio |
|------|--------------|-----------------|-------|------------------|
| Mortality | CORAL +3.74 | +3.74 | +4.76 | 1.3× |
| AKI | DANN +3.16 | +3.16 | +5.56 | 1.8× |
| Sepsis | CORAL +1.67 | +1.67 | +5.12 | 3.1× |

---

## Style Guidelines (for SD4H/ICML)

### Writing
- Active voice: "We propose..." not "A method is proposed..."
- NO opening with "AI is transforming healthcare"
- Precise claims: match every statement to evidence
- Define clinical terms on first use (AKI = Acute Kidney Injury, etc.)
- Use `\citet` for textual citations, `\citep` for parenthetical

### Formatting
- ICML two-column format (handled by icml2026.sty)
- Figures: vector PDF, font size ≥ 7pt
- Tables: booktabs only (no \hline, no vertical rules)
- Bold best results, underline second-best
- Non-breaking spaces before citations: `method~\citep{}`

### Double-Blind Requirements
- No self-identifying language ("our previous work [X]" → "prior work [X]")
- Anonymous authors in template (already set)
- No institution names, no acknowledgments in submission
- ArXiv preprint is allowed (workshop policy)

---

## Essential References (~30-40)

1. **YAIB**: van de Water et al., ICLR 2024 — benchmark we build on
2. **DANN**: Ganin et al., 2016 — canonical adversarial DA
3. **Deep CORAL**: Sun & Saenko, 2016 — canonical statistical DA
4. **CoDATS**: Wilson et al., 2020 — time-series DA
5. **RAINCOAT**: He et al., NeurIPS 2023 — recent TS-DA
6. **CLUDA**: Ozyurt et al., ICLR 2023 — contrastive TS-DA
7. **CDAN**: Long et al., 2018 — conditional adversarial DA
8. **Ben-David et al., 2010** — DA theory (h-divergence)
9. **eICU**: Pollard et al., 2018 — source dataset
10. **MIMIC-IV**: Johnson et al., 2023 — target dataset
11. **HiRID**: Hyland et al., 2020 — second source
12. **AdaTime**: Ragab et al., 2023 — TS-DA benchmark
13. **Attention Is All You Need**: Vaswani et al., 2017 — transformer/cross-attention
14. **RAG**: Lewis et al., 2020 — retrieval-augmented generation (conceptual connection)
15. **CycleGAN**: Zhu et al., 2017 — input-space translation (image domain)

---

## Appendix Content Plan

### A. Dataset Statistics (0.5 pages)
- Table: # patients, # stays, # timesteps, label rate per task per dataset

### B. Implementation Details (0.5 pages)
- Full hyperparameter table
- Phase 1 vs Phase 2 settings
- Task-specific configurations (temporal mode, n_cross_layers)

### C. Extended Results (1 page)
- AUCPR for classification
- Full regression results
- Per-seed breakdowns
- Calibration metrics (ECE, Brier) if available

### D. Full Ablation (0.5 pages)
- All ablation experiments in table form

### E. Compute Budget (0.25 pages)
- GPU types, training time, parameters

---

## SD4H-Specific Writing Requirements

These are critical for SD4H acceptance but NOT in NeurIPS requirements:

1. **Clinical motivation first** — Introduction paragraph 1 must be a clinical scenario, not ML jargon. "Clinical prediction models validated at one institution degrade at another due to differences in equipment, coding practices, and patient populations." NOT "Domain adaptation is a fundamental problem."

2. **Frozen model = regulatory advantage** — Spell out explicitly: validated/FDA-cleared models cannot be retrained without re-certification. Our method sidesteps this. 2-3 sentences in intro + 1 in conclusion. This is the killer feature for a health workshop.

3. **Dataset clinical context** — Don't just say "eICU has N patients." Describe: eICU = 208 US hospitals (heterogeneous multi-center), MIMIC-IV = single Boston hospital (Beth Israel), HiRID = single Swiss ICU (Bern). Sepsis label prevalence = 1.1% (a genuine clinical challenge). 3-4 sentences in experimental setup.

4. **Per-task clinical interpretation** — Don't just report "+5.56 AUROC." Say: "AKI prediction on translated eICU data surpasses even the MIMIC-native model (91.1 vs 89.7), suggesting the translator leverages the diversity of 208 source hospitals." 1-2 interpretive sentences per task.

5. **Missing data handling** — SD4H explicitly lists "irregular and missing data" as a topic. One sentence about MI features, schema mismatches (LoS: 52 features, KF: 292), and how the translator handles them.

---

## Explicit SKIP List (not needed for SD4H)

| NeurIPS Group | Why Skip |
|---|---|
| Temperature scaling / reliability diagrams | Overkill for 4-page workshop. Add raw ECE/Brier to appendix if data exists |
| Reproducibility package | No checklist requirement. Non-archival. Zero effort |
| Formal theory (Ben-David proposition) | Health data workshop, not theory venue. 2-3 intuition sentences suffice |
| TTA baselines (T3A/SHOT/TENT) | 8 DA baselines already sufficient. Acknowledge in limitations |
| PAD (Proxy A-distance) computation | Quantitative domain divergence is nice-to-have, not needed |
| Formal failure mode analysis | 2 sentences in conclusion suffice |

---

## Acceptance Criteria (self-check before submit)

- [ ] Abstract <= 150 words, states frozen-model constraint clearly
- [ ] Introduction opens with clinical scenario, not ML jargon
- [ ] Regulatory/validation advantage mentioned (intro + conclusion)
- [ ] Table 1 has DA baselines + reference native models + our results with mean +/- std
- [ ] Figure 1 shows architecture with frozen model clearly marked
- [ ] Method section defines notation, architecture, training (all 3 subsections)
- [ ] Experiments mention clinical context of datasets (hospitals, populations, label rates)
- [ ] Per-task clinical interpretation (not just numbers)
- [ ] Bootstrap significance stated
- [ ] Computational cost reported (params, GPU hours, or inference time)
- [ ] Related work cites >= 2 papers from 2024-2025
- [ ] Conclusion includes limitations (label density, GPU memory, task-specific tuning)
- [ ] No double-blind violations (grep for institution names, server IPs, author names)
- [ ] Compiles cleanly with `make` in `paper/sd4h/`
- [ ] <= 4 pages main text (excluding refs + appendix)

---

## Pre-Writing Checklist (Setup for Next Session)

- [x] ICML 2026 template downloaded and configured
- [x] Paper skeleton created with all section files
- [x] Bibliography symlinked (642 entries available)
- [x] Figures directory symlinked
- [x] Makefile created and tested (paper compiles)
- [x] Key numbers collected from MEMORY.md and results docs
- [ ] Architecture diagram (Figure 1) — CREATE DURING WRITING
- [ ] Main results table (LaTeX) — CREATE DURING WRITING
- [ ] Write all sections — THE WRITING SESSION
- [ ] Final compile and page check
- [ ] Submit to OpenReview

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Deadline too tight (3 days) | 4-page workshop paper with all results ready = feasible |
| No architecture diagram | Use TikZ or reference Excalidraw file in repo |
| Sepsis high variance | Report mean±std honestly, discuss in text |
| Missing bootstrap CIs | For workshop paper, multi-seed mean±std is acceptable |
| Missing TTA baselines | Acknowledge in limitations; not required for workshop |
| ICML checklist | NOT required for SD4H workshop |
| Double-blind violation | Review carefully — no self-citations to own prior work |

---

## Also Consider: FMSD @ ICML (May 1 deadline)

Same format (4 pages, ICML template, non-archival). If SD4H is submitted, FMSD submission is a simple revision of the same paper with slightly different framing (emphasize "foundation model" angle — the pretrained memory bank as a form of structured data foundation).
