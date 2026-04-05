# NeurIPS 2026 — Related Work

## Key Positioning Statement

> No prior work combines (1) a frozen target predictor, (2) neural input-space translation, (3) retrieval-augmented cross-attention, (4) clinical EHR time-series, and (5) systematic gradient alignment analysis. We study three translation paradigms (delta, shared latent, retrieval) and discover that retrieval uniquely satisfies the gradient alignment condition across all task types.

---

## Category 1: Domain Adaptation for Clinical/EHR Time-Series

This is our primary comparison field. Existing work either (a) fine-tunes the target model, (b) learns domain-invariant representations, or (c) harmonizes data schemas. None operate in the frozen-model, input-space translation setting.

### Papers

| # | Paper | Venue | Approach | Key Difference from Ours |
|---|-------|-------|----------|--------------------------|
| 1 | VRADA (Purushotham et al., 2017) | ICLR 2017 | Variational RNN + adversarial alignment for clinical time-series | Representation-space alignment; requires end-to-end training |
| 2 | AdaDiag (Zhang et al., 2022) | JBI 2022 | Adversarial DA with pre-trained Transformer for diagnosis prediction | Representation-space; works on discrete events, not continuous TS |
| 3 | METRE (Liao & Voldman, 2023) | JBI 2023 | Standardized extraction pipeline across MIMIC-IV and eICU | Data harmonization only; no learned adaptation |
| 4 | TransEHR (Zhao et al., 2026) | ESWA 2026 | Contrastive learning for continual learning across feature spaces | Temporal feature drift, not cross-hospital shift |
| 5 | ExtraCare (2026) | arXiv 2026 | Concept-grounded orthogonal decomposition for interpretable DA | Representation decomposition; eICU test bed |
| 6 | Mutnuri et al. (2024) | JMIR 2024 | Inductive transfer + DA for ICU outcomes (mortality, AKI, LoS) | Fine-tunes target model; standard adversarial methods |
| 7 | Driever et al. (2025) | medRxiv 2025 | Supervised/unsupervised DA for Transformer EHR models | Fine-tunes backbone; different disease endpoints |
| 8 | OTTEHR (2024) | JAMIA 2025 | Optimal transport for EHR transfer (MIMIC-III/IV, eICU) | OT-based; no temporal modeling; population-level transport |
| 9 | KnowRare (2025) | npj Dig Med 2025 | Self-supervised pre-training + knowledge graph for rare conditions | Cross-condition, not cross-hospital adaptation |
| 10 | Purushotham et al. (2018) | JBI 2018 | Benchmark: deep learning on MIMIC-III for clinical tasks | Benchmark; no DA component |
| 11 | Harutyunyan et al. (2019) | Sci. Data 2019 | Multitask clinical benchmarks on MIMIC-III | Foundational benchmark |

### Our Case Against (NeurIPS Reviewer Perspective)

**Reviewer concern**: "How does this differ from standard clinical DA?"

**Response**: All prior clinical DA methods either (a) fine-tune the target model (Mutnuri, Driever), (b) learn domain-invariant representations (VRADA, AdaDiag, ExtraCare), or (c) use population-level statistical alignment (OTTEHR). None preserve a frozen target predictor — a critical constraint for regulatory compliance (the FDA's SaMD framework distinguishes "locked" algorithms with a simpler regulatory pathway), auditability, and deployment in clinical settings where the predictor has been separately validated. Our work is the first to show that input-space translation for a frozen clinical LSTM can match or exceed the performance of the target-domain native model.

### Citation Plan
- **Must cite**: VRADA (seminal), Purushotham 2018 (benchmark), Harutyunyan 2019 (benchmark), OTTEHR (same databases), ExtraCare (concurrent)
- **Should cite**: AdaDiag, METRE, Mutnuri, KnowRare
- **Optional**: TransEHR, Driever

---

## Category 2: General Time-Series Domain Adaptation

These methods address distribution shift in time-series data broadly (sensors, HAR, industrial). They establish the methodological landscape but none handle the frozen-model or clinical EHR setting.

### Papers

| # | Paper | Venue | Approach | Key Difference from Ours |
|---|-------|-------|----------|--------------------------|
| 1 | CoDATS (Wilson et al., 2020) | KDD 2020 | 1D CNN + gradient reversal for sensor time-series | Feature-space alignment; not frozen-model |
| 2 | CALDA (Wilson et al., 2023) | IEEE TPAMI 2023 | Contrastive + adversarial for multi-source TS DA | Extension of CoDATS; still representation-space |
| 3 | RAINCOAT (He et al., 2023) | ICML 2023 | Feature + label shift DA via temporal + frequency features | Handles label shift; representation-space |
| 4 | CLUDA (Ozyurt et al., 2023) | ICLR 2023 | Contrastive learning for domain-invariant TS representations | kNN-based contrastive; not input-space |
| 5 | AdvSKM (Liu & Xue, 2021) | IJCAI 2021 | Adversarial spectral kernel matching for TS UDA | Spectral-kernel MMD in representation-space |
| 6 | POND (NEC Labs, 2024) | KDD 2024 | Prompt tuning for multi-source TS DA | Prompt ≈ delta, but single model not frozen target |
| 7 | ACON (Liu et al., 2024) | NeurIPS 2024 | Temporal + frequency transferability for TS DA | Feature-space; not frozen-model |
| 8 | DAAC (2025) | NeurIPS 2025 | GAN-enhanced adaptive contrastive for medical TS | Medical TS DA; representation-space |
| 9 | MAPU (Ragab et al., 2023) | KDD 2023 | Source-free DA with temporal imputation for TS | Source-free constraint (restricts data access, not model weights) |
| 10 | ADATIME (Ragab et al., 2023) | ACM TKDD 2023 | Benchmarking suite for TS domain adaptation (11 methods) | Benchmark; establishes evaluation standards |
| 11 | Multi-View CL (Oh et al., 2025) | PMLR 2025 | Multi-view contrastive for medical TS (EEG, ECG, EMG) | Different modality (physiological signals, not EHR) |
| 12 | TS DA Benchmark (2025) | DMKD 2025 | Comprehensive UDA benchmark for time series | Benchmark paper; 7+ datasets |

### Our Case Against

**Reviewer concern**: "Why not use established TS-DA methods like RAINCOAT or CLUDA?"

**Response**: Standard TS-DA methods learn domain-invariant representations through the feature extractor. Our setting has a fundamental constraint: the target LSTM is frozen (weights cannot be updated). This means representation-space alignment via the feature extractor is impossible. We must operate exclusively in input space. In Section 5, we adapt DANN, CORAL, and CoDATS to our frozen-model setting and show they underperform our neural translators, because simple input transforms lack the capacity for complex domain mappings.

**ACON (Liu et al., NeurIPS 2024)** analyzes temporal vs. frequency feature transferability in TS-DA — our frozen-model constraint precludes their approach of joint temporal-frequency adaptation in the encoder, motivating our input-space alternative.

**Source-free DA (MAPU)** restricts access to *source data* during adaptation; our constraint restricts access to *model weights*. These are orthogonal constraints — MAPU still modifies the target model, which we cannot.

**Technical note**: POND's prompt-tuning approach is conceptually closest to our delta translator — both learn additive input perturbations. However, POND tunes prompts for a trainable model, while we must translate for a truly frozen one. Our experiments show delta translation alone is insufficient for sparse-label tasks (sepsis), motivating the progression to retrieval.

### Citation Plan
- **Must cite**: CoDATS (baseline), RAINCOAT (SOTA), CLUDA (SOTA), ACON (NeurIPS 2024)
- **Should cite**: POND (prompt ≈ delta connection), DAAC (NeurIPS 2025), AdvSKM (spectral MMD), ADATIME (benchmark), MAPU (source-free)
- **Optional**: CALDA, Multi-View CL, TS DA Benchmark

---

## Category 3: Frozen-Model / Input-Space Adaptation

This is our most novel positioning category. Work here comes primarily from NLP (prompt tuning) and vision (VPT, reprogramming), with recent extensions to time-series. No prior work applies this paradigm to clinical EHR domain adaptation.

### Papers

| # | Paper | Venue | Approach | Key Difference from Ours |
|---|-------|-------|----------|--------------------------|
| 1 | PixelDA (Bousmalis et al., 2017) | CVPR 2017 | GAN-based pixel-level input-space DA for images | Input-space DA but model is not frozen; vision domain |
| 2 | Adversarial Reprogramming (Elsayed et al., 2019) | ICLR 2019 | Universal additive perturbation to reprogram frozen ImageNet classifiers | Single universal perturbation; cross-task not cross-domain |
| 3 | Prompt Tuning (Lester et al., 2021) | EMNLP 2021 | Learned soft prompts for frozen language models | Fixed-length prompt vector; not sample-conditional |
| 4 | Voice2Series (Yang et al., 2021) | ICML 2021 | Reprogram frozen acoustic model for time-series classification | Cross-task (speech→TS); not cross-domain |
| 5 | VPT (Jia et al., 2022) | ECCV 2022 | Learnable input tokens for frozen ViTs | Vision; prepended tokens not full input transform |
| 6 | One Fits All / FPT (Zhou et al., 2023) | NeurIPS 2023 | Frozen pretrained Transformer for general TS analysis | Adapts layer norms, not inputs; different architecture |
| 7 | Time-LLM (Jin et al., 2024) | ICLR 2024 | Reprogram frozen LLM (GPT-2/LLaMA) for TS forecasting via input transformation | Cross-modality (language→TS) not cross-domain; generic pretrained LLM not task-specific LSTM |
| 8 | Model Reprogramming survey (Chen, 2024) | AAAI 2024 | Survey unifying input transformation + output mapping for frozen models | Taxonomy paper; frames the paradigm |
| 9 | SHOT (Liang et al., 2020) | ICML 2020 | Source-free DA: adapts frozen classifier head via feature extractor | Partial freeze (classifier frozen, encoder adapted); we freeze everything |
| 10 | FDA (Yang & Soatto, 2020) | CVPR 2020 | Fourier spectrum swap for training-free input-space DA | Fixed handcrafted transform, not learned; vision domain |
| 11 | Side-Tuning (Zhang et al., 2020) | ECCV 2020 | Additive side network for frozen model adaptation | Additive output modification; conceptually related to delta translator |
| 12 | L2C (2025) | ICLR 2025 | Input-space learning for frozen CLIP test-time DA | Vision; frozen CLIP, not clinical LSTM |
| 13 | TATO (Qiu et al., 2026) | ICLR 2026 | Adapt data (not model) via transformation pipelines for frozen TS foundation models | Handcrafted transforms, not learned neural translation |

### Our Case Against

**Reviewer concern**: "This is just prompt tuning / reprogramming for clinical data."

**Response**: Our work goes significantly beyond prompt tuning in four ways:

1. **Capacity**: Prompt tuning (Lester 2021) and VPT (Jia 2022) learn a small number of fixed parameters. Our translator is a full conditional neural network (Transformer encoder-decoder) that generates sample-specific translations — essential for the heterogeneous, variable-length clinical time-series setting.

2. **Retrieval augmentation**: No prior frozen-model work incorporates instance-level retrieval from the target domain. Our retrieval translator's memory bank + cross-attention mechanism provides target-domain grounding that is impossible with fixed perturbations.

3. **Gradient alignment theory**: We provide an analysis of when frozen-model translation succeeds or fails, via the gradient alignment condition cos(∇L_task, ∇L_fidelity). While gradient conflicts have been studied in MTL (Yu et al., 2020) and UDA (Du et al., 2024; Phan et al., 2024), our analysis uniquely characterizes frozen-model translation and shows retrieval architecturally resolves the conflict rather than requiring gradient surgery.

4. **Domain adaptation, not task adaptation**: Prompt tuning, VPT, and Voice2Series adapt a frozen model to a *new task*. We adapt *input data* from a different domain to an *existing task* — fundamentally different.

| Aspect | Prompt Tuning | VPT | Voice2Series | Time-LLM | **Ours** |
|--------|--------------|-----|-------------|----------|----------|
| Input modification | Fixed prompt vector | Prepended tokens | Learned transform | Reprogramming layer | **Full neural translator** |
| Sample-conditional | No | No | No | No | **Yes** |
| Retrieval | No | No | No | No | **Yes (memory bank)** |
| Variable-length | No | No | No | Fixed patches | **Yes** |
| Purpose | Task adaptation | Task adaptation | Cross-task | Cross-modality | **Cross-domain DA** |

**Time-LLM (Jin et al., ICLR 2024)** is the most prominent frozen-model reprogramming work for time series (1000+ citations). Key differences: Time-LLM reprograms across *modalities* (language→time-series) for a generic pretrained LLM; we translate across *domains* (eICU→MIMIC) for a task-specific validated clinical LSTM. Time-LLM does forecasting; we do clinical prediction with retrieval augmentation.

**TATO (Qiu et al., ICLR 2026)** is the closest concurrent work — they explicitly propose "adapt data to model" for frozen time-series foundation models. Key differences: (a) TATO uses handcrafted transformation operators while we learn the transformation end-to-end, (b) TATO targets forecasting while we target cross-hospital clinical prediction, (c) we provide theoretical grounding via gradient alignment.

**SHOT (Liang et al., ICML 2020)** freezes the classifier head but adapts the feature extractor — a partial freeze. Our constraint is stronger: the *entire* model (encoder + classifier) is frozen, requiring purely input-space adaptation. This positions our work at the extreme end of the model-freezing spectrum.

**Anticipated Q: "How is this different from CycleGAN / image-to-image translation?"** PixelDA (Bousmalis et al., 2017) performs input-space DA in vision via GANs. Key differences: (a) CycleGAN/PixelDA use cycle-consistency or adversarial fidelity; we use task loss through a frozen predictor, (b) they have no frozen downstream model constraint, (c) we handle irregular, variable-length multivariate clinical time-series with missingness patterns.

### Citation Plan
- **Must cite**: Adversarial Reprogramming, Prompt Tuning, VPT, Voice2Series, Time-LLM, TATO (concurrent)
- **Should cite**: One Fits All, L2C, SHOT, PixelDA, Model Reprogramming survey (Chen 2024)
- **Optional**: Side-Tuning, FDA

---

## Category 4: Retrieval-Augmented Models

Our retrieval translator draws on the retrieval augmentation paradigm, but applies it in a novel way: retrieving target-domain time-series windows to guide cross-domain translation (not generation or classification).

### Papers

| # | Paper | Venue | Approach | Key Difference from Ours |
|---|-------|-------|----------|--------------------------|
| 1 | RAG (Lewis et al., 2020) | NeurIPS 2020 | Retrieval-augmented generation for knowledge-intensive NLP | Foundational RAG; NLP |
| 2 | kNN-LM (Khandelwal et al., 2020) | ICLR 2020 | Nearest-neighbor interpolation for frozen language models | Foundation for our memory bank; interpolates output distributions |
| 3 | kNN-MT (Khandelwal et al., 2021) | ICLR 2021 | Frozen NMT model + per-step kNN from domain-specific datastore | Closest architectural ancestor; interpolates output distributions, not cross-attention |
| 4 | kNN-Prompt (Shi et al., 2022) | EMNLP 2022 | kNN retrieval + frozen GPT-2 for zero-shot classification | Frozen model + kNN; conceptual ancestor |
| 5 | RETRO (Borgeaud et al., 2022) | ICML 2022 | Chunked cross-attention over retrieved document chunks | Closest cross-attention precedent; NLP, not TS |
| 6 | RATD (Liu et al., 2024) | NeurIPS 2024 | Retrieval-augmented diffusion for TS forecasting | Retrieval guides diffusion denoising; forecasting |
| 7 | TS-RAG (Ning et al., 2025) | NeurIPS 2025 | RAG for TS foundation models with adaptive retrieval mixer | Foundation model augmentation; forecasting |
| 8 | RAM-EHR (Xu et al., 2024) | ACL 2024 | Retrieval from PubMed/KGs to augment EHR predictions | Retrieves text/knowledge, not time-series |
| 9 | RAFT (2025) | ICML 2025 | Retrieval-augmented time-series forecasting | Forecasting; general TS |

### Our Case Against

**Reviewer concern**: "This is just RAG applied to time series."

**Response**: Our retrieval mechanism differs from standard RAG in fundamental ways:

1. **What is retrieved**: We retrieve encoded *target-domain time-series windows*, not text passages or knowledge graph entities. The memory bank contains pre-encoded MIMIC-IV windows that represent what the frozen LSTM "expects to see."

2. **How retrieval is used**: Standard RAG concatenates/interpolates retrieved content with the query. Our retrieval translator uses retrieved windows as *cross-attention context* in a Transformer decoder, enabling selective feature-level borrowing from the target domain.

3. **Purpose**: Standard RAG retrieves content from the *same domain* to supplement incomplete information. Our retrieval bridges *between domains* — the retrieved target-domain windows provide translation references that show what the frozen model expects to see. This is retrieval for domain alignment, not knowledge augmentation.

4. **Instance-level, per-timestep**: Unlike kNN-LM (which interpolates output distributions) or TS-RAG (which retrieves at the sequence level), we perform k-NN retrieval at every timestep, enabling fine-grained temporal alignment.

**Connection to kNN-MT**: kNN-MT (Khandelwal et al., ICLR 2021) is the closest architectural ancestor — it uses a frozen model + per-timestep kNN from a domain-specific datastore for domain adaptation. Key difference: kNN-MT interpolates *output token distributions* at the decoder; we use retrieved embeddings as *cross-attention context* to guide input transformation. kNN-MT works within the same language; we bridge fundamentally different data distributions (eICU vs MIMIC feature spaces).

**Connection to RETRO**: RETRO's chunked cross-attention over retrieved document chunks (Borgeaud et al., ICML 2022) is the closest architectural precedent for our cross-attention mechanism. We extend this to time-series windows from a different domain, with k-NN lookup at each timestep rather than chunk-level retrieval.

**Retrieval for TS is an emerging paradigm** at top venues: RATD (NeurIPS 2024), TS-RAG (NeurIPS 2025), RAFT (ICML 2025). Our contribution extends this to domain adaptation specifically — no prior retrieval-augmented TS work addresses cross-domain translation.

### Citation Plan
- **Must cite**: RAG, kNN-LM, kNN-MT, RATD, TS-RAG (NeurIPS papers)
- **Should cite**: kNN-Prompt, RAM-EHR, RETRO
- **Optional**: RAFT

---

## Category 5: Foundation Models & Clinical AI

Foundation models represent an alternative paradigm: pre-train a large model on massive multi-site data, then transfer via fine-tuning or zero-shot. Our frozen-model approach is complementary — it enables adaptation without access to a large pre-trained model.

### Papers

| # | Paper | Venue | Approach | Key Difference from Ours |
|---|-------|-------|----------|--------------------------|
| 1 | YAIB (van de Water et al., 2024) | ICLR 2024 | Multi-center ICU benchmark framework | Our infrastructure; we build on YAIB |
| 2 | Med-BERT (Rasmy et al., 2021) | npj Dig Med 2021 | BERT pre-trained on 28.5M structured EHR patients | Pre-training paradigm; requires massive data |
| 3 | BEHRT (Li et al., 2020) | Sci. Rep. 2020 | Transformer for sequential EHR (1.6M patients) | Diagnosis sequences, not continuous TS |
| 4 | LLEMR (Wu et al., 2024) | NeurIPS 2024 | LLM instruction-tuned on 400K+ MIMIC-IV examples | LLM approach; requires massive compute |
| 5 | EHRSHOT/CLMBR (Wornow et al., 2023) | NeurIPS 2023 | 6,739-patient benchmark + 141M-param EHR FM for few-shot evaluation | FM benchmark; replaces deployed predictor |
| 6 | ICareFM (Burger et al., 2025) | medRxiv/ML4H 2025 | First ICU foundation model (650K stays, multi-hospital) | Scale-everything approach; 4000+ patient-years |
| 7 | Cross-Cohort KT (Zhang et al., 2025) | NeurIPS 2025 | Multimodal→unimodal knowledge transfer across cohorts | Cross-modal distillation; genomics+EHR→EHR |
| 8 | YAIB FM (2025) | NeurIPS 2025 sub | Self-supervised foundation model on pooled YAIB data | Same infrastructure; pre-trains new model |
| 9 | Harutyunyan et al. (2019) | Sci. Data 2019 | Multitask clinical benchmarks on MIMIC-III | Foundational benchmark |

### Our Case Against

**Reviewer concern**: "Foundation models solve this — why use a lightweight translator?"

**Response**: Foundation models (ICareFM, Med-BERT) and our approach address different practical scenarios:

| Factor | Foundation Models | Our Approach |
|--------|-------------------|--------------|
| Data requirement | 100K–650K stays from multiple hospitals | Source + target domain data only |
| Compute | Days/weeks of pre-training | Hours per experiment |
| Target model | New model (replaces existing) | Existing frozen model preserved |
| Regulatory | Requires re-validation of new model | Frozen model retains certification |
| Auditability | Black-box foundation model | Translator is modular; frozen predictor auditable |
| Model-freezing spectrum | No freeze | **Full freeze** (ours) |

Our approach is complementary, not competitive. When a hospital has a validated, deployed LSTM that cannot be retrained (regulatory, institutional, or practical constraints), input-space translation is the only viable adaptation strategy. Foundation models require replacing the predictor entirely. EHR foundation models such as CLMBR (Wornow et al., 2023) have shown strong few-shot transfer, but still require replacing the deployed predictor.

SHOT (Liang et al., ICML 2020) represents a middle ground — freezing the classifier head while adapting the feature extractor. Our constraint is more extreme: the *entire* model is frozen, requiring purely input-space adaptation.

**YAIB**: We build directly on van de Water et al. (2024) — our frozen baselines are YAIB-trained LSTMs, and we use YAIB's preprocessing pipeline and 5 task definitions.

### Citation Plan
- **Must cite**: YAIB (foundational), ICareFM (alternative paradigm), LLEMR (NeurIPS), EHRSHOT/CLMBR (NeurIPS FM benchmark)
- **Should cite**: Med-BERT, BEHRT, Cross-Cohort KT
- **Optional**: YAIB FM

---

## Category 6: Foundational DA Methods & Gradient Theory

These are the theoretical and methodological foundations of domain adaptation. Our DA baselines (Group 1) will be adapted versions of these methods. The gradient theory papers ground our cos(∇L_task, ∇L_fidelity) analysis.

### Papers

| # | Paper | Venue | Approach | Relation to Our Work |
|---|-------|-------|----------|---------------------|
| 1 | Ben-David et al. (2010) | MLJ 2010 | DA theory: target error ≤ source error + H-divergence | Our theoretical foundation; gradient alignment connects to H-divergence |
| 2 | Mansour, Mohri & Rostamizadeh (2009) | COLT 2009 | Discrepancy distance (tailored to loss + hypothesis class) | More practical than H-divergence for our setting |
| 3 | Zhang, Liu, Long & Jordan (2019) | ICML 2019 | Margin Disparity Discrepancy; bridges theory and algorithms | Modern extension; operationalizes theoretical bounds |
| 4 | DANN (Ganin et al., 2016) | JMLR 2016 | Gradient reversal layer + domain discriminator | Canonical baseline; adapted to frozen-model setting |
| 5 | Deep CORAL (Sun & Saenko, 2016) | ECCV-W 2016 | Second-order statistics alignment | Baseline; our MMD loss is related |
| 6 | DAN (Long et al., 2015) | ICML 2015 | Multi-kernel MMD for deep features | Foundation for our MMD alignment loss |
| 7 | MMD (Gretton et al., 2012) | JMLR 2012 | Kernel two-sample test; foundational distribution matching | Mathematical basis for our λ_align loss |
| 8 | OT for DA (Courty et al., 2017) | TPAMI 2017 | Optimal transport for domain adaptation | Alternative paradigm; related to OTTEHR |
| 9 | Sener & Koltun (2018) | NeurIPS 2018 | Multi-task learning as multi-objective optimization (MGDA) | Our task+fidelity is a 2-objective problem; Pareto framing |
| 10 | PCGrad (Yu et al., 2020) | NeurIPS 2020 | Gradient surgery: projects conflicting gradients onto normal plane | Uses exact same metric (cos < 0 = conflict); we diagnose, they intervene |
| 11 | CAGrad (Liu et al., 2021) | NeurIPS 2021 | Conflict-averse gradient descent for MTL | Convergence guarantees for gradient conflict resolution |
| 12 | Gradient Harmonization (Du et al., 2024) | IEEE TPAMI 2024 | Analyzes obtuse-angle gradients between alignment and classification in UDA | **Closest to our gradient analysis**; we extend to frozen-model setting |
| 13 | Prompt Gradient Alignment (Phan et al., 2024) | NeurIPS 2024 | Gradient alignment between domain losses for prompt-based DA | Same venue, same insight (gradient alignment in DA) |

### Our Case Against

**Reviewer concern**: "You should compare against DANN and CORAL."

**Response**: We agree — this is why DA baselines are our highest priority experiment. We adapt DANN, CORAL, and CoDATS to the frozen-model setting (learning an input transform instead of adapting the feature extractor) and show:

1. **Simple input transforms are insufficient**: DANN/CORAL in input-space have limited capacity for the complex, heterogeneous transformations needed across clinical domains.
2. **Our translators provide structured approaches**: Delta (additive residuals), SL (latent bottleneck), and Retrieval (target-domain grounding) each bring architectural inductive biases that generic input transforms lack.
3. **Gradient alignment explains the gap**: The cos(∇L_task, ∇L_fidelity) metric shows *why* certain methods struggle on certain tasks.

**Theoretical connection**: Ben-David et al.'s bound states ε_T ≤ ε_S + d_H(S,T) + λ\*. Our fidelity loss controls d_H by keeping translated samples close to the original source distribution. When fidelity and task gradients align (cos > 0), reducing fidelity loss simultaneously helps the task — an ideal regime. When they fight (cos < 0), the bound tension becomes irreconcilable without additional information (→ retrieval provides this). Mansour et al. (2009) provide a more refined bound via discrepancy distance, and Zhang et al. (2019) bridge this theory to practical algorithms.

**Gradient theory connection**: Our cos(∇L_task, ∇L_fidelity) diagnostic builds on the multi-objective optimization framework studied by Sener & Koltun (2018). The gradient conflict phenomenon we observe is analogous to what PCGrad (Yu et al., 2020) and CAGrad (Liu et al., 2021) address in multi-task learning, and what Gradient Harmonization (Du et al., 2024) and Phan et al. (2024) study in standard UDA. Our novelty is: (a) we identify this conflict in the *frozen-model* setting specifically, (b) we show that retrieval-augmented architecture *resolves the conflict structurally* rather than requiring gradient manipulation, and (c) the cos(task, fidelity) metric *predicts method-task suitability* a priori.

### Citation Plan
- **Must cite**: ALL foundational DA papers + PCGrad + Gradient Harmonization + Phan et al.
- **Should cite**: Sener & Koltun, CAGrad, Mansour et al., Zhang/Liu/Long/Jordan

---

## Cross-Cutting Themes for Related Work Section

### Theme 1: The Frozen-Model Gap
No prior work in clinical DA uses a frozen target predictor. This is our unique setting, motivated by regulatory compliance (the FDA's SaMD framework distinguishes "locked" algorithms with simpler regulatory pathways) and deployment practicality. SHOT (Liang et al., 2020) represents a partial freeze (classifier frozen, encoder adapted); our constraint is total model freeze.

**Papers bridging this gap**: VPT, Prompt Tuning, Voice2Series, Time-LLM, TATO — all from non-clinical domains. We are the first to apply this paradigm clinically.

### Theme 2: Input-Space vs Representation-Space
| Approach | Works in... | Examples |
|----------|-------------|---------|
| Feature alignment | Representation space | DANN, CORAL, CLUDA, VRADA |
| Input transformation | Input space | PixelDA, FDA, VPT, Voice2Series, Time-LLM, **Ours** |
| Pre-training | Weight space | Med-BERT, ICareFM, BEHRT |
| Partial freeze | Encoder adapted | SHOT |

Input-space DA has precedent in vision (PixelDA, FDA) but our work is the first to apply *learned neural input transformation* to clinical time-series domain adaptation.

### Theme 3: Retrieval for Domain Bridging
Standard RAG retrieves knowledge to augment a model. We retrieve *domain examples* to bridge distributions. kNN-MT (Khandelwal et al., 2021) is the closest precedent — it uses frozen model + per-step kNN from a domain-specific datastore for domain adaptation in NMT.

| System | Retrieves | Mechanism | Purpose | Domain |
|--------|-----------|-----------|---------|--------|
| RAG | Text passages | Concatenation | Knowledge augmentation | NLP |
| kNN-LM/MT | Output distributions | Interpolation | Domain adaptation | NLP |
| RETRO | Document chunks | Cross-attention | Knowledge augmentation | NLP |
| RAM-EHR | Medical knowledge | Feature augmentation | Clinical context | EHR |
| RATD / TS-RAG / RAFT | TS segments | Various | Forecasting guidance | General TS |
| **Ours** | **Target-domain TS windows** | **Cross-attention** | **Cross-domain translation** | **Clinical TS** |

### Theme 4: Gradient Dynamics in DA
Ben-David theory bounds target error but doesn't explain *training dynamics*. Our gradient alignment analysis (cos(task, fidelity)) provides an operational diagnostic: it predicts, during training, whether the multi-objective optimization will converge to a good solution.

Gradient conflicts have been studied in MTL (PCGrad, CAGrad, Sener & Koltun) and standard UDA (Gradient Harmonization, Phan et al.). Our contribution extends this to frozen-model DA specifically, showing that (a) the task-fidelity conflict is *structural* in the frozen-model setting (not tunable away), (b) the cos(task, fidelity) metric *predicts method-task suitability*, and (c) retrieval architecturally resolves the conflict by providing an alternative information path through cross-attention.

---

## Complete Paper Count by Category

| Category | Papers | Must-Cite | Should-Cite | Optional |
|----------|--------|-----------|-------------|----------|
| 1. Clinical/EHR DA | 11 | 5 | 4 | 2 |
| 2. General TS DA | 12 | 4 | 5 | 3 |
| 3. Frozen-Model / Input-Space | 13 | 6 | 5 | 2 |
| 4. Retrieval-Augmented | 9 | 5 | 3 | 1 |
| 5. Foundation Models | 9 | 4 | 3 | 1 |
| 6. Foundational DA + Gradient Theory | 13 | 10 | 3 | 0 |
| **Total** | **67** | **34** | **23** | **9** |

Target for paper: ~38–45 citations (34 must-cite + selective from should/optional).

---

## Suggested Related Work Section Structure (0.75 pages)

```latex
\section{Related Work}

\paragraph{Domain adaptation for clinical time-series.}
Clinical DA has been approached through representation alignment
(VRADA~\cite{purushotham2017}, AdaDiag~\cite{zhang2022}),
optimal transport (OTTEHR~\cite{ottehr2025}), and
pre-training (ICareFM~\cite{burger2025}).
All require access to model internals or train new models.
We instead adapt the \emph{input} to a frozen predictor---a
constraint motivated by regulatory requirements for locked
clinical algorithms.

\paragraph{Time-series domain adaptation.}
General TS-DA methods (CoDATS~\cite{wilson2020},
RAINCOAT~\cite{he2023}, CLUDA~\cite{ozyurt2023},
ACON~\cite{liu2024acon})
align feature distributions through the encoder.
Our frozen-model constraint precludes this;
we show in \S\ref{sec:baselines} that adapting these methods
to input-space underperforms our structured translators.

\paragraph{Frozen-model and input-space adaptation.}
Prompt tuning~\cite{lester2021} and VPT~\cite{jia2022}
learn input modifications for frozen models in NLP and vision.
Voice2Series~\cite{yang2021} and Time-LLM~\cite{jin2024}
reprogram frozen models for time-series tasks.
Concurrent work TATO~\cite{qiu2026} adapts data for frozen
TS foundation models via handcrafted transforms.
Our work extends this paradigm to clinical EHR with
(i)~a conditional neural translator,
(ii)~retrieval-augmented cross-attention, and
(iii)~gradient alignment theory.

\paragraph{Retrieval-augmented models.}
Our memory bank draws on kNN-LM~\cite{khandelwal2020}
and kNN-MT~\cite{khandelwal2021}, which augment frozen
models with domain-specific datastores.
RETRO~\cite{borgeaud2022} introduced cross-attention
over retrieved content.
Recent TS retrieval work
(RATD~\cite{liu2024ratd}, TS-RAG~\cite{ning2025})
augments forecasting; RAM-EHR~\cite{xu2024}
retrieves medical knowledge for EHR predictions.
We retrieve \emph{target-domain time-series windows}
for cross-domain translation---a novel application.

\paragraph{Domain adaptation theory and gradient dynamics.}
Ben-David et al.~\cite{bendavid2010} bound target error
by source error plus domain divergence.
Gradient conflicts between competing objectives have been
studied in MTL~\cite{yu2020pcgrad,liu2021cagrad} and
UDA~\cite{du2024gh,phan2024}.
We connect these ideas to frozen-model DA via the gradient
alignment condition (\S\ref{sec:theory}), showing that
$\cos(\nabla\mathcal{L}_\text{task}, \nabla\mathcal{L}_\text{fid}) > 0$
is necessary for stable translation, and that retrieval
architecturally resolves the conflict when it arises.
```

---

## Detailed Paper Entries

### 1. VRADA — Variational Recurrent Adversarial Deep Domain Adaptation
- **Citation**: Purushotham, S., Carvalho, W., Nilanon, T., & Liu, Y. (2017). Variational Recurrent Adversarial Deep Domain Adaptation. *ICLR 2017*.
- **Summary**: First to combine variational RNNs with adversarial DA for clinical time-series. Captures temporal latent dependencies and transfers them across domains.
- **Our differentiation**: VRADA works in representation space (feature alignment); requires end-to-end training of the full model. We keep the target model frozen and operate in input space. VRADA was evaluated on activity recognition and MIMIC-III (mortality only); we cover 5 tasks across 2 databases.

### 2. AdaDiag — Adversarial Domain Adaptation of Diagnostic Prediction
- **Citation**: Zhang, T., Xu, M., Gao, J., Chen, C., & Li, Y. (2022). AdaDiag: Adversarial Domain Adaptation of Diagnostic Prediction with Clinical Event Sequences. *Journal of Biomedical Informatics*, 134, 104178.
- **Summary**: Adversarial DA with a pre-trained Transformer for clinical event sequences (diagnosis codes). Adapts heart failure prediction from MIMIC-IV to UCLA Health.
- **Our differentiation**: Works on discrete event sequences (diagnosis codes), not continuous time-series. Requires retraining the backbone. Single task (heart failure) vs our 5 tasks.

### 3. METRE — Multidatabase ExTRaction PipEline
- **Citation**: Liao, W. & Voldman, J. (2023). METRE: A Multidatabase ExTRaction PipEline for Facile Cross Validation in Critical Care Research. *Journal of Biomedical Informatics*, 141, 104356.
- **Summary**: Standardized data extraction across MIMIC-IV and eICU (38K + 126K records). Same databases as ours. Achieves AUC 0.723–0.888 across tasks.
- **Our differentiation**: METRE is a preprocessing/harmonization pipeline, not an adaptation method. We use YAIB for preprocessing and add learned translation on top.

### 4. TransEHR — Alignment-free EHR Continual Learning
- **Citation**: Zhao, X., et al. (2026). TransEHR: Alignment-free Electronic Health Records Continual Learning across Feature Spaces. *Expert Systems with Applications*.
- **Summary**: Transformer + dual contrastive losses for continual learning when EHR feature spaces change over time.
- **Our differentiation**: Addresses temporal feature drift, not cross-hospital domain shift with fixed feature spaces. Different problem formulation.

### 5. ExtraCare — Concept-Grounded Orthogonal Inference
- **Citation**: (2026). Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference. *arXiv:2602.12542*.
- **Summary**: Decomposes patient representations into invariant/covariant components with orthogonality constraints. Evaluated on eICU and OCHIN.
- **Our differentiation**: Representation decomposition, not input-space translation. Focuses on interpretability through concept grounding. We share the eICU test bed.

### 6. Mutnuri et al. — DA + Transfer for ICU
- **Citation**: Mutnuri, M.K., et al. (2024). Using Domain Adaptation and Inductive Transfer Learning to Improve Patient Outcome Prediction in the Intensive Care Unit. *JMIR*, 2024.
- **Summary**: Compared DA vs transfer learning for ICU outcomes (mortality, AKI, LoS) between eCritical and MIMIC-III.
- **Our differentiation**: Overlapping tasks. They fine-tune the target model; we keep it frozen.

### 7. OTTEHR — Optimal Transport for EHR Transfer
- **Citation**: (2024). Transport-based Transfer Learning on Electronic Health Records. *JAMIA*, 33(1), 15, 2025.
- **Summary**: OT for unsupervised transfer between EHR populations (MIMIC-III/IV, eICU).
- **Our differentiation**: OT-based; population-level transport. No temporal modeling. We use neural translation with temporal attention.

### 8. CoDATS — Multi-Source Deep Domain Adaptation for Time-Series
- **Citation**: Wilson, G., Doppa, J.R., & Cook, D.J. (2020). Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data. *KDD 2020*.
- **Summary**: 1D CNN + gradient reversal for time-series sensor data. Supports multi-source and weak supervision.
- **Our differentiation**: Sensor data (HAR), not clinical EHR. Feature-space alignment; we operate in input space. We adapt CoDATS to our frozen-model setting as a baseline.

### 9. CALDA — Contrastive Adversarial for Multi-Source TS DA
- **Citation**: Wilson, G., Doppa, J.R., & Cook, D.J. (2023). Improving Multi-Source Time Series Domain Adaptation with Contrastive Adversarial Learning. *IEEE TPAMI*, 2023.
- **Summary**: Extension of CoDATS with contrastive + adversarial alignment for multi-source TS DA.
- **Our differentiation**: Same authors as CoDATS; still representation-space alignment.

### 10. RAINCOAT — DA for Time Series Under Feature and Label Shifts
- **Citation**: He, H., et al. (2023). Domain Adaptation for Time Series Under Feature and Label Shifts. *ICML 2023*.
- **Summary**: First model for both closed-set and universal DA on time-series, handling feature and label shifts. Up to 16.33% improvement.
- **Our differentiation**: Handles label shift (relevant: sepsis 1.13% vs AKI 11.95%), but operates in representation space. Not frozen-model.

### 11. CLUDA — Contrastive Learning for Unsupervised DA of Time Series
- **Citation**: Ozyurt, Y., Feuerriegel, S., & Zhang, C. (2023). Contrastive Learning for Unsupervised Domain Adaptation of Time Series. *ICLR 2023*.
- **Summary**: Nearest-neighbor contrastive learning for contextual representations that preserve label information.
- **Our differentiation**: kNN-based contrastive in representation space; ours uses kNN for cross-attention context in input space.

### 12. AdvSKM — Adversarial Spectral Kernel Matching
- **Citation**: Liu, Q. & Xue, H. (2021). Adversarial Spectral Kernel Matching for Unsupervised Time Series Domain Adaptation. *IJCAI 2021*.
- **Summary**: Spectral kernel + MMD for time-series UDA.
- **Our differentiation**: Spectral-kernel MMD in representation space. Our MMD alignment loss operates in latent space and complements (rather than replaces) task-driven translation.

### 13. POND — Prompt Tuning for TS DA
- **Citation**: (NEC Labs, 2024). POND: Multi-Source Time Series Domain Adaptation with Information-Aware Prompt Tuning. *KDD 2024*.
- **Summary**: First prompt-based TS DA framework. Three-step: pretrain, prompt tune, prompt adapt. +66% F1.
- **Our differentiation**: Prompts ≈ delta approach conceptually. But POND tunes for a trainable model, not frozen.

### 14. ACON — Boosting Transferability and Discriminability for TS DA
- **Citation**: Liu, J., et al. (2024). Boosting Transferability and Discriminability for Time Series Domain Adaptation. *NeurIPS 2024*.
- **Summary**: Analyzes temporal vs frequency feature transferability in TS-DA.
- **Our differentiation**: Feature-space; not frozen-model. Our constraint precludes joint temporal-frequency adaptation in the encoder.

### 15. MAPU — Source-Free DA for Time Series
- **Citation**: Ragab, M., et al. (2023). Source-Free Domain Adaptation with Temporal Imputation for Time Series Data. *KDD 2023*.
- **Summary**: First source-free DA method for time series.
- **Our differentiation**: Source-free restricts *data access*; our constraint restricts *model access*. Orthogonal constraints — MAPU still modifies the target model.

### 16. ADATIME — Benchmarking Suite for TS DA
- **Citation**: Ragab, M., Eldele, E., et al. (2023). ADATIME: A Benchmarking Suite for Domain Adaptation on Time Series Data. *ACM TKDD*, 2023.
- **Summary**: Standard benchmarking suite for TS-DA (11 methods, 50 scenarios).
- **Our differentiation**: Benchmark paper. Establishes evaluation standards we follow.

### 17. PixelDA — Unsupervised Pixel-Level Domain Adaptation
- **Citation**: Bousmalis, K., Silberman, N., Dohan, D., Erhan, D., & Krishnan, D. (2017). Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks. *CVPR 2017*.
- **Summary**: GAN-based pixel-level (input-space) domain adaptation that transforms source images to look like target domain.
- **Our differentiation**: The original input-space DA in vision. Model is not frozen; uses adversarial fidelity instead of task-driven fidelity through a frozen predictor. Vision, not clinical time-series.

### 18. Adversarial Reprogramming of Neural Networks
- **Citation**: Elsayed, G.F., Sohl-Dickstein, J., & Goodfellow, I. (2019). Adversarial Reprogramming of Neural Networks. *ICLR 2019*.
- **Summary**: Universal additive perturbation repurposes a frozen ImageNet classifier for different tasks. Network weights frozen.
- **Our differentiation**: Single universal perturbation (not sample-conditional). Cross-task reprogramming, not cross-domain. Our delta translator is a conditional, sample-specific extension.

### 19. Prompt Tuning — The Power of Scale
- **Citation**: Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. *EMNLP 2021*.
- **Summary**: Learned soft prompts condition frozen LMs for downstream tasks. Competitive with full fine-tuning at scale.
- **Our differentiation**: Fixed-length prompt vector vs our full conditional neural translator. Our approach is sample-specific and handles variable-length sequences.

### 20. Voice2Series — Reprogramming Acoustic Models for Time Series
- **Citation**: Yang, C.-H.H., Tsai, Y.-Y., & Chen, P.-Y. (2021). Voice2Series: Reprogramming Acoustic Models for Time Series Classification. *ICML 2021*.
- **Summary**: Reprograms a frozen acoustic model for time-series classification via input transformation + output label mapping. Matches SOTA on 22/31 tasks.
- **Our differentiation**: Cross-task reprogramming (speech→time-series), not cross-domain. Single transform function, not retrieval-augmented.

### 21. VPT — Visual Prompt Tuning
- **Citation**: Jia, M., et al. (2022). Visual Prompt Tuning. *ECCV 2022*.
- **Summary**: Learnable input tokens (<1% of model params) for frozen ViTs.
- **Our differentiation**: Prepended tokens vs full input transformation. Vision domain. Our translator is orders of magnitude larger.

### 22. Time-LLM — Time Series Forecasting by Reprogramming LLMs
- **Citation**: Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J.Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., & Wen, Q. (2024). Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. *ICLR 2024*.
- **Summary**: Reprograms a frozen LLM (GPT-2/LLaMA) for TS forecasting via text prototype reprogramming + Prompt-as-Prefix. 1000+ citations.
- **Our differentiation**: Cross-modality (language→TS) not cross-domain (eICU→MIMIC). Generic pretrained LLM, not task-specific validated clinical LSTM. Forecasting, not clinical prediction. No retrieval component.

### 23. Model Reprogramming Survey
- **Citation**: Chen, P.-Y. (2024). Model Reprogramming: Resource-Efficient Cross-Domain Machine Learning. *AAAI 2024*.
- **Summary**: Survey/tutorial unifying model reprogramming as input transformation + output mapping for frozen models.
- **Our differentiation**: Taxonomy paper. Frames our approach within the broader reprogramming paradigm.

### 24. SHOT — Source Hypothesis Transfer
- **Citation**: Liang, J., Hu, D., & Feng, J. (2020). Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. *ICML 2020*.
- **Summary**: Adapts frozen classifier head to target domain via information maximization + pseudo-labeling of features.
- **Our differentiation**: Partial freeze (classifier frozen, encoder adapted). Our constraint is total model freeze — we cannot modify any weights.

### 25. FDA — Fourier Domain Adaptation
- **Citation**: Yang, Y. & Soatto, S. (2020). FDA: Fourier Domain Adaptation for Semantic Segmentation. *CVPR 2020*.
- **Summary**: Swaps low-frequency Fourier spectrum between domains for training-free input-space DA.
- **Our differentiation**: Fixed handcrafted transform, not learned. Vision domain. Shows input-space DA is a known concept; our contribution is learned neural translation for clinical TS.

### 26. Side-Tuning
- **Citation**: Zhang, J.O., Sax, A., Zamir, A., Guibas, L., & Malik, J. (2020). Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks. *ECCV 2020*.
- **Summary**: Trains lightweight side network whose output is summed with frozen pretrained network's output.
- **Our differentiation**: Additive output modification. Conceptually related to our delta translator (additive input modification).

### 27. One Fits All / FPT — Power of Pretrained LM
- **Citation**: Zhou, T., et al. (2023). One Fits All: Power General Time Series Analysis by Pretrained LM. *NeurIPS 2023*.
- **Summary**: Frozen Pretrained Transformer (GPT-2) for time-series analysis. Fine-tunes only positional embeddings and layer norms.
- **Our differentiation**: Adapts internal components (layer norms), not inputs. General TS analysis, not domain adaptation.

### 28. L2C — Learning to Adapt Frozen CLIP
- **Citation**: Chi, L., Gu, B., Liu, G., Wang, Y., Wu, Y., Wang, Z., & Plataniotis, K.N. (2025). Learning to Adapt Frozen CLIP for Few-Shot Test-Time Domain Adaptation. *ICLR 2025*.
- **Summary**: Input-space learning with a side branch ("revert attention") to complement frozen CLIP for test-time DA.
- **Our differentiation**: Vision (CLIP) vs clinical time-series (LSTM). Test-time DA vs training-time DA.

### 29. TATO — Adaptive Transformation Optimization
- **Citation**: Qiu, Y., Cen, Y., Pei, J., Wang, H., & Wang, W. (2026). Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models. *ICLR 2026*.
- **Summary**: Adapts data through automated transformation pipelines for frozen large time-series models. Up to 65.4% MSE reduction.
- **Our differentiation**: Handcrafted transformation operators (slicing, normalization, outlier correction) vs our learned neural translation. General forecasting vs clinical DA. No retrieval component. No theoretical analysis.
- **Note**: Most closely related concurrent work. Same philosophy ("adapt data, not model") but different execution.

### 30. RAG — Retrieval-Augmented Generation
- **Citation**: Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- **Summary**: Combines pre-trained parametric (seq2seq) and non-parametric (dense vector index) memory for knowledge-intensive NLP.
- **Our differentiation**: Foundational RAG paper. We retrieve time-series windows, not text passages.

### 31. kNN-LM — Generalization through Memorization
- **Citation**: Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., & Lewis, M. (2020). Generalization through Memorization: Nearest Neighbor Language Models. *ICLR 2020*.
- **Summary**: Extends frozen LM with kNN retrieval from a datastore. Domain adaptation by swapping datastores without retraining.
- **Our differentiation**: Interpolates output distributions; we use retrieved embeddings as cross-attention context.

### 32. kNN-MT — Nearest Neighbor Machine Translation
- **Citation**: Khandelwal, U., Fan, A., Jurafsky, D., Zettlemoyer, L., & Lewis, M. (2021). Nearest Neighbor Machine Translation. *ICLR 2021*.
- **Summary**: Augments a frozen NMT model with k-NN retrieval from a domain-specific datastore at each decoding step. Domain adaptation by swapping datastores. +9.2 BLEU average over zero-shot.
- **Our differentiation**: Closest architectural ancestor. kNN-MT interpolates *output token distributions*; we use retrieved embeddings as *cross-attention context* for input transformation. kNN-MT works within the same language; we bridge fundamentally different data distributions.

### 33. kNN-Prompt — Nearest Neighbor Zero-Shot Inference
- **Citation**: Shi, W., et al. (2022). Nearest Neighbor Zero-Shot Inference. *EMNLP 2022*.
- **Summary**: Augments frozen GPT-2 with kNN retrieval for zero-shot classification.
- **Our differentiation**: Frozen model + kNN paradigm but in NLP, for classification (not translation).

### 34. RETRO — Retrieval-Enhanced Transformer
- **Citation**: Borgeaud, S., Mensch, A., Hoffmann, J., et al. (2022). Improving Language Models by Retrieving from Trillions of Tokens. *ICML 2022*.
- **Summary**: Conditions generation on retrieved document chunks via chunked cross-attention. Uses frozen BERT retriever + differentiable encoder.
- **Our differentiation**: Closest cross-attention architectural precedent. RETRO retrieves text chunks with chunk-level cross-attention; we retrieve target-domain TS windows with per-timestep cross-attention.

### 35. RATD — Retrieval-Augmented Diffusion for TS Forecasting
- **Citation**: Liu, J., et al. (2024). Retrieval-Augmented Diffusion Models for Time Series Forecasting. *NeurIPS 2024*.
- **Summary**: Retrieves relevant time series to guide diffusion model denoising for forecasting.
- **Our differentiation**: Diffusion-based forecasting, not domain adaptation.

### 36. TS-RAG — Retrieval-Augmented TS Foundation Models
- **Citation**: Ning, K., et al. (2025). TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster. *NeurIPS 2025*.
- **Summary**: Retrieves semantically relevant segments with an Adaptive Retrieval Mixer for zero-shot forecasting.
- **Our differentiation**: Foundation model augmentation for forecasting; we augment a frozen LSTM for DA.

### 37. RAM-EHR — Retrieval Augmentation for Clinical Predictions
- **Citation**: Xu, R., et al. (2024). RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records. *ACL 2024*.
- **Summary**: Retrieves from PubMed/knowledge graphs to augment EHR predictions. +3.4% AUROC, +7.2% AUPR.
- **Our differentiation**: Retrieves textual medical knowledge; we retrieve target-domain time-series windows.

### 38. RAFT — Retrieval-Augmented TS Forecasting
- **Citation**: (2025). Retrieval Augmented Time Series Forecasting. *ICML 2025* (PMLR v267).
- **Summary**: Retrieval-augmented forecasting with inductive biases from retrieved examples.
- **Our differentiation**: Forecasting, not domain adaptation.

### 39. YAIB — Yet Another ICU Benchmark
- **Citation**: van de Water, R., et al. (2024). Yet Another ICU Benchmark: A Flexible Multi-Center Framework for Clinical ML. *ICLR 2024*.
- **Summary**: Modular framework for reproducible clinical ML across MIMIC-III/IV, eICU, HiRID, AUMCdb. Five tasks.
- **Our differentiation**: We build directly on YAIB — our frozen baselines are YAIB-trained LSTMs.

### 40. Med-BERT
- **Citation**: Rasmy, L., et al. (2021). Med-BERT: Pretrained Contextualized Embeddings on Large-Scale Structured Electronic Health Records for Disease Prediction. *npj Digital Medicine*, 4(86).
- **Summary**: BERT pre-trained on 28.5M patients for disease prediction. +1.21–6.14% AUC.
- **Our differentiation**: Pre-training paradigm requiring massive data. We adapt with only source + target data.

### 41. BEHRT — Transformer for EHR
- **Citation**: Li, Y., et al. (2020). BEHRT: Transformer for Electronic Health Records. *Scientific Reports*, 10.
- **Summary**: Treats diagnosis codes as language. Pre-trained on 1.6M patients.
- **Our differentiation**: Diagnosis sequences (discrete), not continuous time-series. Pre-training, not frozen-model DA.

### 42. LLEMR — LLM for EHR
- **Citation**: Wu, Z., et al. (2024). Instruction Tuning Large Language Models to Understand Electronic Health Records. *NeurIPS 2024* (DB track, Spotlight).
- **Summary**: LLM instruction-tuned on 400K+ MIMIC-IV examples. Comparable to SOTA clinical prediction.
- **Our differentiation**: Massive LLM approach. Our lightweight translator achieves strong results without LLMs.

### 43. EHRSHOT/CLMBR — EHR Foundation Model Benchmark
- **Citation**: Wornow, M., et al. (2023). EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models. *NeurIPS 2023* (Datasets & Benchmarks, Spotlight).
- **Summary**: 6,739-patient benchmark + 141M-param CLMBR foundation model for few-shot EHR evaluation.
- **Our differentiation**: FM benchmark. Strong few-shot transfer, but requires replacing the deployed predictor.

### 44. ICareFM — ICU Foundation Model
- **Citation**: Burger, M., et al. (2025). A Foundation Model for Intensive Care. *medRxiv 2025*; ML4H 2025 Spotlight.
- **Summary**: First ICU foundation model: 650K stays, multi-hospital, self-supervised time-to-event objective. Zero-shot transfer.
- **Our differentiation**: Scale-everything approach. Requires replacing the predictor; we preserve it frozen.

### 45. Cross-Cohort Knowledge Transfer
- **Citation**: Zhang, Q., et al. (2025). Democratizing Clinical Risk Prediction with Cross-Cohort Cross-Modal Knowledge Transfer. *NeurIPS 2025*.
- **Summary**: Transfers multimodal (EHR+genomics) knowledge to EHR-only local cohorts.
- **Our differentiation**: Cross-modal distillation, not cross-domain adaptation within same modality.

### 46. YAIB Foundation Model
- **Citation**: (2025). Towards Self-Supervised Foundation Models for Critical Care Time Series. *Submitted to NeurIPS 2025*.
- **Summary**: Bi-Axial Transformer pre-trained on pooled YAIB data. Effective transfer to unseen datasets.
- **Our differentiation**: Pre-trains a new model; we adapt inputs for existing frozen model.

### 47. Harutyunyan et al. (2019) — Multitask Clinical Benchmarks
- **Citation**: Harutyunyan, H., et al. (2019). Multitask Learning and Benchmarking with Clinical Time Series Data. *Scientific Data*, 6(96).
- **Summary**: Four clinical prediction benchmarks on MIMIC-III.
- **Our differentiation**: Foundational benchmark.

### 48. Purushotham et al. (2018) — Benchmarking Deep Learning
- **Citation**: Purushotham, S., et al. (2018). Benchmarking Deep Learning Models on Large Healthcare Datasets. *Journal of Biomedical Informatics*, 83, 112–134.
- **Summary**: Benchmarked deep learning on MIMIC-III for mortality, LoS, ICD-9.
- **Our differentiation**: Benchmark paper; no DA.

### 49. Ben-David et al. — DA Theory
- **Citation**: Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J.W. (2010). A Theory of Learning from Different Domains. *Machine Learning*, 79, 151–175.
- **Summary**: Bounds target error by source error + H-divergence + ideal joint error. Foundational DA theory.
- **Our differentiation**: Our gradient alignment condition connects to the H-divergence bound operationally.

### 50. Mansour, Mohri & Rostamizadeh — Discrepancy Distance
- **Citation**: Mansour, Y., Mohri, M., & Rostamizadeh, A. (2009). Domain Adaptation: Learning Bounds and Algorithms. *COLT 2009*.
- **Summary**: Introduces discrepancy distance tailored to loss function and hypothesis class. Alternative to H-divergence.
- **Our differentiation**: More practical than H-divergence for our setting. Complements Ben-David.

### 51. Zhang, Liu, Long & Jordan — Margin Disparity Discrepancy
- **Citation**: Zhang, Y., Liu, T., Long, M., & Jordan, M.I. (2019). Bridging Theory and Algorithm for Domain Adaptation. *ICML 2019*.
- **Summary**: Introduces MDD; bridges DA theory with practical adversarial algorithms.
- **Our differentiation**: Modern bridge between theory and algorithms. Strengthens our theoretical framing.

### 52. DANN — Domain-Adversarial Neural Networks
- **Citation**: Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. *JMLR*, 17, 1–35.
- **Summary**: Gradient reversal layer + domain discriminator. Features trained to be task-discriminative but domain-invariant.
- **Our differentiation**: Canonical baseline. We adapt to frozen-model setting (learn input transform, not feature alignment).

### 53. Deep CORAL
- **Citation**: Sun, B. & Saenko, K. (2016). Deep CORAL: Correlation Alignment for Deep Domain Adaptation. *ECCV Workshops 2016*.
- **Summary**: Aligns second-order statistics (covariance) of source/target deep features.
- **Our differentiation**: Covariance alignment in feature space. Our MMD loss is conceptually related. We adapt to input space.

### 54. DAN — Deep Adaptation Networks
- **Citation**: Long, M., et al. (2015). Learning Transferable Features with Deep Adaptation Networks. *ICML 2015*.
- **Summary**: Multi-kernel MMD for matching task-specific layer distributions.
- **Our differentiation**: Foundation for our MMD alignment loss (λ_align). We apply in latent space.

### 55. MMD — Kernel Two-Sample Test
- **Citation**: Gretton, A., et al. (2012). A Kernel Two-Sample Test. *JMLR*, 13, 723–773.
- **Summary**: Defines MMD as the largest expectation difference in RKHS. Foundation for distribution matching.
- **Our differentiation**: Mathematical basis for our alignment loss.

### 56. OT for DA
- **Citation**: Courty, N., et al. (2017). Optimal Transport for Domain Adaptation. *IEEE TPAMI*, 39(9), 1853–1865.
- **Summary**: Regularized OT for domain adaptation with class-preserving constraints.
- **Our differentiation**: Alternative to MMD. Related to OTTEHR. We use MMD, not OT.

### 57. Sener & Koltun — Multi-Task Learning as Multi-Objective Optimization
- **Citation**: Sener, O. & Koltun, V. (2018). Multi-Task Learning as Multi-Objective Optimization. *NeurIPS 2018*.
- **Summary**: Frames MTL as MOO; proposes MGDA-UB for Pareto-optimal gradient directions.
- **Our differentiation**: Our task + fidelity loss is a 2-objective problem. Sener & Koltun provide the Pareto framing.

### 58. PCGrad — Gradient Surgery for Multi-Task Learning
- **Citation**: Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient Surgery for Multi-Task Learning. *NeurIPS 2020*.
- **Summary**: Defines gradient conflict as negative cosine similarity between task gradients; projects conflicting gradients onto the normal plane.
- **Our differentiation**: Uses the exact same metric (cos < 0 = conflict) we use. PCGrad intervenes algorithmically; we diagnose the conflict and show retrieval resolves it architecturally.

### 59. CAGrad — Conflict-Averse Gradient Descent
- **Citation**: Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q. (2021). Conflict-Averse Gradient Descent for Multi-task Learning. *NeurIPS 2021*.
- **Summary**: Finds update direction maximizing worst-case local improvement among all objectives, with provable convergence.
- **Our differentiation**: Provides convergence guarantees for gradient conflict resolution. Complements PCGrad.

### 60. Gradient Harmonization
- **Citation**: Du, Y., et al. (2024). Gradient Harmonization in Unsupervised Domain Adaptation. *IEEE TPAMI*, 2024.
- **Summary**: Directly analyzes obtuse-angle (conflicting) gradients between domain alignment and classification in UDA. Proposes GH (rotate to acute) and GH++ (rotate to orthogonal).
- **Our differentiation**: **Closest existing work to our gradient analysis.** They study task vs. alignment gradient conflict in standard UDA. Our novelty: (a) frozen-model setting, (b) retrieval resolves the conflict architecturally rather than requiring gradient manipulation, (c) cos(task, fidelity) predicts method-task suitability.

### 61. Prompt Gradient Alignment
- **Citation**: Phan, H., Tran, L., Tran, Q., & Le, T. (2024). Enhancing Domain Adaptation through Prompt Gradient Alignment. *NeurIPS 2024*.
- **Summary**: Casts UDA as multi-objective optimization, aligns per-domain gradients for prompt-based adaptation of frozen vision-language models.
- **Our differentiation**: Same venue, same insight (gradient alignment in DA). They align gradients between domain losses for prompt learning; we analyze alignment between task and fidelity losses for frozen-model translation. Different objective decomposition.

### 62–67. Other papers (Driever, KnowRare, TransEHR, Multi-View CL, DAAC, TS DA Benchmark)
These are listed in the summary tables above with full details. See Category 1 and 2 paper tables for venue and differentiation info.
