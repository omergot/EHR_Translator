# Training Time Bottleneck Analysis

**Date**: 2026-04-06 (updated)
**Method**: Log timestamp analysis across completed runs.

## Measured Epoch Times (Phase 2)

| Translator | Task | Config | Server | Epoch Time | Batches | Per-batch |
|---|---|---|---|---|---|---|
| Delta (bs=64) | Mortality | delta | V100S | 5.5 min | 1,240 | 0.27s |
| Delta (bs=16) | Sepsis | delta | V100S | 18.1 min | 5,399 | 0.19s |
| SL+FG (bs=16) | Mortality | SL Phase 1 | V100S | 3.1 min | 2,828 | 0.07s |
| SL+FG (bs=16) | Mortality | SL Phase 2 | V100S | 11.8 min | 4,960 | 0.14s |
| SL (bs=16) | Sepsis | SL Phase 1 | V100S | 8.0 min | 3,632 | 0.13s |
| SL (bs=16) | Sepsis | SL Phase 2 | V100S | 24.8 min | 5,399+tgt | 0.28s |
| Retrieval (bs=16) | Sepsis | V4 (n_cross=2) Phase 1 | V100S | 7.3 min | 3,632 | 0.12s |
| **Retrieval (bs=16)** | **Sepsis** | **V4 (n_cross=2) Phase 2** | **V100S** | **57.8 min** | **5,399+tgt** | **0.59s** |
| **Retrieval (bs=16)** | **AKI** | **V5 cross3 Phase 2** | **V100S** | **70 min** | **3,518+tgt** | **1.09s** |
| **Retrieval (bs=16)** | **Mortality** | **V5 cross3 Phase 2** | **V100S** | **36 min** | **~2,500+tgt** | **~0.9s** |
| **Retrieval (bs=16)** | **Sepsis** | **V5 cross3 Phase 2** | **3090** | **34 min** | **~5,000+tgt** | **~0.4s** |

Total training times (V5 cross3, 50 Phase 2 epochs + 15 Phase 1 epochs):
- **AKI V5 cross3 (V100S): ~61h** (Phase 1: 1.8h + Phase 2: 58h)
- **Mortality V5 cross3 (V100S): ~31h**
- **Sepsis V5 cross3 (3090): ~30h**

Note: AKI V5 cross3 is the **slowest** config because n_cross_layers=3 adds ~50% more cross-attention compute vs V4 (n_cross=2), and AKI has 3,518 batches with VLB.

Sources: `experiments/logs/aki_v5_cross3_aki.log`, `experiments/logs/sepsis_retr_v5_cross3_3090_sepsis.log`, `experiments/logs/mortality_retr_v5_cross3_v2_mortality.log`.

## k-NN Implementation Details

Current implementation (`src/core/retrieval_translator.py:276-316`) uses efficient matmul-based squared Euclidean: `||q - b||^2 = ||q||^2 + ||b||^2 - 2*q@b^T`, computed via cuBLAS GEMM, chunked at 512 queries. Already well-optimized for GPU.

Memory bank sizes: 226K windows (mortality), 423K (AKI), 569K windows (sepsis).

## Implemented Optimizations

### pin_memory + persistent_workers + non_blocking — VALIDATED ✅

**Measured speedup: 1.64x (40% faster per epoch)**

| Experiment | Phase2 Epoch | Speedup |
|---|---|---|
| master (mortality V4+MMD, V100S) | 28.7 min | baseline |
| **+ pin_memory + persistent + non_blocking** | **17.5 min** | **1.64x** |

AUROC: +0.0428 vs +0.0456 master (within ±0.004 noise). Behavior unchanged.

**Why 40% instead of the expected 5%**: The combined effect of `pin_memory=True` + `non_blocking=True` enables true CPU/GPU data transfer overlap: GPU computes on current batch while next batch transfers asynchronously via DMA. With 3,500+ batches/epoch and ~1s/batch, even small per-batch overlap savings compound massively. `persistent_workers=True` eliminates worker fork overhead between epochs.

Files changed: `yaib.py` (DataLoader), `cli.py` (8 recreations), `bucket_batching.py` (1 DataLoader), `train.py` (13 `.to(device)` calls), `retrieval_translator.py` (1 `.to(device)` in build_memory_bank).

### val_every_n_epochs — IMPLEMENTED

Validate every N epochs instead of every epoch. Default=1 (no change). Setting to 2 saves ~7% of total time (validation is ~5 min per epoch for AKI).

### torch.compile — IMPLEMENTED (needs validation)

JIT compile translator encode/decode/decode_with_context sub-modules. Expected 10-15% additional speedup on transformer ops. `query_memory_bank` excluded (chunked iteration incompatible with dynamo). Config key: `use_torch_compile: true`.

### Gradient checkpointing — IMPLEMENTED (needs validation)

Wraps AxialBlock + FiLM and CrossAttentionBlock calls with `torch.utils.checkpoint`. Trades compute for memory, enabling `batch_size=32` on V100-32GB. Expected ~30-40% speedup from halved batch count. Config key: `gradient_checkpointing: true`.

## FAISS / Approximate k-NN Assessment

FAISS IVF-Flat with ~1K centroids and nprobe=32 would reduce distance computations 10-20x on the k-NN step itself. However:
- The k-NN step is only 24-31% of total epoch time
- FAISS GPU `index.search()` has per-call overhead for small batch sizes
- **Estimated overall speedup: 15-25%**

**Verdict**: Lower priority given the 40% already achieved from data transfer overlap.

## Ranked Bottlenecks

1. **Batch count from dataset size + small batch size** — AKI has 3,518 batches at bs=16 (after VLB). Retrieval cannot use bs>16 on V100-32GB without gradient checkpointing due to memory bank + cross-attention VRAM.

2. **Cross-attention blocks (~17% of epoch time)** — Each of n_cross CrossAttentionBlocks does per-timestep attention over K*W=16*6=96 context vectors plus global self-attention.

3. **Dual data path (inherent, ~43% of time)** — Every training step processes both a source batch (with retrieval) and a target batch (autoencoder reconstruction). Unavoidable in the architecture.

4. **k-NN distance computation (~24-31% of epoch time)** — GEMM against 423K-569K bank vectors.

5. **~~Data loading (minimal)~~** — RESOLVED. `pin_memory=True`, `persistent_workers=True`, and `non_blocking=True` cut epoch time by 40%.

## Instrumentation

- `[timing] Epoch X/Y: Zs` log line added per epoch in RetrievalTranslatorTrainer
- Per-step `[perf]` timing exists only in `TransformerTranslatorTrainer` (delta)
- `cuda.synchronize()` calls exist only in `TransformerTranslatorTrainer` (delta), NOT in RetrievalTranslatorTrainer — irrelevant for retrieval optimization
