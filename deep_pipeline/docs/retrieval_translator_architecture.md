# RetrievalTranslator Architecture

> **Role**: Model architecture reference for the retrieval-guided translator.
> **See also**: [architecture.md](architecture.md) (delta translator), [shared_latent_results.md](shared_latent_results.md) (shared latent results), [gradient_bottleneck_analysis.md](gradient_bottleneck_analysis.md) (gradient analysis)

**Date**: Feb 25, 2026
**File**: `src/core/retrieval_translator.py` (class `RetrievalTranslator`)

---

## Overview

The RetrievalTranslator is the third paradigm for domain adaptation. Instead of global distribution alignment (SL's MMD) or pure delta correction, it performs **instance-level retrieval**: each eICU timestep finds its K nearest MIMIC neighbors and fuses their information via cross-attention before decoding.

**Why SL fails on sepsis (the reconstruction bottleneck)**: SL's encoder-decoder must reconstruct all 169×48 features through the latent space. At sepsis's 1.13% positive rate, there's ~1 task gradient per sequence vs ~8,000 reconstruction gradients — reconstruction dominates and the task signal cannot steer the latent space. This is NOT a per-timestep/causal issue (AKI is also per-timestep causal and SL succeeds at +0.0370 AUCROC with 11.95% label density). It's purely a label sparsity × reconstruction bottleneck interaction.

Key advantages over SL for sparse-label tasks:
- **Instance-level matching** (not global MMD) — retrieved neighbors carry task-relevant structure, providing richer gradient signal than global distribution alignment
- **Cross-attention fusion** — unlike reconstruction loss which is dense and task-agnostic, cross-attention lets the model selectively attend to task-relevant features from retrieved neighbors
- Fully causal retrieval (backward-looking windows) — native sepsis/AKI support
- Explicit memory bank — interpretable nearest-neighbor retrieval

---

## Architecture Diagram

```
Input: x_val (B,T,48), x_miss (B,T,48), t_abs (B,T), m_pad (B,T), x_static (B,4)
                                    │
                    ┌───────────────┴───────────────┐
                    │     SHARED ENCODER             │
                    │  (Same as SharedLatentTransl.)  │
                    │                                │
                    │  triplet_proj + sensor_emb     │
                    │  lift + time_proj              │
                    │  4x AxialBlock + FiLM          │
                    │  mean-pool features → to_latent│
                    └───────────────┬───────────────┘
                                    │  latent = (B, T, d_latent=128)
                    ┌───────────────┼───────────────┐
                    │               │               │
              importance_weights    │         label_pred_head
              (d_latent,) sigmoid   │         (pretrain only)
                    │               │
                    ▼               │
        ┌──────────────────┐       │
        │   MEMORY BANK    │       │
        │  (pre-encoded    │       │
        │   MIMIC windows) │       │
        │  N_windows ×     │       │
        │  d_latent (GPU)  │       │
        └────────┬─────────┘       │
                 │                 │
        query_memory_bank()        │
        backward-looking pool      │
        K-NN per timestep          │
                 │                 │
                 ▼                 │
        context: (B,T,K*W,d_latent)│
                 │                 │
                 └────────┬────────┘
                          │
                    ┌─────┴──────────────────────┐
                    │   CROSS-ATTENTION BLOCKS    │
                    │   (2x CrossAttentionBlock)  │
                    │                             │
                    │  1. Per-TS cross-attn:      │
                    │     Q=eICU, KV=MIMIC nbrs   │
                    │  2. Global causal self-attn  │
                    │  3. FFN                      │
                    └─────────────┬───────────────┘
                                  │  h = (B, T, d_model)
                    ┌─────────────┴───────────────┐
                    │         DECODER              │
                    │  from_latent → broadcast F   │
                    │  + dec_feature_emb           │
                    │  2x AxialBlock + FiLM        │
                    │  output_head Linear(d,1)     │
                    └─────────────┬───────────────┘
                                  │  (B, T, 48)
                                  │
                        x_out = x_val + decoded   ← residual (output_mode="residual")
```

---

## Components

### MemoryBank (dataclass)

Pre-encoded MIMIC latent representations stored for fast retrieval.

| Field | Shape | Location | Description |
|---|---|---|---|
| `window_latents` | (N_windows, d_latent) | GPU | Mean-pooled per non-overlapping window |
| `timestep_latents` | list of (T_i, d_latent) | CPU | Per-stay full-resolution latents |
| `pad_masks` | list of (T_i,) | CPU | Per-stay padding masks |
| `window_to_stay_idx` | (N_windows,) | CPU | Maps window → parent stay |
| `window_to_time_range` | (N_windows, 2) | CPU | (start_t, end_t) per window |

Built by `build_memory_bank()` which encodes all MIMIC training data through the shared encoder. Rebuilt every `memory_refresh_epochs` epochs during training.

### query_memory_bank()

Per-timestep retrieval from the memory bank:

1. **Backward-looking pool**: For timestep t, pool latents from [max(0, t-W+1)..t+1] using cumsum trick (efficient O(T) computation)
2. **Weighted distance**: Optional `importance_weights` (learned, sigmoid-gated) weight dimensions for Euclidean distance
3. **K-NN search**: Find K nearest MIMIC windows per query timestep (chunked cdist for memory safety)
4. **Context gathering**: Fetch full timestep-level latents for each retrieved window

Returns `(B, T, K*W, d_latent)` — per-timestep retrieved context.

### CrossAttentionBlock

Two-stage attention per block:

| Stage | Operation | Description |
|---|---|---|
| 1. Cross-attention | Q=eICU timestep, KV=MIMIC neighbors | Per-timestep fusion with retrieved context |
| 2. Self-attention | Global causal self-attention | Temporal coherence across the full sequence |
| 3. FFN | d_model → d_ff → d_model | Nonlinear transformation |

All stages use LayerNorm + residual connections and respect padding masks.

### Importance Weights

Learned `(d_latent,)` sigmoid-gated weights that control which latent dimensions matter most for nearest-neighbor retrieval. Regularized with `lambda_importance_reg` to prevent collapse (all dimensions equally weighted).

---

## Training Phases

### Phase 1: Autoencoder Pretrain (pretrain_epochs)

Same as SharedLatentTranslator: encode MIMIC → latent → decode, trained with reconstruction + range loss. Optionally adds `lambda_label_pred` for latent label prediction. Uses `decode()` (no retrieval).

### Phase 2: Retrieval-Guided Training

1. **Build/refresh memory bank**: Encode all MIMIC data, rebuild every `memory_refresh_epochs`
2. **Per batch**:
   - Encode eICU source → `src_latent`
   - Query memory bank with `src_latent.detach()` (stop gradient through retrieval)
   - Decode with cross-attention context → translated output
   - Task loss (frozen LSTM) + fidelity + range + smoothness + importance reg
3. **Target task loss** (`lambda_target_task`): Optional MIMIC label supervision

---

## Config Keys

All keys go in the `"training"` section of the config JSON.

| Key | Default | Description |
|---|---|---|
| `k_neighbors` | 16 | Number of nearest MIMIC windows per query timestep |
| `retrieval_window` | 6 | Backward-looking window size for query pooling |
| `n_cross_layers` | 2 | Number of CrossAttentionBlocks |
| `output_mode` | "residual" | "residual" (x + decoded) or "absolute" |
| `memory_refresh_epochs` | 5 | Rebuild memory bank every N epochs |
| `lambda_importance_reg` | 0.01 | L2 regularization on importance weights |
| `lambda_smooth` | 0.1 | Temporal smoothness loss weight |

Translator section uses `"type": "retrieval"`.

---

## Forward Passes

### `forward()` — Without retrieval (pretraining/compatibility)

1. Encode → latent
2. Decode with zero context (cross-attention sees only zeros)
3. Add residual if `output_mode == "residual"`

### `forward_with_retrieval()` — Full retrieval translation

1. Encode → latent
2. Decode with pre-retrieved context via cross-attention
3. Add residual if `output_mode == "residual"`
4. Returns both `x_out` and `latent` (for logging/importance analysis)
