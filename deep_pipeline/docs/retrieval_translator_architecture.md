# RetrievalTranslator Architecture

> **Role**: Model architecture reference for the retrieval-guided translator.
> **See also**: [architecture.md](architecture.md) (delta translator), [shared_latent_results.md](shared_latent_results.md) (shared latent results), [gradient_bottleneck_analysis.md](gradient_bottleneck_analysis.md) (gradient analysis)

**Date**: Feb 25, 2026 (updated Apr 14, 2026 — added CCR, self-retrieval, gradient checkpointing, full API)
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
| `all_latents_flat` | (sum(T_i), d_latent) | CPU | All timestep latents flattened for vectorized gather |
| `window_flat_start` | (N_windows,) | CPU | Start index per window into `all_latents_flat` |
| `window_actual_len` | (N_windows,) | CPU | Actual (non-padded) length per window |
| `window_labels` | (N_windows,) or None | CPU | Mean label per window (for CCR, optional) |
| `window_label_masks` | (N_windows,) or None | CPU | Whether window has valid labels (for CCR) |

Built by `build_memory_bank()` which encodes all MIMIC training data through the shared encoder. Rebuilt every `memory_refresh_epochs` epochs during training. When `store_labels=True`, also stores per-window mean labels for class-conditional retrieval.

### query_memory_bank()

Per-timestep retrieval from the memory bank:

1. **Backward-looking pool**: For timestep t, pool latents from [max(0, t-W+1)..t+1] using cumsum trick (efficient O(T) computation)
2. **Weighted distance**: Optional `importance_weights` (learned, sigmoid-gated) weight dimensions for Euclidean distance
3. **K-NN search**: Find K nearest MIMIC windows per query timestep (chunked cdist for memory safety)
4. **Context gathering**: Fetch full timestep-level latents for each retrieved window

Returns `(B, T, K*W, d_latent)` — per-timestep retrieved context.

### Class-Conditional Retrieval (CCR)

When `ccr_alpha > 0`, retrieval distances are scaled by label agreement between query and bank windows. For each query timestep with predicted label probability `q` and each bank window with mean label `l`:

```
same_class = q * l + (1 - q) * (1 - l)    # soft class agreement [0, 1]
scale = 1 / (1 + ccr_alpha * same_class)   # lower distance for same-class neighbors
dists = dists * scale
```

This biases retrieval toward same-class neighbors without hard thresholds. Windows without valid labels (`window_label_masks == False`) are unscaled. CCR requires `store_labels=True` during `build_memory_bank()` and `query_label_probs` from the translator's label prediction head.

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

Same as SharedLatentTranslator: encode MIMIC → latent → decode, trained with reconstruction + range loss. Optionally adds `lambda_label_pred` for latent label prediction via `predict_labels()`.

#### Self-Retrieval Pretraining (`phase1_self_retrieval: true`)

Without self-retrieval, Phase 1 passes zero tensors to cross-attention blocks, which learn a degenerate pass-through. With `phase1_self_retrieval`, Phase 1 builds a MIMIC memory bank from the encoder's own representations and provides real context to cross-attention blocks during pretraining:

1. Memory bank is built from the encoder's current representations of MIMIC training data
2. Each MIMIC batch retrieves from this bank (excluding self), providing meaningful cross-attention input
3. Bank is refreshed every `phase1_memory_refresh_epochs` epochs (default: same as `memory_refresh_epochs`)
4. At end of Phase 1, the bank is discarded so Phase 2 rebuilds fresh from the final encoder

**Important**: Self-retrieval pretrain checkpoints are **incompatible** with non-self-retrieval ones (different cross-attention weight initialization). The fingerprint system in `manage_pretrain.py` tracks this automatically.

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

| Key | Default | Section | Description |
|---|---|---|---|
| `k_neighbors` | 16 | training | Number of nearest MIMIC windows per query timestep |
| `retrieval_window` | 6 | training | Backward-looking window size for query pooling |
| `n_cross_layers` | 2 | training | Number of CrossAttentionBlocks. Task-dependent: 3 for AKI/LoS/KF (dense labels), 2 for mortality/sepsis (sparse) |
| `output_mode` | "residual" | training | `"residual"` adds decoded delta to input; `"absolute"` outputs directly |
| `memory_refresh_epochs` | 5 | training | Rebuild memory bank every N epochs during Phase 2 |
| `lambda_importance_reg` | 0.01 | training | L2 regularization on importance weights (prevents dimension collapse) |
| `lambda_smooth` | 0.1 | training | Temporal smoothness loss weight |
| `gradient_checkpointing` | false | training | Trades compute for VRAM: checkpoints encoder, decoder, and cross-attention blocks. Enables `batch_size=32` on V100-32GB |
| `phase1_self_retrieval` | false | training | Build MIMIC self-retrieval bank during Phase 1 pretrain (see above) |
| `phase1_memory_refresh_epochs` | `memory_refresh_epochs` | training | Bank refresh interval during self-retrieval Phase 1 |
| `ccr_alpha` | 0.0 | training | Class-conditional retrieval scaling (0 = disabled). See CCR section above |

Translator section uses `"type": "retrieval"`.

---

## Public API

### Core forward passes

| Method | Used in | Description |
|---|---|---|
| `forward()` | Phase 1 pretrain | Encode → decode with zero context (no retrieval). Returns `x_out` |
| `forward_with_retrieval()` | Phase 2 training + eval | Encode → decode with retrieved context via cross-attention. Returns `(x_out, latent)` |

Both add residual (`x_val + decoded`) when `output_mode == "residual"`.

### Building blocks

| Method | Description |
|---|---|
| `encode(x_val, x_miss, t_abs, m_pad, x_static)` | Shared encoder → `(B, T, d_latent)`. Called standalone during memory bank building |
| `decode_with_context(latent, context, x_val, m_pad, x_static)` | Cross-attention + decoder with retrieved context |
| `decode(latent, x_val, m_pad, x_static)` | Decoder with zero context (Phase 1 path) |

### Auxiliary methods

| Method | Description |
|---|---|
| `predict_labels(latent, m_pad)` | Label prediction head for Phase 1 pretraining (`lambda_label_pred`). Returns `(B, T)` logits |
| `get_importance_weights()` | Returns `(d_latent,)` sigmoid-gated weights in [0,1] for retrieval distance weighting |
| `set_temporal_mode(mode)` | Switch all attention blocks between `"causal"` and `"bidirectional"`. Called per-task at startup |

### Gradient checkpointing

When `gradient_checkpointing=True`, all AxialBlock (encoder/decoder) and CrossAttentionBlock forward passes use `torch.utils.checkpoint.checkpoint()` to trade compute for VRAM. This halves peak memory, enabling `batch_size=32` on V100-32GB. Applied to encoder blocks, cross-attention blocks, and decoder blocks independently.
