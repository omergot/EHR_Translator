# Delta Translator (EHRTranslator) Architecture

> **Role**: Model architecture reference for the delta-based translator. Describes the transformer translator's components, data flow, and attention modes.
> **See also**: [retrieval_translator_architecture.md](retrieval_translator_architecture.md) (retrieval-guided translator), [shared_latent_results.md](shared_latent_results.md) (shared latent results), [gradient_bottleneck_analysis.md](gradient_bottleneck_analysis.md) (gradient analysis)
> **Historical**: [archive/gradient_flow_mechanics.md](archive/gradient_flow_mechanics.md) (gradient flow details)

**Date**: Feb 14, 2026
**File**: `src/core/translator.py` (class `EHRTranslator`)

---

## Overview

The EHRTranslator is a transformer-based model that learns to produce per-feature **deltas** added to the input clinical features. It uses axial attention (separate variable-wise and temporal attention), FiLM conditioning from static features, and a residual output design.

---

## Architecture Diagram

```
Input: x_val (B,T,48), x_miss (B,T,48), t_abs (B,T), m_pad (B,T), x_static (B,4)
                                    │
                    ┌───────────────┴───────────────┐
                    │         EMBEDDING STAGE        │
                    │                                │
                    │  triplet_proj: [val,miss,Δt]   │──→ (B,T,F,d_latent)
                    │  + sensor_emb (per-feature)    │
                    │  lift: d_latent → d_model      │──→ (B,T,F,d_model)
                    │  + time_proj(sin/cos(t_abs))   │
                    └───────────────┬───────────────┘
                                    │  h = (B, T, 48, d_model)
                    ┌───────────────┴───────────────┐
                    │     BACKBONE: 4x AxialBlock    │
                    │                                │
                    │  var_attn:  (B*T, 48, d)       │  features attend to each other
                    │  temp_attn: (B*48, T, d)       │  timesteps attend to each other
                    │  ffn: d_model → d_ff → d_model │
                    │  + FiLM: γ*h + β               │  (from x_static via film_mlp)
                    └───────────────┬───────────────┘
                                    │  h = (B, T, 48, d_model)
                    ┌───────┬───────┴───────┬────────────┐
                    │       │               │            │
              delta_head  forecast_head  reconstruction_head  (MLM only, discarded)
             Linear(d,1)  Linear(d,1)    Linear(d,1)
                    │       │               │
              (B,T,48)  (B,T,48)        (B,T,48)
                    │
              x_out = x_val + delta    ← residual connection
```

---

## Component Details

### Embedding Stage

Converts raw clinical data into a rich representation:

| Component | Shape | Description |
|---|---|---|
| `triplet_proj` | Linear(3, d_latent) | Combines a feature's value, its missingness flag, and the time gap since last measurement into a single vector |
| `sensor_emb` | Parameter(num_features, d_latent) | Learned identity for each of the 48 features (so the model knows "this is heart rate" vs "this is glucose") |
| `lift` | Linear(d_latent, d_model) | Projects from d_latent (16) to d_model (64 or 128) |
| `time_proj` | Linear(d_time, d_model) | Projects sinusoidal absolute time encoding into model dimension |

### Backbone (4x AxialBlock)

The core reasoning engine. Each block has:

| Sub-component | Tensor shape | Description |
|---|---|---|
| `var_attn` | (B\*T, 48, d_model) | At each timestep, features attend to other features. Learns cross-feature correlations ("heart rate is high AND lactate is high → this means something") |
| `temp_attn` | (B\*48, T, d_model) | For each feature, timesteps attend to other timesteps. Learns temporal dynamics ("glucose was rising for the last 5 hours"). Can be **causal** (see only past) or **bidirectional** |
| `ffn` | d_model → d_ff → d_model | Nonlinear transformation of each representation |
| FiLM conditioning | per-layer (γ, β) | Static features (age, sex, height, weight) → `film_mlp` → per-layer scale and shift. Applies `h = γ*h + β` so the model can condition on demographics ("this is an elderly patient, transform differently") |

### Heads

Three separate `Linear(d_model, 1)` projections that read out different things from the same backbone representation `h`:

| Head | Output | Purpose | When used |
|---|---|---|---|
| `delta_head` | (B, T, 48) | Per-feature deltas for translation. Output = `x_val + delta` | Always (main output) |
| `forecast_head` | (B, T, 48) | Predicts next-timestep values | Training only (auxiliary loss, `lambda_forecast > 0`) |
| `reconstruction_head` | (B, T, 48) | Reconstructs masked timestep values | MLM pretraining only (discarded after) |

---

## MLM Pretraining: What Each Part Learns

The MLM (Masked Language Modeling) task randomly masks 15% of timesteps, zeros out their values, and trains the model to reconstruct them. The loss is MSE between reconstructed and original values at masked positions.

### Learning breakdown

| Component | What it learns from MLM |
|---|---|
| `triplet_proj` | How to encode the value+missingness+time_delta triplet into a useful vector. For masked positions (value=0, miss unchanged), it learns that "zero value with this missingness pattern" means "this is masked, look elsewhere" |
| `sensor_emb` | Feature identities — the model needs to know which feature it's reconstructing. Different features have different normal ranges and dynamics |
| `var_attn` | Cross-feature correlations: "if I can't see glucose at time t, but I can see lactate and insulin, I can infer glucose." Learns the clinical covariance structure |
| `temp_attn` | Temporal dynamics: "glucose at t-1 was 120, glucose at t-2 was 115, so glucose at t is probably ~125." Learns trends, periodicity, and temporal patterns |
| `ffn` | Nonlinear combinations of the attention outputs |
| `film_mlp` | How patient demographics affect feature patterns ("elderly patients have different normal ranges") |
| `lift`, `time_proj` | Projection weights that create good representations for the attention layers |

### What gets discarded after pretraining

| Component | Why it's discarded |
|---|---|
| `mask_embedding` | A special d_model vector added at masked positions to signal "reconstruct me." Not needed for translation |
| `reconstruction_head` | Maps backbone output → raw feature values. The downstream `delta_head` maps backbone output → delta corrections, which is a different readout task |

### Key Insight

The backbone learns **general EHR understanding** — feature correlations, temporal dynamics, patient conditioning — all of which are useful for translation. The `reconstruction_head` is just a thin readout layer specific to the "fill in the blank" task. The `delta_head` is a different thin readout layer specific to the "how should I modify this data" task. Both rely on the same backbone representations.

This is exactly analogous to **BERT**: pretrain a transformer on masked language modeling (learning language understanding), discard the MLM head, then add a task-specific classification head on top.

---

## Temporal Attention Modes

The `set_temporal_mode()` method flips `block.use_causal_temporal_attention` for all AxialBlocks:

| Mode | Behavior | Use case |
|---|---|---|
| `"causal"` | Representation at timestep t only sees timesteps 0..t | Required for per-timestep tasks (Sepsis, AKI) to prevent time travel |
| `"bidirectional"` | All timesteps can attend to all others | Allowed for per-stay tasks (Mortality) |

During MLM pretraining, the model temporarily switches to **bidirectional** mode (the reconstruction task requires seeing surrounding context), then switches back to **causal** for downstream fine-tuning.

---

## Forward Passes

### `forward()` — Translation (main task)

1. Build triplet `[x_val, x_miss, time_delta]` → `triplet_proj` → add `sensor_emb` → `lift` → add `time_proj`
2. Pass through 4 AxialBlocks with FiLM conditioning
3. `delta_head(h)` → squeeze → `x_out = x_val + delta`
4. Mask padded positions to zero
5. Optionally compute `forecast_head(h)` for auxiliary loss

### `forward_mlm()` — MLM Pretraining

1. Same embedding stage, but replace masked position embeddings with `mask_embedding`
2. Pass through backbone (bidirectional mode)
3. `reconstruction_head(h)` → squeeze → reconstructed values at all positions
4. Loss computed only at masked positions
