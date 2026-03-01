# Shared Latent Space Translation — Plan

## Core Idea

Instead of learning small deltas on raw features, learn a **shared latent space** where both eICU (source) and MIMIC (target) samples are mapped to nearby representations. Translation is then derived from the sample's position in this shared space through a decoder that reconstructs target-domain features.

```
eICU features ─→ [Shared Encoder] ─→ Latent Space ─→ [Decoder] ─→ MIMIC-like features ─→ Frozen LSTM
MIMIC features ─→ [Shared Encoder] ─→ Latent Space (aligned)
```

## Why This Might Work

1. **Bottleneck forces abstraction**: The latent space forces the model to learn domain-invariant clinical representations, not surface-level feature mappings
2. **Decoder has a strong prior**: Pretraining on MIMIC teaches the decoder what valid MIMIC features look like — it can only produce plausible outputs
3. **Alignment in learned space**: MMD/adversarial alignment in the latent space is more effective than raw feature MMD (the learned representation surfaces task-relevant structure)
4. **Translation is implicit**: Once eICU maps to the same latent region as similar MIMIC patients, the decoder automatically produces the right MIMIC-like features

## Architecture: SharedLatentTranslator

### Encoder (shared for both domains)
- Reuses the existing EHRTranslator's embedding pipeline: triplet projection (value + missingness + time_delta) → sensor embeddings → lift to d_model → sinusoidal time encoding → AxialBlocks (variable-wise + temporal attention)
- FiLM modulation from static features
- After N_enc AxialBlocks, project from (B, T, F, d_model) → (B, T, d_latent) via mean-pool over features + linear projection
- The latent is a per-timestep representation: (B, T, d_latent)

### Latent Bottleneck
- Dimension: d_latent (e.g., 64 or 128)
- Per-timestep: each timestep has its own latent vector
- This is where alignment happens (MMD between source and target latent representations)

### Decoder
- Input: latent (B, T, d_latent) → project to (B, T, F, d_model) by broadcasting + per-feature embeddings
- N_dec AxialBlocks (variable-wise + temporal attention)
- FiLM modulation from static features (separate from encoder)
- Output head: linear (d_model → 1) per feature → (B, T, F)
- Output is the FULL translated features (not deltas), masked at padding

### Key Design Decisions
- **No residual/delta connection**: The decoder produces complete output features. This forces the latent space to carry all the information needed for reconstruction.
- **Shared encoder weights**: The SAME encoder processes both eICU and MIMIC. This is critical — shared weights force domain-invariant representations.
- **Separate decoder**: Only needed for eICU→MIMIC direction. During pretraining, it reconstructs MIMIC features. During training, it generates translated eICU features.

## Training Pipeline

### Phase 1: Autoencoder Pretraining on MIMIC (target domain)
- Encoder: MIMIC features → latent
- Decoder: latent → reconstructed MIMIC features
- Loss: MSE reconstruction on non-padded, non-missing timesteps
- Purpose: Teach the encoder what clinical data looks like, teach the decoder to produce valid MIMIC features
- Epochs: 10-20, learning rate: 1e-4

### Phase 2: Joint Training with Alignment
- Source (eICU) path: encoder → latent → decoder → translated features → frozen LSTM → task loss
- Target (MIMIC) path: encoder → latent (for alignment only)
- Losses:
  - **Task loss**: Frozen LSTM classification on translated eICU features
  - **Alignment loss (MMD)**: multi_kernel_mmd(source_latent, target_latent) in the bottleneck space
  - **Reconstruction loss**: MSE(decoder(encoder(MIMIC)), MIMIC) — keeps decoder calibrated
  - **Range loss**: Penalize outputs outside feature bounds
- Loss weighting: task + λ_align * alignment + λ_recon * reconstruction + λ_range * range

## Hyperparameter Configurations

### Config 1: Base
- d_latent=64, n_enc_layers=3, n_dec_layers=2
- λ_align=0.5, λ_recon=0.1, λ_range=0.5
- Pretrain: 10 epochs
- Train: 30 epochs, lr=1e-4

### Config 2: Stronger alignment + no pretrain
- d_latent=64, n_enc_layers=3, n_dec_layers=2
- λ_align=1.0, λ_recon=0.2, λ_range=0.5
- Pretrain: 0 epochs (to measure pretrain value)
- Train: 30 epochs, lr=1e-4

### Config 3: Larger latent + more capacity
- d_latent=128, n_enc_layers=4, n_dec_layers=3
- λ_align=0.5, λ_recon=0.1, λ_range=0.5
- Pretrain: 15 epochs
- Train: 30 epochs, lr=1e-4

## Implementation Steps

1. Create `src/core/latent_translator.py` — SharedLatentTranslator class
2. Add `LatentTranslatorTrainer` to `src/core/train.py` — pretraining + joint training
3. Wire up in `src/cli.py` — support `"type": "shared_latent"` in config
4. Create 6 configs (3 hyperparams × 2 tasks: mortality + sepsis)
5. Run experiments on mortality full data (where we know translation works)
6. If time permits, run best config on sepsis
7. Write results to `docs/shared_latent_results.md`

## Files to Modify/Create

- **NEW**: `src/core/latent_translator.py` — SharedLatentTranslator model
- **MODIFY**: `src/core/train.py` — Add LatentTranslatorTrainer
- **MODIFY**: `src/cli.py` — Support shared_latent translator type
- **NEW**: `experiments/configs/shared_latent_*.json` — Experiment configs
- **NEW**: `docs/shared_latent_results.md` — Results writeup
