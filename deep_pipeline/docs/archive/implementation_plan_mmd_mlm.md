# Implementation Plan: MMD Domain Matching + MLM Pretraining

> **Role**: Historical — implementation blueprint for MMD+MLM stages. Code is implemented; experiments showed limited benefit (+0.001-0.002).
> **See also**: [strategy_evaluation_mmd_mlm.md](strategy_evaluation_mmd_mlm.md) (why this approach was chosen), [experiment_results_mmd_mlm.md](experiment_results_mmd_mlm.md) (results)

## Overview

Add two complementary techniques to improve the causal translator:
1. **Multi-Kernel MMD loss** with real MIMIC data — gives the translator direct gradient signal about target domain distributions
2. **MLM pretraining** — pretrain the translator backbone with bidirectional attention on a self-supervised masked reconstruction task

Both techniques are clean, defensible, and complementary. MMD addresses the weak gradient problem; MLM gives better temporal representations.

## Ground Rules

1. **Backward compatibility:** Every new feature is controlled via config. If a config key is absent or set to 0/null, behavior is identical to the current codebase. No existing run should break.
2. **Debug mode:** When `"debug": true`, ALL data loading (including MIMIC target) uses `subset_fraction=0.2`. This applies to every dataloader created.
3. **Experiment protocol:** All experiments during development run on **Sepsis** task in **debug mode** (`"debug": true`). Full-data runs happen only after all stages are validated.
4. **Checkpoint experiments:** After each stage from Stage 2 onward, run a debug experiment and log results in the `Experiment Log` section at the bottom of this document.

---

## Stage 0: Multi-Kernel MMD Utility

**Goal:** Implement a standalone, tested MK-MMD loss function.

### 0.1 Create `src/core/mmd.py`

```python
def multi_kernel_mmd(x, y, bandwidths=None) -> torch.Tensor
```

- **Input:** `x` (N, D) source features, `y` (M, D) target features (both flattened across valid timesteps)
- **Kernel:** Sum of RBF kernels: `K(a,b) = Σ_i exp(-||a-b||² / (2 * σ_i²))`
- **Bandwidth selection:** If `bandwidths=None`, use median heuristic on combined samples, then multiply by `[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]`
- **Computation:** Unbiased MMD² estimator: `mean(K_xx) + mean(K_yy) - 2*mean(K_xy)` (exclude diagonal for unbiased version)
- **Subsampling:** If N or M > `max_samples` (e.g., 4096), randomly subsample to keep memory manageable. The `(B, T, D)` tensor flattened across valid timesteps can be huge.
- **Return:** Scalar tensor with gradients

### 0.2 Test

Add `tests/test_mmd.py`:
- Same distribution → MMD ≈ 0
- Different distributions (shifted mean) → MMD > 0
- Gradient flows back through `x`

### Files to create
- `src/core/mmd.py`
- `tests/test_mmd.py`

---

## Stage 1: MIMIC Data Loading Infrastructure

**Goal:** Load MIMIC data in parallel with eICU during transformer translator training.

### 1.1 Config additions

Add `target_data_dir` to the config JSON (same level as `data_dir`):

```json
{
  "data_dir": "/bigdata/omerg/Thesis/cohort_data/sepsis/eicu",
  "target_data_dir": "/bigdata/omerg/Thesis/cohort_data/sepsis/miiv",
  ...
  "training": {
    "lambda_mmd": 1.0,
    ...
  }
}
```

MIMIC data exists at:
- Sepsis: `/bigdata/omerg/Thesis/cohort_data/sepsis/miiv/`
- AKI: `/bigdata/omerg/Thesis/cohort_data/aki/miiv/`
- Has identical schema: `dyn.parquet`, `outc.parquet`, `sta.parquet` with same 48 dynamic features

### 1.2 Modify `src/cli.py` — `train_translator()` function

After creating the eICU runtime and dataloaders (lines 373-394), add:

```python
# Load MIMIC target data if target_data_dir is specified
target_train_loader = None
if config.get("target_data_dir"):
    target_runtime = _build_runtime_from_config(
        config,
        data_dir_override=config["target_data_dir"],
        batch_size_override=training_cfg["batch_size"],
        seed_override=training_cfg["seed"],
    )
    target_runtime.load_data()
    target_train_loader = target_runtime.create_dataloader(
        'train',
        shuffle=True,   # important: shuffle for MMD sampling diversity
        ram_cache=True,
        subset_fraction=0.2 if debug_mode else None,  # debug mode → 20% subset
        subset_seed=training_cfg["seed"],
    )
    # Augment with static features (same handling as eICU)
    # ... same static augmentation logic ...
```

**Note on normalization:** Each YAIBRuntime normalizes data using its own training split statistics. eICU data is eICU-normalized, MIMIC data is MIMIC-normalized. The translator maps eICU-normalized → MIMIC-normalized space. This is exactly what we want for MMD matching.

### 1.3 Pass target loader to trainer

Modify `TransformerTranslatorTrainer.__init__()` to accept optional `target_loader`.

### Files to modify
- `src/cli.py` — load MIMIC data, pass to trainer
- `configs/sample_transformer_config.json` — add `target_data_dir`, `lambda_mmd`

---

## Stage 2: MMD Loss in Training Loop

**Goal:** Add MMD loss between translated eICU features and real MIMIC features.

### 2.1 Modify `TransformerTranslatorTrainer.__init__()`

Add parameters:
- `target_train_loader: DataLoader | None = None`
- `lambda_mmd: float = 0.0`

Store `self.target_train_loader`, `self.lambda_mmd`. Create a cycling iterator for the target loader:
```python
if target_train_loader:
    self._target_iter = iter(target_train_loader)
```

### 2.2 Add target batch sampling helper

```python
def _next_target_batch(self) -> tuple:
    try:
        batch = next(self._target_iter)
    except StopIteration:
        self._target_iter = iter(self.target_train_loader)
        batch = next(self._target_iter)
    return tuple(b.to(self.device) for b in batch)
```

### 2.3 Modify `_run_epoch()` — add MMD computation

After computing `x_val_out` (translated eICU features), and before loss summation:

```python
l_mmd = x_val_out.new_tensor(0.0)
if self.lambda_mmd > 0 and self.target_train_loader is not None:
    # Get MIMIC batch
    target_batch = self._next_target_batch()
    target_parts = self.schema_resolver.extract(target_batch)

    # Flatten valid timesteps to (N, D) for source and (M, D) for target
    source_mask = ~parts["M_pad"]          # (B, T)
    target_mask = ~target_parts["M_pad"]   # (B', T')

    source_features = x_val_out[source_mask]           # (N, 48)
    target_features = target_parts["X_val"][target_mask]  # (M, 48)

    l_mmd = multi_kernel_mmd(source_features, target_features)
```

**Important:** `target_parts["X_val"]` is MIMIC-normalized (no gradients needed — it's the target distribution).

Update total loss:
```python
l_total = l_task + (self.lambda_fidelity * l_fidelity) + (self.lambda_range * l_range) + (self.lambda_forecast * l_forecast) + (self.lambda_mmd * l_mmd)
```

### 2.4 Modify `_validate()` — add MMD computation

Same pattern as training, but within `torch.no_grad()`. Need a separate target validation iterator (or reuse target train loader for validation MMD).

### 2.5 Logging updates

Add `"mmd"` to the `totals` dict in both `_run_epoch` and `_validate`. Update all logging format strings and `self.history` to include MMD loss.

### 2.6 Plot updates

Add MMD loss to `_plot_losses()` alongside task/fidelity/range curves.

### 2.7 Config / CLI wiring

In `_get_training_config()` add (default 0.0 = disabled, backward compatible):
```python
"lambda_mmd": training.get("lambda_mmd", 0.0),
```

In `train_translator()`, pass to trainer:
```python
trainer = TransformerTranslatorTrainer(
    ...,
    target_train_loader=target_train_loader,
    lambda_mmd=training_cfg["lambda_mmd"],
)
```

### Files to modify
- `src/core/train.py` — `TransformerTranslatorTrainer` class
- `src/cli.py` — `_get_training_config()`, `train_translator()`

### First experiment config

```json
{
  "data_dir": "/bigdata/omerg/Thesis/cohort_data/sepsis/eicu",
  "target_data_dir": "/bigdata/omerg/Thesis/cohort_data/sepsis/miiv",
  "translator": {
    "type": "transformer",
    "d_model": 64,
    "temporal_attention_mode": "causal"
  },
  "training": {
    "lr": 1e-4,
    "epochs": 30,
    "batch_size": 64,
    "lambda_fidelity": 0.1,
    "lambda_mmd": 1.0,
    "lambda_range": 0.001,
    "early_stopping_patience": 5,
    "best_metric": "val_task"
  }
}
```

**Lambda_mmd tuning:** Start with 1.0, try [0.1, 0.5, 1.0, 5.0, 10.0]. Monitor: if l_mmd drops but l_task doesn't improve, lambda is too high (translator matches distributions but loses task signal). If l_mmd doesn't drop, lambda is too low.

---

## Stage 3: Transition MMD (Level 2)

**Goal:** Also match temporal dynamics — the distribution of feature changes between consecutive timesteps.

### 3.1 Add transition MMD option

In the MMD computation block:

```python
if self.lambda_mmd_transition > 0 and self.target_train_loader is not None:
    # Compute consecutive differences
    source_delta = x_val_out[:, 1:, :] - x_val_out[:, :-1, :]
    target_delta = target_parts["X_val"][:, 1:, :] - target_parts["X_val"][:, :-1, :]

    # Mask: both current and next timestep must be valid
    source_trans_mask = source_mask[:, :-1] & source_mask[:, 1:]
    target_trans_mask = target_mask[:, :-1] & target_mask[:, 1:]

    source_trans = source_delta[source_trans_mask]
    target_trans = target_delta[target_trans_mask]

    l_mmd_transition = multi_kernel_mmd(source_trans, target_trans)
```

### 3.2 Config addition (default 0.0 = disabled, backward compatible)

```json
"training": {
    "lambda_mmd": 1.0,
    "lambda_mmd_transition": 0.5,
    ...
}
```

### Files to modify
- `src/core/train.py` — add transition MMD alongside marginal MMD
- `src/cli.py` — `_get_training_config()`

---

## Stage 4: MLM Pretraining

**Goal:** Pretrain the EHRTranslator backbone on masked reconstruction using bidirectional attention, then fine-tune causally with task loss + MMD.

### 4.1 Data for pretraining

**Use source (eICU) training data only.**

Rationale:
- The translator will be applied to eICU data, so it should learn eICU temporal patterns
- MIMIC data has different normalization — mixing would confuse the model
- Simpler: reuse existing eICU dataloader
- The MMD loss (during fine-tuning) provides the cross-domain signal

No data split needed — MLM pretraining is self-supervised (no labels), so using the same training data for pretraining and fine-tuning is standard practice (analogous to BERT pretraining on the same corpus used for fine-tuning).

### 4.2 Create `src/core/pretrain.py`

```python
class MLMPretrainer:
    def __init__(
        self,
        translator: EHRTranslator,
        schema_resolver: SchemaResolver,
        mask_prob: float = 0.15,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ):
        ...
```

**Key design decisions:**

1. **What to mask:** Mask entire timesteps (not individual features). Set masked timestep values to 0 and add a learned `[MASK]` token embedding. Mask probability: 15% of non-padded timesteps.

2. **Masking strategy (following BERT):**
   - 80% of selected timesteps → replace with mask token (zeros + mask embedding)
   - 10% → replace with random values from another timestep in the same batch
   - 10% → keep original (forces model to represent even "known" timesteps well)

3. **Attention mode during pretraining:** **Bidirectional.** This is legitimate because:
   - No labels are used — purely self-supervised
   - The model learns temporal representations, not clinical predictions
   - Standard practice (BERT, MAE, etc.)

4. **Reconstruction target:** Predict the original `X_val` at masked timesteps.

5. **Loss:** MSE between predicted and original feature values at masked positions only.

6. **What to train:** All translator parameters EXCEPT `delta_head` (the output head for translation deltas). Instead, add a temporary `reconstruction_head` (Linear(d_model, 1)) for MLM. After pretraining, discard `reconstruction_head` and keep the pretrained backbone.

### 4.3 Switching attention mode

The EHRTranslator's `AxialBlock` already supports `use_causal_temporal_attention` as a constructor flag. For MLM pretraining we need bidirectional mode.

**Approach:** Create the translator with `temporal_attention_mode="bidirectional"` for pretraining, then create a NEW translator with `temporal_attention_mode="causal"` for fine-tuning and **load the pretrained weights** (excluding attention-mode-specific parameters — but since the only difference is the mask applied at runtime, all weights transfer directly).

Actually, looking at the code: `use_causal_temporal_attention` only controls whether an attention MASK is applied at runtime. The weight matrices are identical. So we can:
1. Create translator with `temporal_attention_mode="bidirectional"` (no mask)
2. Pretrain
3. Change `block.use_causal_temporal_attention = True` for each block
4. Fine-tune — same weights, different mask

Even simpler: add a `set_temporal_mode(mode)` method to EHRTranslator:
```python
def set_temporal_mode(self, mode: str):
    causal = (mode == "causal")
    for block in self.blocks:
        block.use_causal_temporal_attention = causal
```

### 4.4 MLMPretrainer training loop

```python
def train_epoch(self, loader: DataLoader) -> float:
    self.translator.train()
    total_loss = 0.0
    for batch in loader:
        parts = schema_resolver.extract(batch)
        x_val = parts["X_val"]       # (B, T, D)
        m_pad = parts["M_pad"]       # (B, T)

        # Generate mask: 15% of non-padded timesteps
        valid = ~m_pad               # (B, T)
        mask_prob_tensor = torch.rand_like(valid.float()) < self.mask_prob
        mlm_mask = valid & mask_prob_tensor  # (B, T) — True = masked

        # Apply masking to input
        x_masked = x_val.clone()
        x_masked[mlm_mask] = 0.0     # Zero out masked timesteps

        # Forward (bidirectional, returns reconstructed values)
        x_reconstructed = self.translator.forward_mlm(
            x_masked, parts["X_miss"], parts["t_abs"], m_pad,
            parts["X_static"], mlm_mask
        )

        # Loss: MSE at masked positions only
        loss = F.mse_loss(
            x_reconstructed[mlm_mask],  # (N_masked, D)
            x_val[mlm_mask]             # (N_masked, D)
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(loader)
```

### 4.5 Add `forward_mlm()` to EHRTranslator

This is similar to the regular `forward()` but:
- Uses a `reconstruction_head` instead of `delta_head`
- Outputs absolute values (not input + delta), since the input is masked
- Optionally adds a mask embedding to masked positions

```python
def forward_mlm(self, x_val, x_miss, t_abs, m_pad, x_static, mlm_mask):
    # Same embedding pipeline as forward()
    # ...
    # Add mask embedding at masked positions
    h[mlm_mask] = h[mlm_mask] + self.mask_embedding  # learned (d_model,)
    # Run through axial blocks (bidirectional)
    # ...
    # Reconstruct
    x_reconstructed = self.reconstruction_head(h).squeeze(-1)  # (B, T, D)
    return x_reconstructed
```

### 4.6 CLI orchestration

In `train_translator()`, before the main training loop:

```python
if translator_cfg.get("mlm_pretrain_epochs", 0) > 0:
    logging.info("Starting MLM pretraining (%d epochs, bidirectional)", mlm_epochs)

    # Set bidirectional mode for pretraining
    translator.set_temporal_mode("bidirectional")

    pretrainer = MLMPretrainer(
        translator=translator,
        schema_resolver=schema_resolver,
        mask_prob=translator_cfg.get("mlm_mask_prob", 0.15),
        learning_rate=translator_cfg.get("mlm_lr", 1e-4),
        device=device,
    )
    pretrainer.train(
        epochs=mlm_pretrain_epochs,
        train_loader=train_loader,
    )

    # Switch back to causal for translator fine-tuning
    translator.set_temporal_mode("causal")

    # Discard reconstruction head, reinitialize delta head
    translator.reconstruction_head = None  # free memory
    translator.delta_head.reset_parameters()  # fresh output head

    logging.info("MLM pretraining completed. Switching to causal mode for fine-tuning.")

# Continue with normal TransformerTranslatorTrainer...
```

### 4.7 Config additions (all default to 0/disabled, backward compatible)

```json
{
  "translator": {
    "type": "transformer",
    "d_model": 64,
    "temporal_attention_mode": "causal",
    "mlm_pretrain_epochs": 10,
    "mlm_mask_prob": 0.15,
    "mlm_lr": 1e-4
  }
}
```

When `mlm_pretrain_epochs` is 0 or absent, no pretraining happens — goes straight to translator training (current behavior).

### Files to create/modify
- **Create:** `src/core/pretrain.py` — `MLMPretrainer` class
- **Modify:** `src/core/translator.py` — add `set_temporal_mode()`, `forward_mlm()`, `mask_embedding`, `reconstruction_head`
- **Modify:** `src/cli.py` — add MLM pretraining phase before fine-tuning
- **Create:** `tests/test_pretrain.py` — verify masking, loss computation, mode switching

---

## Stage 5: Combined Experiments (full-data runs after debug validation)

### 5.1 Experiment matrix

Run each on Sepsis with d_model=64, causal, 30 epochs:

| Experiment | MMD | Trans. MMD | MLM Pretrain | Config |
|------------|-----|-----------|--------------|--------|
| A: Baseline (current) | - | - | - | `exp_baseline.json` |
| B: MMD only | lambda=1.0 | - | - | `exp_mmd.json` |
| C: MMD + Transition | lambda=1.0 | lambda=0.5 | - | `exp_mmd_trans.json` |
| D: MLM only | - | - | 10 epochs | `exp_mlm.json` |
| E: MLM + MMD | lambda=1.0 | - | 10 epochs | `exp_mlm_mmd.json` |
| F: Full | lambda=1.0 | lambda=0.5 | 10 epochs | `exp_full.json` |

### 5.2 Hyperparameter sweep (for best variant)

- `lambda_mmd`: [0.1, 0.5, 1.0, 5.0, 10.0]
- `lambda_mmd_transition`: [0, 0.1, 0.5, 1.0]
- `mlm_pretrain_epochs`: [5, 10, 20]
- `mlm_mask_prob`: [0.1, 0.15, 0.2]

### 5.3 Evaluation

For each experiment:
1. Train with `train_translator`
2. Evaluate with `translate_and_eval` → get AUCROC, AUCPR, loss
3. Compare val_task learning curves (does MMD unblock learning?)
4. Compare test metrics against baseline and cheaty model

---

## Implementation Order

```
Stage 0 (MMD utility)           — implement + unit tests
  └→ Stage 1 (MIMIC loading)    — implement + verify loading works
       └→ Stage 2 (MMD in loop) — implement
            └→ 🧪 Experiment B (MMD only, debug mode, Sepsis) — log results below
            └→ Stage 3 (Transition MMD)
                 └→ 🧪 Experiment C (MMD + Transition, debug mode, Sepsis) — log results below
Stage 4 (MLM pretraining)
  └→ 🧪 Experiment D (MLM only, debug mode, Sepsis) — log results below
  └→ 🧪 Experiment E (MLM + MMD, debug mode, Sepsis) — log results below
Stage 5 (Combined + full-data runs after debug validation)
```

**First milestone:** After Stage 2, run Experiment B in debug mode. If val_task drops meaningfully (below 0.65, compared to current 0.675 floor), the approach is working and worth continuing. If val_task barely moves, revisit lambda_mmd tuning before proceeding.

**All debug experiments** use: Sepsis task, `"debug": true` (20% data), d_model=64, causal mode, `train_and_eval` subcommand so we get AUCROC/AUCPR numbers.

---

## Key Risks and Mitigations

### Risk: MMD dominates and translator ignores task loss
**Mitigation:** Start with lambda_mmd=1.0 (same order as task loss ~0.7). If MMD loss drops but task loss doesn't improve, reduce lambda_mmd. The task loss MUST be the primary objective.

### Risk: MIMIC dataloader has different sequence lengths, causing MMD artifacts
**Mitigation:** We flatten to (N, D) across valid timesteps before computing MMD. Sequence length doesn't matter — we're comparing feature distributions, not aligned sequences.

### Risk: MMD memory explosion for large batches
**Mitigation:** Subsample to max 4096 timesteps per domain in the MMD computation. Multi-kernel MMD with 7 bandwidths on (4096, 48) is ~600MB — manageable on V100 32GB.

### Risk: MLM pretraining doesn't transfer to causal mode
**Mitigation:** The weights are identical — only the runtime mask changes. Worst case, pretrained weights provide a better starting point that degrades slightly under causal constraint. This is still better than random init.

### Risk: MIMIC and eICU SchemaResolver incompatibility
**Mitigation:** Both datasets have identical 48 dynamic features and 4 static features (confirmed from data). Use the SAME SchemaResolver instance for both — it only cares about feature names, not data content. Verify with an assertion that feature names match.

---

## Config Backward Compatibility Summary

All new config keys and their defaults (absent = same as current behavior):

| Config key | Location | Default | Effect when absent/default |
|---|---|---|---|
| `target_data_dir` | top-level | `null` | No MIMIC loading, no MMD |
| `training.lambda_mmd` | training | `0.0` | MMD loss disabled |
| `training.lambda_mmd_transition` | training | `0.0` | Transition MMD disabled |
| `translator.mlm_pretrain_epochs` | translator | `0` | No MLM pretraining |
| `translator.mlm_mask_prob` | translator | `0.15` | (only used if mlm_pretrain_epochs > 0) |
| `translator.mlm_lr` | translator | `1e-4` | (only used if mlm_pretrain_epochs > 0) |

---

## Experiment Log

All experiment results (configs, tables, per-experiment details, analysis, and next steps) are maintained in **[docs/experiment_results_mmd_mlm.md](experiment_results_mmd_mlm.md)**.
