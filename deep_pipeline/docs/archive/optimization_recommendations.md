# Optimization Recommendations for Sepsis Pipeline

## Problem
Sepsis sequences are padded to 169 timesteps but ~73% is padding (median actual length ~45).
The shared latent translator uses ~2.5x more memory than the original (4 forward passes per batch).

## 1. Minimizing Padding Waste

### A. Bucket Batching (highest impact, minimal code change) ✅ IMPLEMENTED
Sort sequences by actual length, batch similarly-lengthed sequences together, pad only to per-batch max instead of global max (169). Most batches pad to 40-60 timesteps. Preserves exact model behavior. Attention memory drops from O(169²) to O(L_batch²).

### B. Truncation to Percentile Cap (simple, small info loss)
Cap sequences at e.g. the 95th or 99th percentile of actual length. Few long sequences lose tail timesteps. Max padding drops dramatically.

### C. Pack Multiple Short Sequences (advanced, most efficient)
Concatenate multiple short sequences into one packed tensor of length 169 with segment masks. Eliminates padding entirely but requires careful masking in translator attention and LSTM forward. More invasive.

### D. Padding-Aware FlashAttention ✅ IMPLEMENTED
PyTorch 2.x `scaled_dot_product_attention` with proper masking enables FlashAttention to physically skip padding computation. Use NestedTensor or `torch.nn.attention.sdpa_kernel` for variable-length mode.

## 2. Runtime/Memory Optimization

### A. Gradient Checkpointing on AxialBlock layers
Recompute activations during backward instead of storing. ~30% more compute, ~40-50% less memory. Could enable batch_size=64.

### B. Freeze Encoder After Pretraining
Freeze encoder during Phase 2. Decoder still adapts. Saves ~40% backward pass memory/compute.

### C. Temporal Compression in Latent Space
Add temporal pooling in encoder to reduce latent temporal dimension (169 → ~42). Cuts attention cost by 16x. Design change.

### D. Full Float16 LSTM Forward
Ensure frozen LSTM forward is fully float16 during training. Saves ~30% on LSTM activations.

### E. Gradient Accumulation
batch_size=16 with accumulate_steps=2 for same effective batch=32, half peak memory.

## Priority Order
1. Bucket Batching (A1) — best effort:impact, zero info loss
2. Padding-Aware FlashAttention (A4) — synergizes with bucket batching
3. Gradient Checkpointing (B1) — memory savings
4. Freeze Encoder (B2) — memory + speed
5. Gradient Accumulation (B5) — simple fallback
