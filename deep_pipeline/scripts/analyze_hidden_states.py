#!/usr/bin/env python3
"""Analyze how translation affects the frozen LSTM's internal representations.

Compares hidden state distributions for:
  1. Original eICU inputs (source domain)
  2. Translated eICU inputs (best translator)
  3. Native MIMIC inputs (target domain)

Computes:
  - Wasserstein-1 distance in hidden state space
  - Multi-kernel MMD in hidden state space
  - Same metrics in input space (for contrast)
  - Per-layer hidden state analysis (multi-layer LSTMs)
  - LSTM gate activation pattern analysis

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/analyze_hidden_states.py --task aki
  CUDA_VISIBLE_DEVICES=3 python scripts/analyze_hidden_states.py --task sepsis
  CUDA_VISIBLE_DEVICES=3 python scripts/analyze_hidden_states.py --task mortality
  CUDA_VISIBLE_DEVICES=3 python scripts/analyze_hidden_states.py --task all
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.yaib import YAIBRuntime
from src.core.hidden_extractor import HiddenStateExtractor
from src.core.mmd import multi_kernel_mmd
from src.core.schema import SchemaResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
#  Task configurations (best models)
# ============================================================================

TASK_CONFIGS = {
    "aki": {
        "config_path": PROJECT_ROOT / "configs" / "aki_v5_cross3.json",
        "translator_type": "retrieval",
        "checkpoint": PROJECT_ROOT / "runs" / "aki_v5_cross3" / "best_translator.pt",
    },
    "sepsis": {
        "config_path": PROJECT_ROOT / "configs" / "sepsis_retr_v4_mmd.json",
        "translator_type": "retrieval",
        "checkpoint": PROJECT_ROOT / "runs" / "sepsis_retr_v4_mmd" / "best_translator.pt",
    },
    "mortality": {
        "config_path": PROJECT_ROOT / "configs" / "mortality_sl_featgate_full.json",
        "translator_type": "shared_latent",
        "checkpoint": PROJECT_ROOT / "runs" / "mortality_sl_featgate_full" / "best_translator.pt",
    },
}


# ============================================================================
#  Utility: collect hidden states from all LSTM layers
# ============================================================================

class MultiLayerHiddenExtractor:
    """Hook into nn.LSTM to capture per-layer hidden states and gate activations."""

    def __init__(self, model: nn.Module):
        self.layer_hidden_states = {}  # layer_idx -> (B, T, H)
        self._hooks = []
        self._lstm_module = self._find_lstm(model)
        self._register_output_hook()

    def _find_lstm(self, model):
        if hasattr(model, "rnn") and isinstance(model.rnn, nn.LSTM):
            return model.rnn
        if hasattr(model, "lstm") and isinstance(model.lstm, nn.LSTM):
            return model.lstm
        raise ValueError("Could not find nn.LSTM in model")

    def _register_output_hook(self):
        """Hook the full LSTM output. For multi-layer, we manually compute per-layer."""
        def hook_fn(module, inp, output):
            # output: (seq_output, (h_n, c_n))
            # h_n: (num_layers, B, H), c_n: (num_layers, B, H)
            seq_out = output[0]  # (B, T, H) — final layer output
            h_n = output[1][0]   # (num_layers, B, H)
            c_n = output[1][1]   # (num_layers, B, H)
            self.layer_hidden_states["final_output"] = seq_out
            self.layer_hidden_states["h_n"] = h_n  # final hidden per layer
            self.layer_hidden_states["c_n"] = c_n  # final cell per layer
        hook = self._lstm_module.register_forward_hook(hook_fn)
        self._hooks.append(hook)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def compute_gate_activations(lstm_module: nn.LSTM, x_input: torch.Tensor,
                              pad_mask: torch.Tensor) -> dict:
    """Manually compute LSTM gate activations for analysis.

    Args:
        lstm_module: The frozen nn.LSTM module
        x_input: (B, T, input_dim) input to the LSTM
        pad_mask: (B, T) True = padded

    Returns:
        Dict with gate activation stats per layer.
    """
    device = x_input.device
    num_layers = lstm_module.num_layers
    hidden_dim = lstm_module.hidden_size
    B, T, _ = x_input.shape

    gate_stats = {}

    current_input = x_input
    for layer in range(num_layers):
        # Get weights for this layer
        W_ih = getattr(lstm_module, f"weight_ih_l{layer}")  # (4*H, input_dim)
        W_hh = getattr(lstm_module, f"weight_hh_l{layer}")  # (4*H, H)
        b_ih = getattr(lstm_module, f"bias_ih_l{layer}")     # (4*H,)
        b_hh = getattr(lstm_module, f"bias_hh_l{layer}")     # (4*H,)

        h = torch.zeros(B, hidden_dim, device=device)
        c = torch.zeros(B, hidden_dim, device=device)

        # Collect gate activations across timesteps
        all_input_gates = []
        all_forget_gates = []
        all_output_gates = []
        all_cell_candidates = []

        for t in range(T):
            x_t = current_input[:, t, :]  # (B, input_dim)

            # gates = W_ih @ x_t + b_ih + W_hh @ h + b_hh
            gates = x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh  # (B, 4*H)

            # PyTorch LSTM gate order: input, forget, cell, output
            i_gate = torch.sigmoid(gates[:, 0*hidden_dim:1*hidden_dim])
            f_gate = torch.sigmoid(gates[:, 1*hidden_dim:2*hidden_dim])
            g_cell = torch.tanh(gates[:, 2*hidden_dim:3*hidden_dim])
            o_gate = torch.sigmoid(gates[:, 3*hidden_dim:4*hidden_dim])

            c = f_gate * c + i_gate * g_cell
            h = o_gate * torch.tanh(c)

            # Only collect for non-padded timesteps
            valid = ~pad_mask[:, t]  # (B,)
            if valid.any():
                all_input_gates.append(i_gate[valid].detach().cpu())
                all_forget_gates.append(f_gate[valid].detach().cpu())
                all_output_gates.append(o_gate[valid].detach().cpu())
                all_cell_candidates.append(g_cell[valid].detach().cpu())

        # Stack and compute stats
        if all_input_gates:
            ig = torch.cat(all_input_gates, dim=0)  # (N_valid, H)
            fg = torch.cat(all_forget_gates, dim=0)
            og = torch.cat(all_output_gates, dim=0)
            gc = torch.cat(all_cell_candidates, dim=0)

            gate_stats[f"layer_{layer}"] = {
                "input_gate_mean": ig.mean().item(),
                "input_gate_std": ig.std().item(),
                "forget_gate_mean": fg.mean().item(),
                "forget_gate_std": fg.std().item(),
                "output_gate_mean": og.mean().item(),
                "output_gate_std": og.std().item(),
                "cell_candidate_mean": gc.mean().item(),
                "cell_candidate_std": gc.std().item(),
                # Distribution shape: fraction of gate units > 0.5 (active)
                "input_gate_active_frac": (ig > 0.5).float().mean().item(),
                "forget_gate_active_frac": (fg > 0.5).float().mean().item(),
                "output_gate_active_frac": (og > 0.5).float().mean().item(),
            }

        # For next layer, use h as input (after applying dropout if present,
        # but we skip dropout since model is in eval mode and we're analyzing)
        # Actually, we need the full sequence hidden states as input to next layer
        # Re-run to get full sequence:
        if layer < num_layers - 1:
            # Collect per-timestep outputs for next layer input
            h2 = torch.zeros(B, hidden_dim, device=device)
            c2 = torch.zeros(B, hidden_dim, device=device)
            next_input = []
            for t in range(T):
                x_t = current_input[:, t, :]
                gates = x_t @ W_ih.T + b_ih + h2 @ W_hh.T + b_hh
                i_g = torch.sigmoid(gates[:, 0*hidden_dim:1*hidden_dim])
                f_g = torch.sigmoid(gates[:, 1*hidden_dim:2*hidden_dim])
                g_c = torch.tanh(gates[:, 2*hidden_dim:3*hidden_dim])
                o_g = torch.sigmoid(gates[:, 3*hidden_dim:4*hidden_dim])
                c2 = f_g * c2 + i_g * g_c
                h2 = o_g * torch.tanh(c2)
                next_input.append(h2.unsqueeze(1))
            current_input = torch.cat(next_input, dim=1)  # (B, T, H)

    return gate_stats


def compute_per_layer_hidden_states(lstm_module: nn.LSTM, x_input: torch.Tensor,
                                     pad_mask: torch.Tensor) -> dict:
    """Compute hidden state output of each LSTM layer.

    Returns:
        Dict mapping layer_idx -> (N_valid_timesteps, hidden_dim) tensor on CPU
    """
    device = x_input.device
    num_layers = lstm_module.num_layers
    hidden_dim = lstm_module.hidden_size
    B, T, _ = x_input.shape

    layer_outputs = {}
    current_input = x_input

    for layer in range(num_layers):
        W_ih = getattr(lstm_module, f"weight_ih_l{layer}")
        W_hh = getattr(lstm_module, f"weight_hh_l{layer}")
        b_ih = getattr(lstm_module, f"bias_ih_l{layer}")
        b_hh = getattr(lstm_module, f"bias_hh_l{layer}")

        h = torch.zeros(B, hidden_dim, device=device)
        c = torch.zeros(B, hidden_dim, device=device)
        all_h = []

        for t in range(T):
            x_t = current_input[:, t, :]
            gates = x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh
            i_g = torch.sigmoid(gates[:, 0*hidden_dim:1*hidden_dim])
            f_g = torch.sigmoid(gates[:, 1*hidden_dim:2*hidden_dim])
            g_c = torch.tanh(gates[:, 2*hidden_dim:3*hidden_dim])
            o_g = torch.sigmoid(gates[:, 3*hidden_dim:4*hidden_dim])
            c = f_g * c + i_g * g_c
            h = o_g * torch.tanh(c)

            valid = ~pad_mask[:, t]
            if valid.any():
                all_h.append(h[valid].detach().cpu())

        if all_h:
            layer_outputs[layer] = torch.cat(all_h, dim=0)

        # Build next layer input
        if layer < num_layers - 1:
            h2 = torch.zeros(B, hidden_dim, device=device)
            c2 = torch.zeros(B, hidden_dim, device=device)
            seq_out = []
            for t in range(T):
                x_t = current_input[:, t, :]
                gates = x_t @ W_ih.T + b_ih + h2 @ W_hh.T + b_hh
                i_g = torch.sigmoid(gates[:, 0*hidden_dim:1*hidden_dim])
                f_g = torch.sigmoid(gates[:, 1*hidden_dim:2*hidden_dim])
                g_c = torch.tanh(gates[:, 2*hidden_dim:3*hidden_dim])
                o_g = torch.sigmoid(gates[:, 3*hidden_dim:4*hidden_dim])
                c2 = f_g * c2 + i_g * g_c
                h2 = o_g * torch.tanh(c2)
                seq_out.append(h2.unsqueeze(1))
            current_input = torch.cat(seq_out, dim=1)

    return layer_outputs


# ============================================================================
#  Distance metrics
# ============================================================================

def compute_wasserstein_1d(x: np.ndarray, y: np.ndarray, n_projections: int = 100,
                           seed: int = 42) -> float:
    """Sliced Wasserstein distance: average 1D Wasserstein over random projections."""
    rng = np.random.RandomState(seed)
    d = x.shape[1]
    total = 0.0
    for _ in range(n_projections):
        direction = rng.randn(d)
        direction /= np.linalg.norm(direction)
        proj_x = x @ direction
        proj_y = y @ direction
        total += wasserstein_distance(proj_x, proj_y)
    return total / n_projections


def compute_mmd(x: torch.Tensor, y: torch.Tensor, max_samples: int = 4096) -> float:
    """Multi-kernel MMD between two sample sets."""
    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0])[:max_samples]
        x = x[idx]
    if y.shape[0] > max_samples:
        idx = torch.randperm(y.shape[0])[:max_samples]
        y = y[idx]
    # Use the codebase MMD function (detaches y internally)
    mmd_val = multi_kernel_mmd(x, y)
    return mmd_val.item()


def compute_cosine_similarity(x: np.ndarray, y: np.ndarray, max_samples: int = 4096) -> float:
    """Mean cosine similarity between centroids of two distributions."""
    centroid_x = x.mean(axis=0)
    centroid_y = y.mean(axis=0)
    dot = np.dot(centroid_x, centroid_y)
    norm = np.linalg.norm(centroid_x) * np.linalg.norm(centroid_y)
    return float(dot / max(norm, 1e-8))


# ============================================================================
#  Data collection
# ============================================================================

def collect_hidden_states_and_inputs(
    yaib_runtime: YAIBRuntime,
    data_loader,
    translator=None,
    schema_resolver=None,
    renorm_scale=None,
    renorm_offset=None,
    device="cuda",
    max_timesteps=50000,
    static_dim=4,
):
    """Pass data through frozen LSTM and collect hidden states + raw inputs.

    If translator is provided, data is translated first.

    Returns:
        dict with:
            "hidden_states": (N, H) numpy array — final LSTM layer output
            "inputs": (N, F_input) numpy array — YAIB-format inputs to LSTM
            "pad_masks": list of (T,) boolean masks per batch
            "raw_inputs": list of raw input tensors for gate analysis
    """
    yaib_runtime.load_baseline_model()
    model = yaib_runtime._model.to(device)
    model.train()  # Required for cuDNN LSTM backward (even though we don't backprop)

    extractor = HiddenStateExtractor(model)

    all_hidden = []
    all_inputs = []
    all_raw_batches = []  # for gate analysis
    all_pad_masks = []
    total_timesteps = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(b.to(device) for b in batch)

            if translator is not None and schema_resolver is not None:
                parts = schema_resolver.extract(batch)
                x_val = parts["X_val"]
                x_static = parts["X_static"]

                # Handle missing static features: inject zeros with correct dim
                if x_static.shape[-1] == 0 and static_dim > 0:
                    x_static = torch.zeros(x_val.shape[0], static_dim,
                                           device=device, dtype=x_val.dtype)

                # Apply renorm if present
                if renorm_scale is not None:
                    rs = renorm_scale.to(device)
                    ro = renorm_offset.to(device)
                    x_val = x_val * rs.view(1, 1, -1) + ro.view(1, 1, -1)
                    x_val = x_val.masked_fill(parts["M_pad"].unsqueeze(-1).bool(), 0.0)

                # Translate
                x_val_out = translator(
                    x_val, parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], x_static,
                )

                # Rebuild YAIB format
                x_yaib_translated = schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"],
                    parts["X_static"], m_pad=parts["M_pad"],
                )

                lstm_input = x_yaib_translated
                pad_mask = parts["M_pad"]
            else:
                lstm_input = batch[0]
                # Infer pad mask: all-zero rows
                pad_mask = (lstm_input.abs().sum(dim=-1) == 0)

            # Forward through LSTM
            _ = yaib_runtime.forward((lstm_input, batch[1], batch[2]))
            hidden = extractor.hidden_states  # (B, T, H)

            if hidden is None:
                continue

            # Mask and collect
            valid = ~pad_mask.bool()  # (B, T)
            for b in range(hidden.shape[0]):
                valid_t = valid[b]  # (T,)
                if valid_t.any():
                    h_valid = hidden[b][valid_t].detach().cpu()  # (n_valid, H)
                    inp_valid = lstm_input[b][valid_t].detach().cpu()  # (n_valid, F)
                    all_hidden.append(h_valid)
                    all_inputs.append(inp_valid)
                    total_timesteps += h_valid.shape[0]

            # Store for gate analysis (keep first few batches)
            if len(all_raw_batches) < 20:
                all_raw_batches.append((lstm_input.detach(), pad_mask.detach()))
            all_pad_masks.append(pad_mask.detach().cpu())

            if total_timesteps >= max_timesteps:
                break

    extractor.remove()

    hidden_np = torch.cat(all_hidden, dim=0).numpy() if all_hidden else np.zeros((0, 1))
    inputs_np = torch.cat(all_inputs, dim=0).numpy() if all_inputs else np.zeros((0, 1))

    logger.info("  Collected %d valid timesteps (hidden dim=%d, input dim=%d)",
                hidden_np.shape[0],
                hidden_np.shape[1] if hidden_np.ndim > 1 else 0,
                inputs_np.shape[1] if inputs_np.ndim > 1 else 0)

    return {
        "hidden_states": hidden_np,
        "inputs": inputs_np,
        "raw_batches": all_raw_batches,
    }


# ============================================================================
#  Translator loading
# ============================================================================

def load_translator(task_info: dict, config: dict, schema_resolver: SchemaResolver,
                    device: str):
    """Load the best translator for a task and return (translator, renorm_scale, renorm_offset)."""
    translator_type = task_info["translator_type"]
    checkpoint_path = task_info["checkpoint"]
    translator_cfg = config.get("translator", {})
    training_cfg = config.get("training", {})
    static_features = config["vars"].get("STATIC", [])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if translator_type == "retrieval":
        from src.core.retrieval_translator import RetrievalTranslator, build_memory_bank
        from src.core.eval import RetrievalTranslatorWrapper

        translator = RetrievalTranslator(
            num_features=len(schema_resolver.indices.dynamic),
            d_latent=translator_cfg.get("d_latent", 128),
            d_model=translator_cfg.get("d_model", 128),
            d_time=translator_cfg.get("d_time", 16),
            n_enc_layers=translator_cfg.get("n_enc_layers", 4),
            n_dec_layers=translator_cfg.get("n_dec_layers", 2),
            n_cross_layers=training_cfg.get("n_cross_layers", 2),
            n_heads=translator_cfg.get("n_heads", 8),
            d_ff=translator_cfg.get("d_ff", 512),
            dropout=translator_cfg.get("dropout", 0.2),
            out_dropout=translator_cfg.get("out_dropout", 0.1),
            static_dim=len(static_features),
            temporal_attention_mode=translator_cfg.get("temporal_attention_mode", "causal"),
            temporal_attention_window=translator_cfg.get("temporal_attention_window", 0),
            output_mode=training_cfg.get("output_mode", "absolute"),
        )

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        translator.load_state_dict(ckpt["translator_state_dict"], strict=False)
        renorm_scale = ckpt.get("renorm_scale")
        renorm_offset = ckpt.get("renorm_offset")

        translator.to(device)
        translator.eval()

        # Build memory bank from MIMIC target data
        target_runtime = YAIBRuntime(
            data_dir=Path(config["target_data_dir"]),
            baseline_model_dir=Path(config["baseline_model_dir"]),
            task_config=Path(config["task_config"]),
            model_config=Path(config["model_config"]) if config.get("model_config") else None,
            model_name=config["model_name"],
            vars=copy.deepcopy(config["vars"]),
            file_names=copy.deepcopy(config["file_names"]),
            seed=training_cfg.get("seed", config.get("seed", 2222)),
            batch_size=training_cfg.get("batch_size", 16),
        )
        target_runtime.load_data()
        target_train_loader = target_runtime.create_dataloader("train", shuffle=False, ram_cache=True)

        logger.info("  Building memory bank for retrieval...")
        memory_bank = build_memory_bank(
            encoder=translator,
            target_loader=target_train_loader,
            schema_resolver=schema_resolver,
            device=device,
            window_size=training_cfg.get("retrieval_window", 6),
            window_stride=training_cfg.get("window_stride", None),
        )

        wrapped = RetrievalTranslatorWrapper(
            translator=translator,
            memory_bank=memory_bank,
            k_neighbors=training_cfg.get("k_neighbors", 16),
            retrieval_window=training_cfg.get("retrieval_window", 6),
        )
        return wrapped, renorm_scale, renorm_offset

    elif translator_type == "shared_latent":
        from src.core.latent_translator import SharedLatentTranslator

        translator = SharedLatentTranslator(
            num_features=len(schema_resolver.indices.dynamic),
            d_latent=translator_cfg.get("d_latent", 128),
            d_model=translator_cfg.get("d_model", 128),
            d_time=translator_cfg.get("d_time", 16),
            n_enc_layers=translator_cfg.get("n_enc_layers", 4),
            n_dec_layers=translator_cfg.get("n_dec_layers", 3),
            n_heads=translator_cfg.get("n_heads", 8),
            d_ff=translator_cfg.get("d_ff", 512),
            dropout=translator_cfg.get("dropout", 0.2),
            out_dropout=translator_cfg.get("out_dropout", 0.1),
            static_dim=len(static_features),
            temporal_attention_mode=translator_cfg.get("temporal_attention_mode", "bidirectional"),
            temporal_attention_window=translator_cfg.get("temporal_attention_window", 0),
        )

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        translator.load_state_dict(ckpt["translator_state_dict"], strict=False)
        renorm_scale = ckpt.get("renorm_scale")
        renorm_offset = ckpt.get("renorm_offset")

        translator.to(device)
        translator.eval()
        return translator, renorm_scale, renorm_offset

    else:
        raise ValueError(f"Unknown translator type: {translator_type}")


# ============================================================================
#  Main analysis per task
# ============================================================================

def analyze_task(task_name: str, device: str = "cuda", max_timesteps: int = 50000):
    """Run full hidden state analysis for one task."""
    logger.info("=" * 70)
    logger.info("ANALYZING TASK: %s", task_name.upper())
    logger.info("=" * 70)

    task_info = TASK_CONFIGS[task_name]
    config = json.loads(task_info["config_path"].read_text())
    training_cfg = config.get("training", {})
    static_features = config["vars"].get("STATIC", [])

    # ── 1. Build source (eICU) runtime ──
    logger.info("[1/6] Loading eICU (source) data...")
    src_runtime = YAIBRuntime(
        data_dir=Path(config["data_dir"]),
        baseline_model_dir=Path(config["baseline_model_dir"]),
        task_config=Path(config["task_config"]),
        model_config=Path(config["model_config"]) if config.get("model_config") else None,
        model_name=config["model_name"],
        vars=copy.deepcopy(config["vars"]),
        file_names=copy.deepcopy(config["file_names"]),
        seed=training_cfg.get("seed", config.get("seed", 2222)),
        batch_size=training_cfg.get("batch_size", 16),
    )
    src_runtime.load_data()
    src_runtime.load_baseline_model()
    src_test_loader = src_runtime.create_dataloader("test", shuffle=False, ram_cache=True)

    # ── 2. Build target (MIMIC) runtime ──
    logger.info("[2/6] Loading MIMIC (target) data...")
    tgt_runtime = YAIBRuntime(
        data_dir=Path(config["target_data_dir"]),
        baseline_model_dir=Path(config["baseline_model_dir"]),
        task_config=Path(config["task_config"]),
        model_config=Path(config["model_config"]) if config.get("model_config") else None,
        model_name=config["model_name"],
        vars=copy.deepcopy(config["vars"]),
        file_names=copy.deepcopy(config["file_names"]),
        seed=training_cfg.get("seed", config.get("seed", 2222)),
        batch_size=training_cfg.get("batch_size", 16),
    )
    tgt_runtime.load_data()
    tgt_test_loader = tgt_runtime.create_dataloader("test", shuffle=False, ram_cache=True)

    # ── 3. Build SchemaResolver and load translator ──
    logger.info("[3/6] Loading translator...")
    feature_names = src_test_loader.dataset.get_feature_names()
    group_col = config["vars"].get("GROUP")
    schema_resolver = SchemaResolver(
        feature_names=feature_names,
        dynamic_features=config["vars"]["DYNAMIC"],
        static_features=static_features,
        allow_missing_static=True,
        group_col=group_col,
    )

    translator, renorm_scale, renorm_offset = load_translator(
        task_info, config, schema_resolver, device
    )

    n_static = len(static_features)

    # ── 4. Collect hidden states for all 3 streams ──
    logger.info("[4/6] Collecting hidden states: eICU original...")
    eicu_data = collect_hidden_states_and_inputs(
        src_runtime, src_test_loader, translator=None, device=device,
        max_timesteps=max_timesteps, static_dim=n_static,
    )

    logger.info("[4/6] Collecting hidden states: eICU translated...")
    trans_data = collect_hidden_states_and_inputs(
        src_runtime, src_test_loader, translator=translator,
        schema_resolver=schema_resolver,
        renorm_scale=renorm_scale, renorm_offset=renorm_offset,
        device=device, max_timesteps=max_timesteps, static_dim=n_static,
    )

    logger.info("[4/6] Collecting hidden states: MIMIC native...")
    mimic_data = collect_hidden_states_and_inputs(
        tgt_runtime, tgt_test_loader, translator=None, device=device,
        max_timesteps=max_timesteps, static_dim=n_static,
    )

    # ── 5. Compute distance metrics ──
    logger.info("[5/6] Computing distance metrics...")

    results = {"task": task_name}

    # Hidden state space
    h_eicu = eicu_data["hidden_states"]
    h_trans = trans_data["hidden_states"]
    h_mimic = mimic_data["hidden_states"]

    logger.info("  Hidden state shapes: eICU=%s, translated=%s, MIMIC=%s",
                h_eicu.shape, h_trans.shape, h_mimic.shape)

    # Wasserstein (sliced)
    w_eicu_mimic = compute_wasserstein_1d(h_eicu, h_mimic)
    w_trans_mimic = compute_wasserstein_1d(h_trans, h_mimic)
    w_reduction = 1.0 - (w_trans_mimic / max(w_eicu_mimic, 1e-8))

    results["hidden_wasserstein_eicu_vs_mimic"] = w_eicu_mimic
    results["hidden_wasserstein_trans_vs_mimic"] = w_trans_mimic
    results["hidden_wasserstein_reduction"] = w_reduction

    logger.info("  [HIDDEN] Wasserstein: eICU->MIMIC=%.6f, translated->MIMIC=%.6f, reduction=%.1f%%",
                w_eicu_mimic, w_trans_mimic, w_reduction * 100)

    # MMD
    h_eicu_t = torch.from_numpy(h_eicu).float()
    h_trans_t = torch.from_numpy(h_trans).float()
    h_mimic_t = torch.from_numpy(h_mimic).float()

    mmd_eicu_mimic = compute_mmd(h_eicu_t, h_mimic_t)
    mmd_trans_mimic = compute_mmd(h_trans_t, h_mimic_t)
    mmd_reduction = 1.0 - (mmd_trans_mimic / max(mmd_eicu_mimic, 1e-8))

    results["hidden_mmd_eicu_vs_mimic"] = mmd_eicu_mimic
    results["hidden_mmd_trans_vs_mimic"] = mmd_trans_mimic
    results["hidden_mmd_reduction"] = mmd_reduction

    logger.info("  [HIDDEN] MMD: eICU->MIMIC=%.6f, translated->MIMIC=%.6f, reduction=%.1f%%",
                mmd_eicu_mimic, mmd_trans_mimic, mmd_reduction * 100)

    # Cosine similarity of centroids
    cos_eicu_mimic = compute_cosine_similarity(h_eicu, h_mimic)
    cos_trans_mimic = compute_cosine_similarity(h_trans, h_mimic)

    results["hidden_cosine_eicu_vs_mimic"] = cos_eicu_mimic
    results["hidden_cosine_trans_vs_mimic"] = cos_trans_mimic

    logger.info("  [HIDDEN] Cosine(centroid): eICU-MIMIC=%.6f, translated-MIMIC=%.6f",
                cos_eicu_mimic, cos_trans_mimic)

    # Input space (for contrast)
    inp_eicu = eicu_data["inputs"]
    inp_trans = trans_data["inputs"]
    inp_mimic = mimic_data["inputs"]

    # Truncate to same feature dim for fair comparison
    min_dim = min(inp_eicu.shape[1], inp_mimic.shape[1])
    inp_eicu_c = inp_eicu[:, :min_dim]
    inp_trans_c = inp_trans[:, :min_dim]
    inp_mimic_c = inp_mimic[:, :min_dim]

    w_inp_eicu = compute_wasserstein_1d(inp_eicu_c, inp_mimic_c)
    w_inp_trans = compute_wasserstein_1d(inp_trans_c, inp_mimic_c)
    w_inp_change = (w_inp_trans - w_inp_eicu) / max(w_inp_eicu, 1e-8)

    results["input_wasserstein_eicu_vs_mimic"] = w_inp_eicu
    results["input_wasserstein_trans_vs_mimic"] = w_inp_trans
    results["input_wasserstein_change"] = w_inp_change

    logger.info("  [INPUT] Wasserstein: eICU->MIMIC=%.6f, translated->MIMIC=%.6f, change=%+.1f%%",
                w_inp_eicu, w_inp_trans, w_inp_change * 100)

    inp_eicu_t = torch.from_numpy(inp_eicu_c).float()
    inp_trans_t = torch.from_numpy(inp_trans_c).float()
    inp_mimic_t = torch.from_numpy(inp_mimic_c).float()

    mmd_inp_eicu = compute_mmd(inp_eicu_t, inp_mimic_t)
    mmd_inp_trans = compute_mmd(inp_trans_t, inp_mimic_t)
    mmd_inp_change = (mmd_inp_trans - mmd_inp_eicu) / max(mmd_inp_eicu, 1e-8)

    results["input_mmd_eicu_vs_mimic"] = mmd_inp_eicu
    results["input_mmd_trans_vs_mimic"] = mmd_inp_trans
    results["input_mmd_change"] = mmd_inp_change

    logger.info("  [INPUT] MMD: eICU->MIMIC=%.6f, translated->MIMIC=%.6f, change=%+.1f%%",
                mmd_inp_eicu, mmd_inp_trans, mmd_inp_change * 100)

    # ── 6. Gate activation analysis ──
    logger.info("[6/6] Computing gate activation analysis...")

    lstm_module = src_runtime._model.rnn
    gate_results = {}

    for stream_name, raw_batches in [
        ("eicu_original", eicu_data["raw_batches"]),
        ("eicu_translated", trans_data["raw_batches"]),
        ("mimic_native", mimic_data["raw_batches"]),
    ]:
        if not raw_batches:
            continue
        # Use first few batches for gate analysis (memory-efficient)
        all_gate_stats = {}
        n_batches = min(10, len(raw_batches))
        for lstm_input, pad_mask in raw_batches[:n_batches]:
            batch_stats = compute_gate_activations(lstm_module, lstm_input, pad_mask)
            for layer_key, stats in batch_stats.items():
                if layer_key not in all_gate_stats:
                    all_gate_stats[layer_key] = {k: [] for k in stats}
                for k, v in stats.items():
                    all_gate_stats[layer_key][k].append(v)

        # Average across batches
        gate_results[stream_name] = {}
        for layer_key, stat_lists in all_gate_stats.items():
            gate_results[stream_name][layer_key] = {
                k: float(np.mean(v)) for k, v in stat_lists.items()
            }

    results["gate_activations"] = gate_results

    # Log gate comparison
    for layer_key in sorted(gate_results.get("eicu_original", {}).keys()):
        eicu_fg = gate_results.get("eicu_original", {}).get(layer_key, {}).get("forget_gate_mean", 0)
        trans_fg = gate_results.get("eicu_translated", {}).get(layer_key, {}).get("forget_gate_mean", 0)
        mimic_fg = gate_results.get("mimic_native", {}).get(layer_key, {}).get("forget_gate_mean", 0)
        logger.info("  [GATES %s] Forget gate mean: eICU=%.4f, translated=%.4f, MIMIC=%.4f",
                    layer_key, eicu_fg, trans_fg, mimic_fg)

        eicu_ig = gate_results.get("eicu_original", {}).get(layer_key, {}).get("input_gate_mean", 0)
        trans_ig = gate_results.get("eicu_translated", {}).get(layer_key, {}).get("input_gate_mean", 0)
        mimic_ig = gate_results.get("mimic_native", {}).get(layer_key, {}).get("input_gate_mean", 0)
        logger.info("  [GATES %s] Input gate mean:  eICU=%.4f, translated=%.4f, MIMIC=%.4f",
                    layer_key, eicu_ig, trans_ig, mimic_ig)

        eicu_og = gate_results.get("eicu_original", {}).get(layer_key, {}).get("output_gate_mean", 0)
        trans_og = gate_results.get("eicu_translated", {}).get(layer_key, {}).get("output_gate_mean", 0)
        mimic_og = gate_results.get("mimic_native", {}).get(layer_key, {}).get("output_gate_mean", 0)
        logger.info("  [GATES %s] Output gate mean: eICU=%.4f, translated=%.4f, MIMIC=%.4f",
                    layer_key, eicu_og, trans_og, mimic_og)

    # ── Per-layer hidden state analysis (multi-layer LSTMs) ──
    num_layers = lstm_module.num_layers
    if num_layers > 1:
        logger.info("  [LAYERS] Multi-layer analysis (%d layers)...", num_layers)
        layer_results = {}

        for stream_name, raw_batches in [
            ("eicu_original", eicu_data["raw_batches"]),
            ("eicu_translated", trans_data["raw_batches"]),
            ("mimic_native", mimic_data["raw_batches"]),
        ]:
            all_layer_hidden = {}
            n_batches = min(10, len(raw_batches))
            for lstm_input, pad_mask in raw_batches[:n_batches]:
                per_layer = compute_per_layer_hidden_states(lstm_module, lstm_input, pad_mask)
                for layer_idx, hidden in per_layer.items():
                    if layer_idx not in all_layer_hidden:
                        all_layer_hidden[layer_idx] = []
                    all_layer_hidden[layer_idx].append(hidden)

            layer_results[stream_name] = {}
            for layer_idx in sorted(all_layer_hidden.keys()):
                cat = torch.cat(all_layer_hidden[layer_idx], dim=0).numpy()
                layer_results[stream_name][layer_idx] = cat

        # Compute per-layer Wasserstein and MMD
        results["per_layer_analysis"] = {}
        for layer_idx in range(num_layers):
            h_e = layer_results.get("eicu_original", {}).get(layer_idx)
            h_t = layer_results.get("eicu_translated", {}).get(layer_idx)
            h_m = layer_results.get("mimic_native", {}).get(layer_idx)

            if h_e is not None and h_t is not None and h_m is not None:
                w_e = compute_wasserstein_1d(h_e, h_m)
                w_t = compute_wasserstein_1d(h_t, h_m)
                red = 1.0 - (w_t / max(w_e, 1e-8))

                mmd_e = compute_mmd(torch.from_numpy(h_e).float(), torch.from_numpy(h_m).float())
                mmd_t = compute_mmd(torch.from_numpy(h_t).float(), torch.from_numpy(h_m).float())
                mmd_red = 1.0 - (mmd_t / max(mmd_e, 1e-8))

                results["per_layer_analysis"][f"layer_{layer_idx}"] = {
                    "wasserstein_eicu": w_e,
                    "wasserstein_trans": w_t,
                    "wasserstein_reduction": red,
                    "mmd_eicu": mmd_e,
                    "mmd_trans": mmd_t,
                    "mmd_reduction": mmd_red,
                }
                logger.info("    Layer %d: W_eicu=%.6f, W_trans=%.6f (red=%.1f%%), MMD_eicu=%.6f, MMD_trans=%.6f (red=%.1f%%)",
                            layer_idx, w_e, w_t, red * 100, mmd_e, mmd_t, mmd_red * 100)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY FOR %s", task_name.upper())
    logger.info("=" * 70)
    logger.info("  INPUT SPACE:  Wasserstein changed by %+.1f%%, MMD changed by %+.1f%%",
                results["input_wasserstein_change"] * 100,
                results["input_mmd_change"] * 100)
    logger.info("  HIDDEN SPACE: Wasserstein reduced by %.1f%%, MMD reduced by %.1f%%",
                results["hidden_wasserstein_reduction"] * 100,
                results["hidden_mmd_reduction"] * 100)

    if results["hidden_wasserstein_reduction"] > 0 and results.get("input_wasserstein_change", 0) > 0:
        logger.info("  >>> SMOKING GUN: Inputs DIVERGE (%+.1f%%) but hidden states CONVERGE (%.1f%% closer)!",
                    results["input_wasserstein_change"] * 100,
                    results["hidden_wasserstein_reduction"] * 100)
    elif results["hidden_wasserstein_reduction"] > 0:
        logger.info("  >>> Hidden states converge (%.1f%% closer to MIMIC)",
                    results["hidden_wasserstein_reduction"] * 100)

    # Save results
    output_dir = PROJECT_ROOT / "runs" / f"{task_name}_hidden_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hidden_state_analysis.json"

    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info("  Results saved to %s", output_path)

    return results


# ============================================================================
#  Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze LSTM hidden state alignment after translation")
    parser.add_argument("--task", choices=["aki", "sepsis", "mortality", "all"], default="aki",
                        help="Which task to analyze")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--max-timesteps", type=int, default=50000,
                        help="Max timesteps to collect per stream")
    args = parser.parse_args()

    if args.task == "all":
        tasks = ["aki", "sepsis", "mortality"]
    else:
        tasks = [args.task]

    all_results = {}
    for task in tasks:
        try:
            results = analyze_task(task, device=args.device, max_timesteps=args.max_timesteps)
            all_results[task] = results
        except Exception as e:
            logger.error("Failed on task %s: %s", task, e, exc_info=True)

    # Final cross-task summary
    if len(all_results) > 1:
        logger.info("")
        logger.info("=" * 70)
        logger.info("CROSS-TASK SUMMARY")
        logger.info("=" * 70)
        logger.info("%-12s | %-20s | %-20s | %-20s | %-20s",
                    "Task", "Input W change", "Hidden W reduction", "Input MMD change", "Hidden MMD reduction")
        logger.info("-" * 100)
        for task, r in all_results.items():
            logger.info("%-12s | %+18.1f%% | %18.1f%% | %+18.1f%% | %18.1f%%",
                        task,
                        r.get("input_wasserstein_change", 0) * 100,
                        r.get("hidden_wasserstein_reduction", 0) * 100,
                        r.get("input_mmd_change", 0) * 100,
                        r.get("hidden_mmd_reduction", 0) * 100)


if __name__ == "__main__":
    main()
