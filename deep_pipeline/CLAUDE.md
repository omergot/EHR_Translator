# CLAUDE.md

You are the best data scientist and deep learning engineer in the world who is eager to succeed in our mission to find a translator that improves the metrics.

## Project Overview

**EHR Translator Deep Pipeline** — domain adaptation for EHR time-series (eICU → MIMIC-IV). Train a `Translator` to transform source-domain data so a **strictly frozen** target-domain LSTM baseline performs well. Three paradigms: delta-based (`EHRTranslator`), shared latent (`SharedLatentTranslator`), and retrieval-guided (`RetrievalTranslator`). Five tasks: Mortality24 (per-stay, binary), AKI (per-timestep, binary), Sepsis (per-timestep, binary), LoS (per-timestep, regression), KidneyFunction (per-stay, regression).

## Commands

```bash
pip install -e .

# Delta-based translator (EHRTranslator)
python run.py train_translator --config configs/<task>_transformer_config.json
python run.py translate_and_eval --config configs/<task>_transformer_config.json --output_parquet out.parquet
python run.py train_and_eval --config configs/<task>_transformer_config.json --output_parquet out.parquet

# Shared latent translator (SharedLatentTranslator) — same CLI, different config
python run.py train_translator --config experiments/configs/sl_v3_mortality.json
python run.py translate_and_eval --config experiments/configs/sl_v3_mortality.json --output_parquet out.parquet

# Retrieval translator (RetrievalTranslator) — requires target_data_dir in config
python run.py train_translator --config configs/sepsis_retrieval_full.json
python run.py translate_and_eval --config configs/sepsis_retrieval_full.json --output_parquet out.parquet

# Tests
pytest tests/

# Screening system (fast iteration)
python scripts/manage_pretrain.py --list                             # Index all pretrain checkpoints
python scripts/manage_pretrain.py --auto-copy configs/new.json       # Auto-copy matching pretrain
python scripts/screen_experiment.py --config configs/new.json        # Screen (submit + wait + evaluate)
python scripts/screen_experiment.py --config configs/new.json --submit-only  # Submit without waiting
python scripts/calibrate_screening.py --task aki --paradigm retrieval --epochs 5 --submit  # Calibrate
python scripts/compare_results.py --task aki --include-screening     # Leaderboard
python scripts/autoresearch.py --task aki --paradigm retrieval --budget 12h  # Autonomous search
```

## Architecture (Key Points)

- **Entry point**: `run.py` → `src/cli.py` (main orchestrator). Subcommands: `train_translator`, `translate_and_eval`, `train_and_eval`.
- **Data flow**: JSON Config → `YAIBRuntime` (baseline + data) → Translator → frozen LSTM → loss → checkpoint best → eval → parquet.
- **Delta-based** (`src/core/translator.py`): Outputs deltas added to input (starts near identity). `set_temporal_mode()` flips causal/bidirectional.
- **Shared latent** (`src/core/latent_translator.py`): Encoder→latent z→Decoder. Outputs absolute values, not deltas.
- **Retrieval-guided** (`src/core/retrieval_translator.py`): Shared encoder → `MemoryBank` (pre-encoded MIMIC windows) → k-NN per timestep → `CrossAttentionBlock` → Decoder. Instance-level matching, naturally causal.
- **Feature gate** (`src/core/feature_gate.py`): Learnable per-feature sigmoid weights for loss weighting. Shared module usable across all translator types.
- **Delta trainer** (`TransformerTranslatorTrainer` in `src/core/train.py`): task loss + fidelity loss + range loss.
- **SL trainer** (`LatentTranslatorTrainer` at end of `src/core/train.py`): Phase 1 = autoencoder pretrain on MIMIC. Phase 2 = task + MMD alignment + reconstruction + range.
- **Retrieval trainer** (`RetrievalTranslatorTrainer` in `src/core/train.py`): Phase 1 = autoencoder pretrain. Phase 2 = task + fidelity + range + smoothness + importance reg, with memory bank rebuilt every `memory_refresh_epochs`.
- **Task strategy**: Retrieval is the universal paradigm (best or tied on all 3 tasks). SL+FG for mortality. Retrieval V5 (n_cross_layers=3) for AKI. Retrieval V4+MMD for sepsis.
- **Cross-domain normalization** (`use_target_normalization`): Affine renorm of source features to target stats. Params saved in checkpoint.
- **Regression support** (`training.task_type: "regression"`): LoS (Length of Stay) and KF (KidneyFunction). Auto-detected from gin config (`Run.mode = "Regression"`). Uses MSE loss, MAE/MSE/RMSE/R2 metrics. Label prediction uses MSE instead of BCE. Oversampling/subsampling disabled for regression.
- **MI-optional schema** (`src/core/schema.py`): LoS LSTM has 52 features (no MissingIndicator). SchemaResolver synthesizes zeros for MI when columns are absent.
- **Generated feature recomputation** (`src/core/schema.py`): KF LSTM has 292 features including cumulative stats (`_min_hist`, `_max_hist`, `_count`, `_mean_hist`). After translation, `rebuild()` recomputes cumulative min/max/mean from translated values; count is unchanged.
- **Phase 1 checkpoint reuse** (SL/Retrieval): Phase 1 (autoencoder pretrain) trains only on target data. The resulting `pretrain_checkpoint.pt` is reusable across experiments for the same task, provided: same architecture (`d_latent`, `d_model`, `n_enc_layers`, `n_dec_layers`, **`n_cross_layers`**), same target data (`target_data_dir`), same `pretrain_epochs`, same seed, **same `phase1_self_retrieval`**. **Important**: self-retrieval pretrain checkpoints are incompatible with non-self-retrieval ones (different cross-attention weights). The fingerprint system in `manage_pretrain.py` handles this automatically.
- **V6 Self-Retrieval Phase 1** (Retrieval only): When `phase1_self_retrieval: true`, Phase 1 builds a MIMIC memory bank from the encoder's own representations and provides real context to cross-attention blocks during pretraining (instead of zero tensors). This eliminates the degenerate pass-through initialization. Memory bank is refreshed every `phase1_memory_refresh_epochs` (default: same as `memory_refresh_epochs`). At end of Phase 1, bank is discarded so Phase 2 rebuilds fresh.
- **V6 LR Scheduling**: `lr_scheduler: "cosine"` with `lr_min: 1e-6` enables CosineAnnealingLR decay from initial LR to eta_min over training. Optional `lr_warmup_epochs` adds linear warmup before cosine decay. `lr_scheduler: "plateau"` uses ReduceLROnPlateau. Phase 2 only (no scheduling during Phase 1 pretrain). Scheduler state is saved/restored in resume checkpoints.
- **V6 Gradient Clipping**: `grad_clip_norm: 1.0` clips gradient norm before optimizer step. Applied in all 3 trainers.
- **V6 Gradient Accumulation**: `accumulate_grad_batches: 4` accumulates gradients over N mini-batches before stepping (effective batch = batch_size × N). Useful for sparse-label tasks (sepsis). Applied in all 3 trainers. Remaining gradients are flushed at end of epoch.
- **Checkpoint resume**: Training saves `latest_checkpoint.pt` every epoch. If training is interrupted, restarting with the same config auto-resumes from the latest checkpoint (epoch, optimizer state, scheduler state, best metric all restored).

## Safety & Validation (CRITICAL)

These rules prevent catastrophic failures. Violating any one can silently ruin results.

- **Frozen Baseline**: `requires_grad=False` on all baseline params. `verify_baseline_determinism` check at startup.
- **Baseline in train() mode**: Must use `model.train()` not `model.eval()` — cuDNN RNN backward requires it.
- **Padding Integrity**: Translator output `masked_fill` ensures padded timesteps remain exactly 0.0.
- **Time-Travel Rules**: Sepsis/AKI → `temporal_attention_mode="causal"`. Mortality → `"bidirectional"`.
- **lambda_fidelity > 0**: Setting `lambda_fidelity=0.0` causes catastrophic divergence (AUCROC -0.101). Never disable.
- **VLB incompatible with mortality**: `variable_length_batching=true` silently truncates per-stay sequences to length 1. Use `false` for mortality.
- **SL OOM**: SharedLatentTranslator uses ~2.5x more GPU memory. Use `batch_size=16` (not 32) on V100-32GB.
- **Retrieval OOM**: Memory bank (`window_latents`) is GPU-resident. Use `batch_size=16`. Control rebuild cost with `memory_refresh_epochs`.
- **Retrieval detach rule**: Always `src_latent.detach()` before querying memory bank — prevents backprop through k-NN/bank.
- **YAIB leakage rule**: NEVER use different `data_dir` for train and eval. Subsampling must happen within the YAIB split (`_apply_negative_subsampling()`), not via separate cohorts.
- **AMP dtype**: Always `.float()` hidden states before passing to discriminators, loss functions, or MLPs (float16→float32 mismatch).

## Coding Standards

- Logging: `logging.info()` — never `print` in core modules.
- Tensors: Always handle device placement explicitly (`.to(device)`).
- Config backward compat: All new config keys must default to disabled (0/None/False).
- Config files: JSON in `configs/` (base) and `experiments/configs/` (experiments).
- **`_get_training_config()` whitelist** (CRITICAL): This function in `cli.py` explicitly lists all config keys. New training config keys MUST be added here or they are **silently dropped**. This is the #1 source of "config change had no effect" bugs.

## Experiments

- **Check before recomputing**: When checking experiment results, ALWAYS check log files (`experiments/results/*.json`, `runs/*/run.log`) and existing outputs first. Never re-evaluate experiments when results are already available.
- **Ablation discipline**: When designing ablation experiments, change EXACTLY ONE variable at a time unless explicitly told otherwise. Verify each config diff against the control before queuing.
- **Remote sync gate**: Before queuing experiments on remote servers, verify that all relevant code changes have been pushed. Run `git log origin/<branch>..HEAD` to check for unpushed commits.
- **Empirical estimates**: When estimating resource usage (GPU VRAM, disk, compute time), prefer actual measurements from prior runs (`nvidia-smi`, log files) over theoretical calculations. Flag when estimates are theoretical.

## Config Structure

JSON configs with two main sections:
- `"translator"`: `type` ("transformer"|"shared_latent"|"retrieval"), `d_model`, `d_latent`, `n_layers`, `n_enc_layers`, `n_dec_layers`, `n_cross_layers`, `output_mode`, etc.
- `"training"`: `epochs`, `lr`, `batch_size`, `lambda_fidelity`, `lambda_range`, `oversampling_factor`, `variable_length_batching`, `pretrain_epochs`, `lambda_align`, `lambda_recon`, `lambda_target_task`, `lambda_label_pred`, `negative_subsample_count`, `shuffle`, `use_target_normalization`, `early_stopping_patience`, `best_metric`, `k_neighbors`, `retrieval_window`, `n_cross_layers`, `output_mode`, `memory_refresh_epochs`, `lambda_importance_reg`, `lambda_smooth`, `feature_gate`, `training_seed`, `task_type` ("classification"|"regression"), `lr_scheduler` ("cosine"|"plateau"|null), `lr_min`, `lr_warmup_epochs`, `grad_clip_norm`, `phase1_self_retrieval`, `phase1_memory_refresh_epochs`, `accumulate_grad_batches`.

## Experiment Queue System

All experiments are managed through `experiments/queue.yaml`. This is the single source of truth.

### Rules for Claude Sessions
- **NEVER launch experiments directly** with `python run.py`. Always add to the queue.
- To add an experiment: append an entry to `experiments/queue.yaml` under the pending section with `status: pending`.
- To prioritize: move the entry higher in the list (scheduler runs top-to-bottom).
- To run the queue: `python scripts/gpu_scheduler.py` (usually already running in a tmux/screen session).
- To check status: `python scripts/gpu_scheduler.py --status`
- Config files must exist before adding to queue. Create the config JSON first, then add the queue entry.

### GPU Rules
- Daytime (09:00-21:00): max 2 GPUs. Prefer GPUs 0, 1. Avoid GPU 3.
- Nighttime (21:00-09:00): max 3 GPUs. Can use GPU 2. GPU 3 only as last resort.
- These rules are enforced by the scheduler automatically.
- If launching a one-off manual experiment (debugging), use GPU 3 to avoid conflicts.

### Queue Entry Format
Each experiment needs: `name` (unique ID), `config` (path to JSON config), `output` (parquet output path), `status: pending`, and optionally `notes`, `server`, `branch`, `command`.
- `command`: The run.py subcommand to execute. Default: `"train_and_eval"`. Alternative: `"translate_and_eval"` (eval-only, reuses existing checkpoint).

### Queue File Locking
The scheduler uses `fcntl.flock()` on `experiments/queue.yaml.lock` for exclusive file locking during all load+modify+save cycles. This prevents race conditions between the scheduler daemon and concurrent `--add` or `--status` operations. The lock file is created automatically.

### SLURM Server Rules
- Servers with `slurm: true` are **never auto-assigned** by the scheduler. They are managed by `athena_submit.py`.
- Pinning an experiment to a SLURM server (via `server` field) will log a warning and skip the experiment.
- `--status` displays active SLURM jobs via `squeue` for SLURM servers.
- **Distributing pending experiments**: The GPU scheduler only auto-assigns to local/a6000/3090 (non-SLURM). When experiments are pending, distribute them: keep some in queue.yaml (`status: pending`) for the scheduler to pick up on local/remote servers, and submit overflow to Athena via `athena_submit.py` (up to 2 concurrent, QoS limit). Aim for balanced utilization across all servers, not all-to-one. Steps for Athena: (1) `athena_submit.py --sync`, (2) `scp` pretrain checkpoints, (3) `athena_submit.py --config ... --name ...`, (4) mark as `athena_pending` in queue.yaml to prevent double-scheduling.
- **Athena sepsis eval segfaults**: Athena has a known polars segfault during eval for sepsis task. If training completes but eval crashes, re-eval locally with `command: translate_and_eval`.

### Screening & Calibration
Fast-iteration screening system for testing config changes before committing to full runs:
- `status: screening` and `status: calibration` are treated like `pending` by the scheduler.
- On completion, the scheduler tags them `screening_done` / `calibration_done`.
- Screening configs override `epochs` to 3-5 and reuse pretrain checkpoints.
- **Pretrain checkpoint management**: `scripts/manage_pretrain.py --auto-copy CONFIG` finds and copies a matching checkpoint into the config's run_dir. Fingerprint = (task, d_latent, d_model, n_enc_layers, n_dec_layers, **n_cross_layers**, pretrain_epochs, seed).
- **Screening workflow**: `scripts/screen_experiment.py --config CONFIG` creates a reduced-epoch config, adds it to the queue, waits for completion, and outputs ACCEPT/UNCERTAIN/REJECT.
- **Calibration**: `scripts/calibrate_screening.py --task TASK --paradigm PARADIGM --submit` validates that short runs predict final rankings (Spearman ρ).
- **AutoResearch**: `scripts/autoresearch.py --task TASK --paradigm PARADIGM --budget 12h` runs autonomous hyperparameter hill-climbing via screening.
- **Leaderboard**: `scripts/compare_results.py --task TASK [--include-screening]` shows unified leaderboard.

### Branch-Aware Experiments
- Add `branch` field to queue entries to run experiments from a specific git branch.
- The scheduler uses git worktrees for code isolation — each branch gets its own copy.
- Omitting `branch` defaults to current branch (backward compatible).
- Checkpoints/logs/outputs are always centralized in the main tree (not in worktrees).
- CLI: `--add --branch BRANCH` sets the branch. `--cleanup [--branch BRANCH]` removes worktrees.
- Worktree locations: `EHR_Translator_worktrees/<sanitized-branch>/` (sibling of `EHR_Translator/`).
- **Branches must be committed AND pushed to origin before queuing remote experiments.** Remote worktrees are created from `origin/<branch>`. If the branch isn't pushed, remote experiments will fail.
- **Local experiments on the current branch run from REPO directly** (no worktree). Remote experiments always use a worktree, even if the branch matches the local checkout.
- `sync_remote.sh` is still needed for YAIB, pretrained models, and gin config path fixes (not for experiment code — that's handled by worktrees).
- **Killing remote experiments**: `kill <PID>` only kills the SSH wrapper. Use `ssh <host> kill <child_PID>` to kill the actual training process (check `nvidia-smi` for the GPU-using PID).
