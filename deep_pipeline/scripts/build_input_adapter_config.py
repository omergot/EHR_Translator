#!/usr/bin/env python3
"""build_input_adapter_config.py

Generate a ready-to-run input-adapter config from user choices that map to the
decision tree in the EHR + AdaTime Input-Adapter Playbooks
(`docs/neurips/input_adapter_playbook.md` and
`docs/neurips/adatime_input_adapter_playbook.md`).

CLI-first; falls back to interactive prompts if any required arg is missing.

Output: a single JSON config written to a user-specified path. Schema fidelity
is guaranteed by loading an on-disk template config per leaf and applying only
the playbook-documented overrides on top of it -- no invented fields.

Importable surface (for tests):
    LeafResolver, LeafTemplate, ConfigBuilder, Validator,
    LEAF_TEMPLATES, deep_merge, resolve_leaf
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

EHR_LEAVES = {"L1", "L2", "L3", "L4", "L5"}
ADATIME_LEAVES = {"LA1", "LA2", "LA3"}

# Logger configured in main()
logger = logging.getLogger("build_input_adapter_config")


# ---------------------------------------------------------------------------
# Leaf templates (per-leaf rationale + on-disk template + override dict)
# ---------------------------------------------------------------------------


@dataclass
class LeafTemplate:
    """A leaf of the decision tree.

    Attributes:
        leaf_id: One of L1..L5, LA1..LA3.
        rationale: One-liner from the playbook leaf entry.
        template_candidates: Ordered list of relative paths (vs REPO_ROOT) to
            try as the on-disk template. The first existing file is used; if
            none exist the resolver reports a fallback failure.
        translator_overrides: Dict of fields to deep-merge into config["translator"].
        training_overrides: Dict of fields to deep-merge into config["training"].
        top_level_overrides: Dict of fields to deep-merge into the top-level config.
        playbook_section: Citation for the playbook §X.Y the override comes from.
        regime: "ehr" or "adatime" (top-of-tree split).
    """

    leaf_id: str
    rationale: str
    template_candidates: List[str]
    translator_overrides: Dict[str, Any] = field(default_factory=dict)
    training_overrides: Dict[str, Any] = field(default_factory=dict)
    top_level_overrides: Dict[str, Any] = field(default_factory=dict)
    playbook_section: str = ""
    regime: str = "ehr"


# Per-leaf override dicts are taken DIRECTLY from the playbooks' §2.3 leaf
# rationales and §10/§12 concrete config snippets. Each override has a comment
# mapping it back to the playbook section.
LEAF_TEMPLATES: Dict[str, LeafTemplate] = {
    # -----------------------------------------------------------------------
    # EHR leaves (L1-L5) -- input_adapter_playbook.md §2.3 + §10
    # -----------------------------------------------------------------------
    "L1": LeafTemplate(
        leaf_id="L1",
        rationale=(
            "KF-class regression with cumulative-stat schema: drop both MMD and "
            "fidelity, lower lr to 3e-5, keep time-delta. Mechanism: 192/292 "
            "features are recomputed downstream from translated dynamics; "
            "monotone cum_max/cum_min compound per-step drift."
        ),
        template_candidates=[
            "configs/seeds/kf_lr3e5_nfnm_s2222.json",
            "configs/seeds/kf_lr3e5_nfnm_s42.json",
        ],
        translator_overrides={
            "temporal_attention_mode": "bidirectional",  # §2.3 L1
        },
        training_overrides={
            "task_type": "regression",                   # §2.3 L1 / §10 L1
            "lr": 3e-05,                                 # §2.3 L1 lr=3e-5
            "lambda_recon": 0.0,                         # §2.3 L1 drop fidelity
            "lambda_align": 0.0,                         # §2.3 L1 drop MMD
            "n_cross_layers": 0,                         # §2.3 L1 / §4 R5
            "output_mode": "absolute",                   # §4 R6
            "use_target_normalization": True,            # §4 (default)
            "best_metric": "val_task",
        },
        playbook_section="EHR §2.3 L1 / §10 L1",
        regime="ehr",
    ),
    "L2": LeafTemplate(
        leaf_id="L2",
        rationale=(
            "LoS-class per-timestep regression, dense MSE: drop MMD only. Keep "
            "fidelity (dense per-timestep MSE residual already low-variance "
            "distribution-matching)."
        ),
        template_candidates=[
            "configs/seeds/los_v5_cross3_C3_no_mmd_s42.json",
            "configs/seeds/los_v5_cross3_C3_no_mmd_s7777.json",
        ],
        translator_overrides={
            "temporal_attention_mode": "causal",         # §2.3 L2
        },
        training_overrides={
            "task_type": "regression",                   # §2.3 L2
            "lr": 1e-04,                                 # §2.3 L2 default
            "lambda_recon": 0.1,                         # §2.3 L2 keep fidelity
            "lambda_align": 0.0,                         # §2.3 L2 drop MMD
            "n_cross_layers": 0,                         # §2.3 L2 / §4 R5
            "output_mode": "absolute",                   # §4 R6
            "use_target_normalization": True,
            "best_metric": "val_task",
        },
        playbook_section="EHR §2.3 L2 / §10 L2",
        regime="ehr",
    ),
    "L3": LeafTemplate(
        leaf_id="L3",
        rationale=(
            "Mortality-class per-stay binary classification: drop fidelity, "
            "keep MMD (null), keep aux target_task. n_cross_layers=2 with "
            "absolute output. Pretrain mandatory; never combine "
            "pretrain_epochs=0 with lambda_recon=0."
        ),
        template_candidates=[
            "configs/seeds/mort_c2_C5_no_fidelity_s2222.json",
            "configs/seeds/mort_c2_C5_no_fidelity_s42.json",
        ],
        translator_overrides={
            "temporal_attention_mode": "bidirectional",  # §2.3 L3 / Mortality24
        },
        training_overrides={
            "lr": 1e-04,                                 # §2.3 L3
            "lambda_recon": 0.0,                         # §2.3 L3 drop fidelity
            "lambda_align": 0.5,                         # §2.3 L3 keep MMD
            "lambda_target_task": 0.5,                   # §2.3 L3 keep aux task
            "n_cross_layers": 2,                         # §2.3 L3
            "output_mode": "absolute",                   # §4 R6
            "pretrain_epochs": 15,                       # §4 R7 mandatory
            "variable_length_batching": False,           # mortality / per-stay
            "use_target_normalization": True,
            "best_metric": "val_task",
        },
        playbook_section="EHR §2.3 L3 / §10 L3",
        regime="ehr",
    ),
    "L4": LeafTemplate(
        leaf_id="L4",
        rationale=(
            "AKI-class per-timestep binary classification (>=10% positive rate): "
            "defaults except n_cross_layers=0 (tightens seed sigma 2.5x without "
            "losing mean). Do NOT drop fidelity at this density."
        ),
        template_candidates=[
            "configs/seeds/aki_v5_cross3_C1_no_retrieval_s42.json",
            "configs/seeds/aki_v5_cross3_C1_no_retrieval_s7777.json",
        ],
        translator_overrides={
            "temporal_attention_mode": "causal",         # §2.3 L4 (per-timestep)
        },
        training_overrides={
            "lr": 1e-04,
            "lambda_recon": 0.1,                         # §2.3 L4 keep fidelity
            "lambda_align": 0.5,                         # default
            "lambda_target_task": 0.5,                   # default
            "n_cross_layers": 0,                         # §2.3 L4 / §4 R5
            "output_mode": "absolute",                   # §4 R6
            "pretrain_epochs": 15,                       # §4 R7
            "variable_length_batching": True,            # per-timestep VLB-safe
            "use_target_normalization": True,
            "best_metric": "val_task",
        },
        playbook_section="EHR §2.3 L4 / §10 L4",
        regime="ehr",
    ),
    "L5": LeafTemplate(
        leaf_id="L5",
        rationale=(
            "Sepsis-class per-timestep binary classification (<5% positive "
            "rate): KEEP fidelity (catastrophic to drop, BCE -> NaN by ep 8), "
            "n_cross_layers=0, consider dropping aux target_task."
        ),
        template_candidates=[
            "configs/seeds/sepsis_v5_cross3_C1_no_retrieval_s2222.json",
            "configs/seeds/sepsis_v5_cross3_C1_no_retrieval_s42.json",
        ],
        translator_overrides={
            "temporal_attention_mode": "causal",         # §2.3 L5
        },
        training_overrides={
            "lr": 1e-04,
            "lambda_recon": 0.1,                         # §2.3 L5 KEEP fidelity
            "lambda_align": 0.5,                         # §2.3 L5 keep MMD
            "lambda_target_task": 0.5,                   # default; drop optional
            "n_cross_layers": 0,                         # §2.3 L5
            "output_mode": "absolute",                   # §4 R6
            "pretrain_epochs": 15,                       # §4 R7
            "variable_length_batching": True,
            "use_target_normalization": True,
            "best_metric": "val_task",
        },
        playbook_section="EHR §2.3 L5 / §10 L5",
        regime="ehr",
    ),
    # -----------------------------------------------------------------------
    # AdaTime leaves (LA1-LA3) -- adatime_input_adapter_playbook.md §2.3 + §12
    # -----------------------------------------------------------------------
    "LA1": LeafTemplate(
        leaf_id="LA1",
        rationale=(
            "HAR-like controlled single-protocol multi-channel: Tiny tier, "
            "p=0, n_cross=2-3, residual (per §1 A3 / §6: AdaTime -> residual "
            "universal; predictor-architecture-keyed). Tune (d_ff=144, "
            "dropout=0.10) per the documented HAR-Tiny multi-seed crosser."
        ),
        template_candidates=[
            "experiments/.athena_configs/adatime_har_cap_T_s0.json",
        ],
        translator_overrides={
            "output_mode": "residual",                   # AdaTime §1 A3 / §6 regime-split: p=0 AND no cumulative -> residual
            "d_ff": 144,                                 # AdaTime §2.3 LA1 / har_tiny_variants
            "dropout": 0.10,                             # AdaTime §2.3 LA1
            "out_dropout": 0.10,                         # match dropout for consistency
            "n_cross_layers": 2,                         # AdaTime §2.3 LA1
        },
        training_overrides={
            "pretrain_epochs": 0,                        # AdaTime §5.1 / R-A1 universal p=0
            "lambda_recon": 0.1,                         # §6 keep at default
            "lambda_fidelity": 0.1,                      # §6 LA1 (HAR cap_T)
        },
        playbook_section="AdaTime §2.3 LA1 / §12 LA1",
        regime="adatime",
    ),
    "LA2": LeafTemplate(
        leaf_id="LA2",
        rationale=(
            "SSC/MFD-like 1-channel quasi-stationary: XT (0.34x) or Tiny "
            "tier, p=0, n_cross=2 (KEEP retrieval; manifold-coherent), "
            "residual default (AdaTime universal-residual per §1 A3; "
            "predictor-architecture-keyed). Pass --output-mode absolute "
            "only to opt into the MFD-XT-style tier-shifted variant "
            "(different n_enc_layers; not a strict toggle)."
        ),
        template_candidates=[
            # SSC cap_T (Tiny tier, residual default) — direct LA2 match.
            "experiments/.athena_configs/adatime_ssc_cap_T_s0.json",
            # MFD residual-Full and SSC residual-best fallbacks.
            "experiments/.athena_configs/adatime_mfd_best_res_s0.json",
            "experiments/.athena_configs/adatime_ssc_best_s0.json",
        ],
        translator_overrides={
            "output_mode": "residual",                   # AdaTime §1 A3 / §6 regime-split default
            "n_cross_layers": 2,                         # AdaTime §3.2 keep retrieval
        },
        training_overrides={
            "pretrain_epochs": 0,                        # AdaTime §5.1 universal
            "lambda_recon": 0.1,                         # §6
            # lambda_fidelity left to template (1.0 for MFD-style, 0.5 typical)
        },
        playbook_section="AdaTime §2.3 LA2 / §12 LA2",
        regime="adatime",
    ),
    "LA3": LeafTemplate(
        leaf_id="LA3",
        rationale=(
            "HHAR/WISDM-like heterogeneous-source: Tiny tier, p=0, "
            "n_cross_layers=0 (DROP retrieval; k-NN returns wrong-cluster "
            "neighbours), residual (per §1 A3 / §6: AdaTime -> residual "
            "universal; predictor-architecture-keyed)."
        ),
        template_candidates=[
            "experiments/.athena_configs/adatime_hhar_cap_T_p0_s0.json",
            "experiments/.athena_configs/adatime_wisdm_cap_T_p0_s0.json",
        ],
        translator_overrides={
            "output_mode": "residual",                   # AdaTime §1 A3 / §6 regime-split default
            "n_cross_layers": 0,                         # AdaTime §3.2 / LA3 DROP retrieval
        },
        training_overrides={
            "pretrain_epochs": 0,                        # AdaTime §5.1
            "lambda_recon": 0.1,                         # §6
            "lambda_fidelity": 0.5,                      # §6 LA3 (HHAR cap_T_p0)
        },
        playbook_section="AdaTime §2.3 LA3 / §12 LA3",
        regime="adatime",
    ),
}


# ---------------------------------------------------------------------------
# Output-mode override (cross-benchmark regime-split rule)
# ---------------------------------------------------------------------------


def apply_output_mode_override(
    leaf: LeafTemplate, output_mode: Optional[str]
) -> LeafTemplate:
    """Apply a user-supplied --output-mode override to the leaf overrides.

    Implements the cross-benchmark `output_mode` rule, post Apr 26
    claim-strengthening run (`adatime_input_adapter_playbook.md` §1 A3 / §6;
    `input_adapter_playbook.md` §1 R6;
    `playbook_drafts/output_mode_multivariable_audit.md` Phase 6):

        AdaTime: residual is universal -- wins or ties at every measured
        pretrain_epochs x lambda_fidelity cell across HAR/HHAR/WISDM/SSC/MFD.
        EHR: absolute is universal -- 5/5 tasks at n=3 via C8 strict toggle.
        Cross-benchmark split is keyed on the predictor + feature regime
        (frozen 1D-CNN over raw low-dim time-series -> residual; frozen
        LSTM over tabular ICU features -> absolute), NOT on pretrain_epochs.

    The previously-documented `pretrain_epochs`-keyed bridge ("residual when
    p=0; otherwise absolute") is DEPRECATED -- the Apr 26 strict-toggle run
    on HAR (cap_T_p10 RES +24.86 MF1) and WISDM (v4_lr67_fid05 at p=10,
    lambda_fid=0.5 RES +13.41 MF1) refuted the `p > 0 -> absolute` direction.
    The HHAR v4_base s0 cell that previously supported it collapsed to a
    2-seed within-sigma tie when s1 was added.

    Per-leaf defaults (the `auto` resolution) remain correct: EHR L1-L5
    default to `"absolute"`; AdaTime LA1-LA3 default to `"residual"`. Both
    are now justified by the predictor + feature regime, not by the
    deprecated `p`-keyed precondition.

    Boundary case (preserved): residual at p>0 with lambda_recon=0 (e.g.
    aki_nf_C8) lands at ~+0.0002 single-seed -- residual reappears
    marginally competitive on the EHR side only when the fidelity anchor
    is also stripped. R7's hard guardrail prevents that combination in
    practice.

    A user-supplied override is honoured verbatim and replaces the leaf
    default. Most users should leave `auto` set; AdaTime universal-residual
    and EHR universal-absolute are the data-best defaults.
    """
    if output_mode in (None, "auto"):
        return leaf
    if output_mode not in ("residual", "absolute"):
        raise LeafResolutionError(
            f"output_mode must be 'auto'|'residual'|'absolute', got {output_mode!r}"
        )
    # The codebase splits `output_mode` between `translator.output_mode`
    # (AdaTime configs) and `training.output_mode` (EHR configs). Honour
    # whichever section the leaf already keys it into; default to translator
    # for AdaTime and training for EHR if neither has it.
    new_translator_overrides = dict(leaf.translator_overrides)
    new_training_overrides = dict(leaf.training_overrides)
    if "output_mode" in new_translator_overrides:
        new_translator_overrides["output_mode"] = output_mode
    elif "output_mode" in new_training_overrides:
        new_training_overrides["output_mode"] = output_mode
    elif leaf.regime == "ehr":
        new_training_overrides["output_mode"] = output_mode
    else:
        new_translator_overrides["output_mode"] = output_mode
    return LeafTemplate(
        leaf_id=leaf.leaf_id,
        rationale=leaf.rationale,
        template_candidates=list(leaf.template_candidates),
        translator_overrides=new_translator_overrides,
        training_overrides=new_training_overrides,
        top_level_overrides=dict(leaf.top_level_overrides),
        playbook_section=leaf.playbook_section,
        regime=leaf.regime,
    )


# ---------------------------------------------------------------------------
# Leaf resolution (CLI / interactive -> leaf_id)
# ---------------------------------------------------------------------------


class LeafResolutionError(RuntimeError):
    """Raised when the inputs do not resolve to a unique leaf."""


def resolve_leaf(
    predictor_regime: str,
    *,
    task_type: Optional[str] = None,
    granularity: Optional[str] = None,
    feature_schema: Optional[str] = None,
    label_density: Optional[str] = None,
    heterogeneity: Optional[str] = None,
) -> str:
    """Walk the decision tree from user choices to a leaf id.

    Mirrors EHR playbook §2.2 and AdaTime playbook §2.2.
    """
    pr = predictor_regime.lower().strip()
    if pr not in {"ehr", "adatime"}:
        raise LeafResolutionError(
            f"predictor_regime must be 'ehr' or 'adatime', got {predictor_regime!r}"
        )

    if pr == "adatime":
        if heterogeneity is None:
            raise LeafResolutionError("AdaTime branch requires --heterogeneity")
        h = heterogeneity.lower().strip()
        mapping = {
            "controlled": "LA1",
            "single-channel": "LA2",
            "heterogeneous": "LA3",
        }
        if h not in mapping:
            raise LeafResolutionError(
                f"heterogeneity must be one of {sorted(mapping)}, got {heterogeneity!r}"
            )
        return mapping[h]

    # EHR branch
    if task_type is None:
        raise LeafResolutionError("EHR branch requires --task-type")
    tt = task_type.lower().strip()
    if tt == "regression":
        if feature_schema is None:
            raise LeafResolutionError(
                "EHR regression requires --feature-schema "
                "(cumulative-stat -> L1 / standard|mi-optional -> L2)"
            )
        fs = feature_schema.lower().strip()
        if fs == "cumulative-stat":
            return "L1"
        if fs in {"standard", "mi-optional"}:
            return "L2"
        raise LeafResolutionError(
            f"feature_schema must be 'cumulative-stat'|'standard'|'mi-optional', got {feature_schema!r}"
        )
    if tt == "classification":
        if granularity is None:
            raise LeafResolutionError(
                "EHR classification requires --granularity (per-stay -> L3 / per-timestep -> L4|L5)"
            )
        g = granularity.lower().strip()
        if g == "per-stay":
            return "L3"
        if g == "per-timestep":
            if label_density is None:
                raise LeafResolutionError(
                    "EHR per-timestep classification requires --label-density "
                    "(dense -> L4 / sparse -> L5 / borderline -> L4)"
                )
            ld = label_density.lower().strip()
            if ld == "dense":
                return "L4"
            if ld == "sparse":
                return "L5"
            if ld == "borderline":
                logger.warning(
                    "label_density=borderline: routing to L4 (defaults). "
                    "Run the §3.3 cos(task,fid) pilot to confirm."
                )
                return "L4"
            raise LeafResolutionError(
                f"label_density must be 'dense'|'sparse'|'borderline', got {label_density!r}"
            )
        raise LeafResolutionError(
            f"granularity must be 'per-stay'|'per-timestep', got {granularity!r}"
        )
    raise LeafResolutionError(
        f"task_type must be 'regression'|'classification', got {task_type!r}"
    )


@dataclass
class LeafResolver:
    """Holds the resolved leaf id together with the choices that produced it."""

    predictor_regime: str
    task_type: Optional[str] = None
    granularity: Optional[str] = None
    feature_schema: Optional[str] = None
    label_density: Optional[str] = None
    heterogeneity: Optional[str] = None
    output_mode: Optional[str] = None

    def resolve(self) -> Tuple[str, LeafTemplate]:
        leaf_id = resolve_leaf(
            self.predictor_regime,
            task_type=self.task_type,
            granularity=self.granularity,
            feature_schema=self.feature_schema,
            label_density=self.label_density,
            heterogeneity=self.heterogeneity,
        )
        leaf = LEAF_TEMPLATES[leaf_id]
        leaf = apply_output_mode_override(leaf, self.output_mode)
        return leaf_id, leaf


# ---------------------------------------------------------------------------
# Config builder (load template + apply overrides + fill user fields)
# ---------------------------------------------------------------------------


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict deep-merge. ``override`` wins on conflict.

    Lists and scalars are replaced wholesale; only dicts merge recursively.
    """
    out: Dict[str, Any] = {}
    keys = list(base.keys()) + [k for k in override.keys() if k not in base]
    for k in keys:
        if k in base and k in override:
            bv, ov = base[k], override[k]
            if isinstance(bv, dict) and isinstance(ov, dict):
                out[k] = deep_merge(bv, ov)
            else:
                out[k] = copy.deepcopy(ov)
        elif k in override:
            out[k] = copy.deepcopy(override[k])
        else:
            out[k] = copy.deepcopy(base[k])
    return out


@dataclass
class UserFields:
    """User-specific fields to fill into the generated config."""

    # EHR
    source_data_dir: Optional[str] = None
    target_data_dir: Optional[str] = None
    # AdaTime
    dataset: Optional[str] = None
    data_path: Optional[str] = None
    # Both
    seed: Optional[int] = None
    pretrain_checkpoint: Optional[str] = None
    run_name: Optional[str] = None
    output_path: Optional[str] = None


class ConfigBuilder:
    """Loads the on-disk leaf template, applies overrides, fills user fields."""

    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.repo_root = repo_root

    def resolve_template_path(self, leaf: LeafTemplate) -> Path:
        attempted: List[Path] = []
        for cand in leaf.template_candidates:
            p = (self.repo_root / cand).resolve()
            attempted.append(p)
            if p.is_file():
                logger.info("[leaf=%s] using template %s", leaf.leaf_id, p)
                return p
        msg = (
            f"Leaf {leaf.leaf_id} could not be resolved: none of the template "
            f"candidates exist on disk:\n  - "
            + "\n  - ".join(str(p) for p in attempted)
        )
        raise FileNotFoundError(msg)

    def load_template(self, leaf: LeafTemplate) -> Dict[str, Any]:
        path = self.resolve_template_path(leaf)
        with path.open("r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Template at {path} is not a JSON object (got {type(cfg).__name__})"
            )
        return cfg

    @staticmethod
    def apply_overrides(cfg: Dict[str, Any], leaf: LeafTemplate) -> Dict[str, Any]:
        """Apply leaf-specific overrides onto cfg (returns a new dict)."""
        out = copy.deepcopy(cfg)
        if leaf.translator_overrides:
            out["translator"] = deep_merge(
                out.get("translator", {}), leaf.translator_overrides
            )
        if leaf.training_overrides:
            out["training"] = deep_merge(
                out.get("training", {}), leaf.training_overrides
            )
        if leaf.top_level_overrides:
            out = deep_merge(out, leaf.top_level_overrides)
        return out

    @staticmethod
    def apply_user_fields(
        cfg: Dict[str, Any], leaf: LeafTemplate, user: UserFields
    ) -> Dict[str, Any]:
        out = copy.deepcopy(cfg)
        if leaf.regime == "ehr":
            if user.source_data_dir is not None:
                out["data_dir"] = user.source_data_dir
            if user.target_data_dir is not None:
                out["target_data_dir"] = user.target_data_dir
            if user.seed is not None:
                out["seed"] = int(user.seed)
            if user.run_name:
                out.setdefault("output", {})
                out["output"]["run_dir"] = f"runs/seeds/{user.run_name}"
                out["output"]["log_file"] = f"runs/seeds/{user.run_name}/run.log"
            if user.pretrain_checkpoint is not None:
                # Conventional location: training.pretrain_checkpoint_path
                out.setdefault("training", {})
                out["training"]["pretrain_checkpoint_path"] = user.pretrain_checkpoint
        else:  # adatime
            if user.dataset is not None:
                out["dataset"] = user.dataset
            if user.data_path is not None:
                out["data_path"] = user.data_path
            if user.seed is not None:
                out["seed"] = int(user.seed)
            if user.run_name:
                out.setdefault("output", {})
                out["output"]["run_dir"] = f"runs/adatime/{user.run_name}"
            if user.pretrain_checkpoint is not None:
                out.setdefault("training", {})
                out["training"]["pretrain_checkpoint_path"] = user.pretrain_checkpoint
        return out

    def build(
        self, leaf: LeafTemplate, user: UserFields
    ) -> Tuple[Dict[str, Any], Path]:
        template_path = self.resolve_template_path(leaf)
        cfg = self.load_template(leaf)
        cfg = self.apply_overrides(cfg, leaf)
        cfg = self.apply_user_fields(cfg, leaf, user)
        return cfg, template_path


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ValidationError(RuntimeError):
    pass


class Validator:
    """Self-checks the produced config against the leaf overrides + a schema
    reference template of the same regime."""

    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.repo_root = repo_root

    def schema_reference(self, leaf: LeafTemplate) -> Dict[str, Any]:
        """Pick a representative on-disk config (from the same leaf family) as
        the field-set reference."""
        # Use the leaf's first-found template as the reference (any other
        # well-formed config from the same regime would also work; keeping
        # them identical avoids spurious schema-drift warnings).
        path = self._find_first(leaf.template_candidates)
        if path is None:
            raise ValidationError(f"No schema reference available for leaf {leaf.leaf_id}")
        with path.open() as f:
            return json.load(f)

    def _find_first(self, candidates: List[str]) -> Optional[Path]:
        for c in candidates:
            p = (self.repo_root / c).resolve()
            if p.is_file():
                return p
        return None

    @staticmethod
    def _flatten_keys(d: Any, prefix: str = "") -> List[str]:
        out: List[str] = []
        if isinstance(d, dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                out.append(key)
                out.extend(Validator._flatten_keys(v, key))
        return out

    def validate(self, written_path: Path, leaf: LeafTemplate) -> None:
        if not written_path.is_file():
            raise ValidationError(f"Output config does not exist: {written_path}")
        with written_path.open() as f:
            cfg = json.load(f)

        # (a) leaf-specific overrides took effect
        for sub_name, overrides in [
            ("translator", leaf.translator_overrides),
            ("training", leaf.training_overrides),
        ]:
            sub = cfg.get(sub_name, {})
            for k, v in overrides.items():
                if sub.get(k) != v:
                    raise ValidationError(
                        f"[leaf={leaf.leaf_id}] override {sub_name}.{k}={v!r} did "
                        f"not take effect; observed {sub.get(k)!r}"
                    )
        for k, v in leaf.top_level_overrides.items():
            if cfg.get(k) != v:
                raise ValidationError(
                    f"[leaf={leaf.leaf_id}] top-level override {k}={v!r} did not "
                    f"take effect; observed {cfg.get(k)!r}"
                )

        # (b) field set is a subset of the reference (no invented keys at
        # any depth). User-injected keys are whitelisted explicitly.
        ref = self.schema_reference(leaf)
        ref_keys = set(self._flatten_keys(ref))
        cfg_keys = set(self._flatten_keys(cfg))
        # Whitelisted keys we may have added on top:
        whitelist = {
            # EHR-side
            "data_dir",
            "target_data_dir",
            "seed",
            "output",
            "output.run_dir",
            "output.log_file",
            "training.pretrain_checkpoint_path",
            # AdaTime-side
            "dataset",
            "data_path",
        }
        # Also whitelist any nested key the leaf overrides explicitly added
        # (those are guaranteed to be schema-valid because they came from
        # the playbook §6 / §10 / §12 catalogues).
        for k in leaf.translator_overrides:
            whitelist.add(f"translator.{k}")
        for k in leaf.training_overrides:
            whitelist.add(f"training.{k}")
        for k in leaf.top_level_overrides:
            whitelist.add(k)

        invented = cfg_keys - ref_keys - whitelist
        if invented:
            raise ValidationError(
                f"[leaf={leaf.leaf_id}] config contains keys not present in the "
                f"schema reference and not in the override whitelist: "
                f"{sorted(invented)[:10]}{' ...' if len(invented) > 10 else ''}"
            )
        logger.info(
            "[leaf=%s] validated: %d overrides applied; %d keys all in schema",
            leaf.leaf_id,
            len(leaf.translator_overrides) + len(leaf.training_overrides) + len(leaf.top_level_overrides),
            len(cfg_keys),
        )


# ---------------------------------------------------------------------------
# Interactive fallback
# ---------------------------------------------------------------------------


def _ask(prompt: str, choices: Optional[List[str]] = None, allow_empty: bool = False) -> str:
    while True:
        if choices:
            full = f"{prompt} [{'|'.join(choices)}]: "
        else:
            full = f"{prompt}: "
        ans = input(full).strip()
        if not ans and not allow_empty:
            print("  (required)")
            continue
        if choices and ans and ans not in choices:
            print(f"  (must be one of {choices})")
            continue
        return ans


def fill_interactively(args: argparse.Namespace) -> argparse.Namespace:
    print("\n=== Interactive build_input_adapter_config ===\n")
    if not args.predictor_regime:
        args.predictor_regime = _ask(
            "Predictor regime (frozen recurrent EHR vs frozen 1D-CNN AdaTime?)",
            ["ehr", "adatime"],
        )
    if args.predictor_regime == "ehr":
        if not args.task_type:
            args.task_type = _ask("Task type", ["classification", "regression"])
        if args.task_type == "classification" and not args.granularity:
            args.granularity = _ask(
                "Prediction granularity", ["per-stay", "per-timestep"]
            )
        if args.task_type == "classification" and args.granularity == "per-timestep" and not args.label_density:
            args.label_density = _ask(
                "Per-timestep positive-label density",
                ["dense", "sparse", "borderline"],
            )
        if args.task_type == "regression" and not args.feature_schema:
            args.feature_schema = _ask(
                "Feature schema",
                ["standard", "cumulative-stat", "mi-optional"],
            )
        if not args.source_data_dir:
            args.source_data_dir = _ask(
                "Source data dir (eICU/HiRID YAIB cohort root)"
            )
        if not args.target_data_dir:
            args.target_data_dir = _ask(
                "Target data dir (MIMIC-IV YAIB cohort root)"
            )
    else:
        if not args.heterogeneity:
            args.heterogeneity = _ask(
                "Source-target heterogeneity profile",
                ["controlled", "single-channel", "heterogeneous"],
            )
        if not args.dataset:
            args.dataset = _ask("AdaTime dataset (HAR/HHAR/WISDM/SSC/MFD)")
        if not args.data_path:
            args.data_path = _ask("AdaTime data_path (root directory)")

    if args.seed is None:
        args.seed = int(_ask("Seed (e.g. 2222)") or "2222")
    # Output-mode override (regime-split rule). Default 'auto' is correct
    # for almost every case; only ask once we have enough to know the leaf.
    if getattr(args, "output_mode", None) in (None, "auto"):
        ans = _ask(
            "output_mode (auto = let the leaf pick: AdaTime -> residual "
            "universal, EHR -> absolute universal; predictor-architecture-keyed)",
            choices=["auto", "residual", "absolute"],
            allow_empty=True,
        )
        args.output_mode = ans or "auto"
    if not args.output:
        args.output = _ask("Output JSON path (e.g. /tmp/my_config.json)")
    return args


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build an input-adapter config from playbook decision-tree choices. "
            "See docs/neurips/{input_adapter_playbook.md,adatime_input_adapter_playbook.md}."
        )
    )
    # Decision-tree inputs
    p.add_argument("--predictor-regime", choices=["ehr", "adatime"], default=None)
    p.add_argument("--task-type", choices=["classification", "regression"], default=None)
    p.add_argument("--granularity", choices=["per-stay", "per-timestep"], default=None)
    p.add_argument(
        "--feature-schema",
        choices=["standard", "cumulative-stat", "mi-optional"],
        default=None,
    )
    p.add_argument(
        "--label-density", choices=["dense", "sparse", "borderline"], default=None
    )
    p.add_argument(
        "--heterogeneity",
        choices=["controlled", "single-channel", "heterogeneous"],
        default=None,
    )
    p.add_argument(
        "--output-mode",
        choices=["auto", "residual", "absolute"],
        default="auto",
        help=(
            "Translator output_mode override (rewritten Apr 26 after "
            "claim-strengthening run; adatime_input_adapter_playbook.md "
            "§1 A3 / §6, output_mode_multivariable_audit.md Phase 6). "
            "AdaTime: residual universal, regardless of pretrain. "
            "EHR: absolute universal, predictor-architecture-keyed. "
            "'auto' (default) lets the leaf pick: EHR -> 'absolute' "
            "(frozen LSTM + tabular features); AdaTime -> 'residual' "
            "(frozen 1D-CNN + raw low-dim time-series). The previously-"
            "documented `pretrain_epochs`-keyed bridge is deprecated."
        ),
    )
    # User-specific fields
    p.add_argument("--dataset", default=None, help="AdaTime dataset name")
    p.add_argument("--source-data-dir", default=None)
    p.add_argument("--target-data-dir", default=None)
    p.add_argument("--data-path", default=None, help="AdaTime data_path")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pretrain-checkpoint", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output", default=None, help="Output config JSON path")
    # Modes
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print config; do not write")
    p.add_argument(
        "--explain",
        action="store_true",
        help="Print leaf rationale + override list and exit",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _ehr_required(args: argparse.Namespace) -> List[str]:
    missing: List[str] = []
    if not args.task_type:
        missing.append("--task-type")
        return missing
    if args.task_type == "regression" and not args.feature_schema:
        missing.append("--feature-schema")
    if args.task_type == "classification":
        if not args.granularity:
            missing.append("--granularity")
        elif args.granularity == "per-timestep" and not args.label_density:
            missing.append("--label-density")
    return missing


def _adatime_required(args: argparse.Namespace) -> List[str]:
    if not args.heterogeneity:
        return ["--heterogeneity"]
    return []


def _user_fields_required(args: argparse.Namespace) -> List[str]:
    missing: List[str] = []
    if args.predictor_regime == "ehr":
        if not args.source_data_dir:
            missing.append("--source-data-dir")
        if not args.target_data_dir:
            missing.append("--target-data-dir")
    elif args.predictor_regime == "adatime":
        if not args.dataset:
            missing.append("--dataset")
        if not args.data_path:
            missing.append("--data-path")
    if args.seed is None:
        missing.append("--seed")
    if not args.output:
        missing.append("--output")
    return missing


def explain_leaf(
    leaf: LeafTemplate, requested_output_mode: Optional[str] = None
) -> str:
    lines: List[str] = []
    lines.append(f"=== Leaf {leaf.leaf_id} ({leaf.regime}) ===")
    lines.append(f"Rationale: {leaf.rationale}")
    lines.append(f"Playbook section: {leaf.playbook_section}")
    # Output-mode resolution (cross-benchmark regime-split rule).
    resolved_om = (
        leaf.translator_overrides.get("output_mode")
        or leaf.training_overrides.get("output_mode")
        or "(template default)"
    )
    if requested_output_mode in (None, "auto"):
        om_source = "auto (per leaf default; predictor-architecture-keyed)"
    else:
        om_source = f"user override --output-mode {requested_output_mode}"
    lines.append(
        f"output_mode resolution: {resolved_om!r}  ({om_source})"
    )
    if leaf.regime == "adatime":
        lines.append(
            "  rule (Apr 26): AdaTime -> residual universal. Wins or ties at every "
            "measured pretrain_epochs x lambda_fidelity cell (HAR cap_T_p10 RES "
            "+24.86 MF1; WISDM v4_lr67_fid05 p=10 RES +13.41 MF1; HHAR v4_base "
            "p=10 2-seed within-sigma tie). Mechanism: frozen 1D-CNN + raw "
            "low-dim time-series. The previously-documented `p > 0 -> absolute` "
            "bridge is deprecated -- see output_mode_multivariable_audit.md "
            "Phase 6."
        )
    else:
        lines.append(
            "  rule (Apr 26): EHR -> absolute universal. 5/5 tasks at n=3 via C8 "
            "strict toggle. Mechanism: frozen LSTM + tabular ICU features regime "
            "(predictor-architecture-keyed; not `p > 0`-keyed). Boundary case: "
            "aki_nf_C8 (residual + lambda_recon=0) ~+0.0002 single-seed; R7 hard "
            "guardrail prevents that combination."
        )
    lines.append(f"Template candidates (first existing wins):")
    for c in leaf.template_candidates:
        lines.append(f"  - {c}")
    if leaf.translator_overrides:
        lines.append("translator overrides:")
        for k, v in leaf.translator_overrides.items():
            lines.append(f"  {k} = {v!r}")
    if leaf.training_overrides:
        lines.append("training overrides:")
        for k, v in leaf.training_overrides.items():
            lines.append(f"  {k} = {v!r}")
    if leaf.top_level_overrides:
        lines.append("top-level overrides:")
        for k, v in leaf.top_level_overrides.items():
            lines.append(f"  {k} = {v!r}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --explain branch: don't require user fields, just print and exit.
    if args.explain:
        if not args.predictor_regime:
            args.predictor_regime = _ask(
                "Predictor regime", ["ehr", "adatime"]
            )
        # Need just enough to pick a leaf
        if args.predictor_regime == "ehr":
            missing = _ehr_required(args)
        else:
            missing = _adatime_required(args)
        if missing and not args.interactive:
            args.interactive = True
        if args.interactive:
            args = fill_interactively(args)
        try:
            leaf_id, leaf = LeafResolver(
                predictor_regime=args.predictor_regime,
                task_type=args.task_type,
                granularity=args.granularity,
                feature_schema=args.feature_schema,
                label_density=args.label_density,
                heterogeneity=args.heterogeneity,
                output_mode=getattr(args, "output_mode", None),
            ).resolve()
        except LeafResolutionError as e:
            logger.error("Leaf resolution failed: %s", e)
            return 2
        print(explain_leaf(leaf, requested_output_mode=getattr(args, "output_mode", None)))
        return 0

    # Standard generate-config branch
    # Decide whether interactive fallback is needed.
    if not args.predictor_regime:
        args.interactive = True
    if args.predictor_regime == "ehr":
        if _ehr_required(args) or _user_fields_required(args):
            args.interactive = True
    elif args.predictor_regime == "adatime":
        if _adatime_required(args) or _user_fields_required(args):
            args.interactive = True
    if args.interactive:
        args = fill_interactively(args)

    # Final validation of inputs
    final_missing: List[str] = []
    if args.predictor_regime == "ehr":
        final_missing.extend(_ehr_required(args))
    elif args.predictor_regime == "adatime":
        final_missing.extend(_adatime_required(args))
    final_missing.extend(_user_fields_required(args))
    if final_missing:
        logger.error("Missing required arguments: %s", ", ".join(final_missing))
        parser.print_help(sys.stderr)
        return 2

    try:
        leaf_id, leaf = LeafResolver(
            predictor_regime=args.predictor_regime,
            task_type=args.task_type,
            granularity=args.granularity,
            feature_schema=args.feature_schema,
            label_density=args.label_density,
            heterogeneity=args.heterogeneity,
            output_mode=getattr(args, "output_mode", None),
        ).resolve()
    except LeafResolutionError as e:
        logger.error("Leaf resolution failed: %s", e)
        return 2

    user = UserFields(
        source_data_dir=args.source_data_dir,
        target_data_dir=args.target_data_dir,
        dataset=args.dataset,
        data_path=args.data_path,
        seed=args.seed,
        pretrain_checkpoint=args.pretrain_checkpoint,
        run_name=args.run_name,
        output_path=args.output,
    )

    builder = ConfigBuilder()
    try:
        cfg, template_path = builder.build(leaf, user)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 3

    out_path = Path(args.output).resolve()
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        print(f"\n[dry-run] would write -> {out_path}")
        print(f"[leaf={leaf_id}] {leaf.rationale}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("[leaf=%s] wrote config -> %s", leaf_id, out_path)

    # Self-validate
    validator = Validator()
    try:
        validator.validate(out_path, leaf)
    except ValidationError as e:
        logger.error("Self-validation failed: %s", e)
        return 4

    print(f"\nLeaf: {leaf_id}  ({leaf.regime})")
    print(f"Rationale: {leaf.rationale}")
    resolved_om = (
        leaf.translator_overrides.get("output_mode")
        or leaf.training_overrides.get("output_mode")
        or "(template default)"
    )
    requested = getattr(args, "output_mode", "auto") or "auto"
    print(
        f"output_mode: {resolved_om!r}  "
        f"({'auto' if requested == 'auto' else 'user override'})"
    )
    print(f"Template:  {template_path}")
    print(f"Output:    {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
