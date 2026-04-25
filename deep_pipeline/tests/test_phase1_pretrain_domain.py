"""Tests for the NTU `phase1_pretrain_domain` knob.

Covers:
- `_get_training_config()` round-trips the new key (silent-bug class — CLAUDE.md flags
  the whitelist as the #1 source of "config change had no effect" bugs).
- `PretrainFingerprint` disambiguates target-pretrain vs source-pretrain checkpoints,
  so `manage_pretrain.py --auto-copy` will not cross-contaminate.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from src.cli import _get_training_config

_REPO = Path(__file__).resolve().parent.parent
_MP_PATH = _REPO / "scripts" / "manage_pretrain.py"
_spec = importlib.util.spec_from_file_location("manage_pretrain", _MP_PATH)
_mp = importlib.util.module_from_spec(_spec)
sys.modules["manage_pretrain"] = _mp
_spec.loader.exec_module(_mp)
PretrainFingerprint = _mp.PretrainFingerprint
fingerprint_from_config = _mp.fingerprint_from_config


def test_phase1_pretrain_domain_default_is_target():
    cfg = {"training": {}}
    out = _get_training_config(cfg)
    assert out["phase1_pretrain_domain"] == "target"


def test_phase1_pretrain_domain_round_trip_source():
    cfg = {"training": {"phase1_pretrain_domain": "source"}}
    out = _get_training_config(cfg)
    assert out["phase1_pretrain_domain"] == "source"


def test_phase1_pretrain_domain_round_trip_target_explicit():
    cfg = {"training": {"phase1_pretrain_domain": "target"}}
    out = _get_training_config(cfg)
    assert out["phase1_pretrain_domain"] == "target"


def _base_config(domain=None):
    cfg = {
        "translator": {
            "type": "retrieval",
            "d_latent": 128,
            "d_model": 128,
            "n_enc_layers": 4,
            "n_dec_layers": 2,
            "n_cross_layers": 3,
        },
        "training": {"pretrain_epochs": 15},
        "seed": 2222,
    }
    if domain is not None:
        cfg["training"]["phase1_pretrain_domain"] = domain
    return cfg


def test_fingerprint_distinguishes_target_vs_source():
    fp_target = fingerprint_from_config(_base_config("target"), "aki_v5_cross3.json")
    fp_source = fingerprint_from_config(_base_config("source"), "aki_v5_cross3.json")
    assert fp_target is not None and fp_source is not None
    assert fp_target != fp_source
    assert hash(fp_target) != hash(fp_source)


def test_fingerprint_default_matches_explicit_target():
    fp_default = fingerprint_from_config(_base_config(None), "aki_v5_cross3.json")
    fp_target = fingerprint_from_config(_base_config("target"), "aki_v5_cross3.json")
    assert fp_default == fp_target  # backward-compat: omitted key == target


def test_fingerprint_str_marks_source_pretrain():
    fp_source = fingerprint_from_config(_base_config("source"), "aki_v5_cross3.json")
    s = str(fp_source)
    assert "dom=source" in s


def test_fingerprint_str_silent_for_target_default():
    fp_target = fingerprint_from_config(_base_config(None), "aki_v5_cross3.json")
    assert "dom=" not in str(fp_target)
