"""Unit tests for resume-checkpoint integrity checks.

Background: on 2026-04-20 a silent contamination bug was found where a `_v2`
requeue silently resumed from a pre-fix `latest_checkpoint.pt`, leaking ~80%
of the v2 test fold via memorized weights. The fix adds integrity tags
(split_seed, training_seed, config_fingerprint) to every resume checkpoint
and validates them on load.

These tests cover the pure-Python validation helper and the fingerprint
function — they do NOT spin up a full trainer.
"""

import pytest

from src.core.train import compute_config_fingerprint, validate_resume_checkpoint


# ----------------------------- fingerprint -----------------------------


def test_fingerprint_stable_for_identical_config():
    """Same training_config + translator_config → same fingerprint (deterministic)."""
    tc = {"lr": 1e-4, "epochs": 30, "batch_size": 16}
    xc = {"type": "retrieval", "d_model": 128, "d_latent": 64, "n_cross_layers": 3}
    assert compute_config_fingerprint(tc, xc) == compute_config_fingerprint(tc, xc)


def test_fingerprint_sensitive_to_arch_change():
    tc = {"lr": 1e-4, "epochs": 30}
    a = compute_config_fingerprint(tc, {"type": "retrieval", "d_model": 128})
    b = compute_config_fingerprint(tc, {"type": "retrieval", "d_model": 256})
    assert a != b


def test_fingerprint_sensitive_to_loss_weight_change():
    xc = {"type": "retrieval", "d_model": 128}
    a = compute_config_fingerprint({"lambda_recon": 0.1}, xc)
    b = compute_config_fingerprint({"lambda_recon": 0.0}, xc)  # C5 ablation
    assert a != b


def test_fingerprint_insensitive_to_seed_fields():
    """Seeds live in dedicated tags, NOT in the fingerprint."""
    xc = {"type": "retrieval"}
    a = compute_config_fingerprint({"lr": 1e-4, "seed": 42, "split_seed": 2222}, xc)
    b = compute_config_fingerprint({"lr": 1e-4, "seed": 7777, "split_seed": 2222}, xc)
    assert a == b, "seed changes must NOT alter the config fingerprint"


# ----------------------------- validation ------------------------------


def _make_ckpt(split=2222, training=42, fp="deadbeefdeadbeef"):
    return {
        "epoch": 5,
        "split_seed": split,
        "training_seed": training,
        "config_fingerprint": fp,
    }


def test_valid_tagged_checkpoint_passes(tmp_path):
    ckpt = _make_ckpt(split=2222, training=42, fp="abc")
    # Should not raise.
    validate_resume_checkpoint(
        ckpt,
        expected_split_seed=2222,
        expected_training_seed=42,
        expected_fingerprint="abc",
        resume_path=tmp_path / "latest_checkpoint.pt",
    )


def test_split_seed_mismatch_raises(tmp_path):
    """The exact pattern that caused the mort_c2 contamination."""
    ckpt = _make_ckpt(split=42, training=42, fp="abc")  # pre-fix: split tied to seed
    with pytest.raises(RuntimeError, match=r"split_seed mismatch"):
        validate_resume_checkpoint(
            ckpt,
            expected_split_seed=2222,
            expected_training_seed=42,
            expected_fingerprint="abc",
            resume_path=tmp_path / "latest_checkpoint.pt",
        )


def test_training_seed_mismatch_raises(tmp_path):
    ckpt = _make_ckpt(split=2222, training=100, fp="abc")
    with pytest.raises(RuntimeError, match=r"training_seed mismatch"):
        validate_resume_checkpoint(
            ckpt, 2222, 42, "abc", tmp_path / "x.pt",
        )


def test_fingerprint_mismatch_raises(tmp_path):
    ckpt = _make_ckpt(fp="oldhash")
    with pytest.raises(RuntimeError, match=r"config_fingerprint mismatch"):
        validate_resume_checkpoint(
            ckpt, 2222, 42, "newhash", tmp_path / "x.pt",
        )


def test_untagged_checkpoint_raises(tmp_path):
    """Pre-fix checkpoints with no integrity tags are rejected by default."""
    ckpt = {"epoch": 10, "translator_state_dict": {}}  # no split_seed / fp
    with pytest.raises(RuntimeError, match=r"NO integrity tags"):
        validate_resume_checkpoint(
            ckpt, 2222, 42, "abc", tmp_path / "x.pt",
        )


def test_force_resume_suppresses_mismatch(tmp_path, caplog):
    import logging
    ckpt = _make_ckpt(split=42)  # mismatched
    caplog.set_level(logging.WARNING)
    # Should NOT raise when force_resume=True
    validate_resume_checkpoint(
        ckpt,
        expected_split_seed=2222,
        expected_training_seed=42,
        expected_fingerprint="abc",
        resume_path=tmp_path / "x.pt",
        force_resume=True,
    )
    # Must emit a loud warning.
    assert any("force-resume" in rec.message.lower() for rec in caplog.records), (
        "force_resume must log a warning"
    )


def test_force_resume_suppresses_untagged(tmp_path, caplog):
    """Even untagged (pre-fix) checkpoints can be consumed with --force-resume."""
    import logging
    ckpt = {"epoch": 10}
    caplog.set_level(logging.WARNING)
    validate_resume_checkpoint(
        ckpt, 2222, 42, "abc", tmp_path / "x.pt", force_resume=True,
    )
    assert any("force-resume" in rec.message.lower() for rec in caplog.records)


def test_error_message_includes_remedy(tmp_path):
    ckpt = _make_ckpt(split=42)
    with pytest.raises(RuntimeError) as exc_info:
        validate_resume_checkpoint(ckpt, 2222, 42, "abc", tmp_path / "latest_checkpoint.pt")
    msg = str(exc_info.value)
    assert "delete" in msg.lower()
    assert "--force-resume" in msg
    assert "latest_checkpoint.pt" in msg
