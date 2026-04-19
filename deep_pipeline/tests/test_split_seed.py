"""Regression tests for the decoupled split_seed / training_seed logic.

Background: Historically, the top-level ``config["seed"]`` was passed to
``YAIBRuntime(seed=...)`` and then to ``StratifiedKFold(random_state=...)``
inside YAIB's ``preprocess_data``. This meant multi-seed runs silently landed
on different test folds. ``split_seed`` is now decoupled, defaults to 2222
(the canonical fold used by all historical single-seed runs), and can be
overridden explicitly if needed.
"""

from src.cli import _get_split_seed, _get_training_seed, _get_training_config


def test_legacy_seed_2222_unchanged():
    """Historical ``seed: 2222`` configs keep both seeds at 2222."""
    cfg = {"seed": 2222, "training": {}}
    assert _get_split_seed(cfg) == 2222
    assert _get_training_seed(cfg) == 2222


def test_new_multi_seed_preserves_canonical_fold():
    """``seed: 7777`` at top level now only changes training_seed; fold stays at 2222."""
    cfg = {"seed": 7777, "training": {}}
    assert _get_split_seed(cfg) == 2222, "split_seed must default to 2222"
    assert _get_training_seed(cfg) == 7777


def test_explicit_split_seed_override():
    """Explicit ``split_seed`` in config overrides default."""
    cfg = {"seed": 7777, "split_seed": 2222, "training": {}}
    assert _get_split_seed(cfg) == 2222
    assert _get_training_seed(cfg) == 7777

    cfg = {"seed": 42, "split_seed": 1234, "training": {}}
    assert _get_split_seed(cfg) == 1234
    assert _get_training_seed(cfg) == 42


def test_no_seeds_defaults():
    """Missing both seeds yields canonical defaults (split=2222, training=42)."""
    cfg = {"training": {}}
    assert _get_split_seed(cfg) == 2222
    assert _get_training_seed(cfg) == 42


def test_training_training_seed_overrides_top_level():
    """``training.training_seed`` beats top-level ``seed``."""
    cfg = {"seed": 42, "training": {"training_seed": 9999}}
    assert _get_split_seed(cfg) == 2222
    assert _get_training_seed(cfg) == 9999


def test_training_seed_falls_back_to_training_seed_key():
    """``training.seed`` beats top-level ``seed`` but loses to ``training.training_seed``."""
    cfg = {"seed": 42, "training": {"seed": 100}}
    assert _get_training_seed(cfg) == 100

    cfg = {"seed": 42, "training": {"seed": 100, "training_seed": 200}}
    assert _get_training_seed(cfg) == 200


def test_training_config_exposes_split_seed_and_resolved_training_seed():
    """``_get_training_config`` exposes both seeds for downstream callers."""
    cfg = {"seed": 7777, "training": {}}
    tc = _get_training_config(cfg)
    assert tc["split_seed"] == 2222
    assert tc["seed"] == 7777  # backward-compat: "seed" in training_cfg = training_seed
    assert tc["training_seed_resolved"] == 7777
