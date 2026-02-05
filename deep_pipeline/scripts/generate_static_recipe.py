import argparse
import json
import logging
import sys
from pathlib import Path

import gin
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment, DataSplit
from icu_benchmarks.data.preprocessor import PolarsClassificationPreprocessor
from icu_benchmarks.data.split_process_data import (
    check_sanitize_data,
    make_single_split_polars,
    make_train_val_polars,
)

from src.adapters.yaib import import_yaib_run_module, _is_yaib_run_registered, _pushd, _find_yaib_root


def _get_gin_param(name: str, default):
    try:
        return gin.query_parameter(name)
    except Exception:
        return default


def _wrap_load_gin_config(config_path: Path) -> None:
    yaib_root = _find_yaib_root(str(config_path))
    gin.add_config_file_search_path(yaib_root)
    gin.add_config_file_search_path(str(Path(yaib_root) / "configs"))
    with _pushd(yaib_root):
        gin.parse_config_file(str(config_path))


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static-only recipe cache for YAIB.")
    parser.add_argument("--config", required=True, help="Path to deep_pipeline JSON config.")
    parser.add_argument(
        "--output",
        default="",
        help="Output recipe cache path. Defaults to <data_dir>/preproc/static_recipe.",
    )
    parser.add_argument(
        "--runmode",
        default="classification",
        choices=["classification", "regression"],
        help="Run mode for split logic.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = _load_config(Path(args.config))
    data_dir = Path(config["data_dir"])
    file_names = config["file_names"]
    vars_cfg = dict(config["vars"])

    if "STATIC" not in file_names:
        raise ValueError("file_names.STATIC is required for static recipe generation.")
    if "OUTCOME" not in file_names:
        raise ValueError("file_names.OUTCOME is required for split generation.")
    if "STATIC" not in vars_cfg:
        raise ValueError("vars.STATIC is required for static recipe generation.")

    gin.clear_config()
    if not _is_yaib_run_registered():
        import_yaib_run_module()
    _wrap_load_gin_config(Path(config["task_config"]))
    if config.get("model_config"):
        _wrap_load_gin_config(Path(config["model_config"]))
    baseline_train = Path(config["baseline_model_dir"]) / "train_config.gin"
    if baseline_train.exists():
        _wrap_load_gin_config(baseline_train)

    cv_repetitions = _get_gin_param("execute_repeated_cv.cv_repetitions", 5)
    cv_folds = _get_gin_param("execute_repeated_cv.cv_folds", 5)
    repetition_index = _get_gin_param("execute_repeated_cv.repetition_index", 0)
    fold_index = _get_gin_param("execute_repeated_cv.fold_index", 0)
    train_size = _get_gin_param("execute_repeated_cv.train_size", None)
    complete_train = _get_gin_param("execute_repeated_cv.complete_train", False)
    scaling = _get_gin_param("base_classification_preprocessor.scaling", True)
    use_100_features = _get_gin_param("base_classification_preprocessor.use_100_features", False)
    use_post_imputation_stats = _get_gin_param(
        "base_classification_preprocessor.use_post_imputation_stats", False
    )

    runmode = RunMode.classification if args.runmode == "classification" else RunMode.regression

    logging.info("Loading static/outcome data from %s", data_dir)
    data = {
        DataSegment.static: pl.read_parquet(data_dir / file_names[DataSegment.static]),
        DataSegment.outcome: pl.read_parquet(data_dir / file_names[DataSegment.outcome]),
    }

    data = check_sanitize_data(data, vars_cfg)

    if complete_train:
        splits = make_train_val_polars(
            data,
            vars_cfg,
            train_size=train_size,
            seed=config.get("seed", 42),
            debug=config.get("debug", False),
            runmode=runmode,
        )
    else:
        splits = make_single_split_polars(
            data,
            vars_cfg,
            cv_repetitions=cv_repetitions,
            repetition_index=repetition_index,
            cv_folds=cv_folds,
            fold_index=fold_index,
            train_size=train_size,
            seed=config.get("seed", 42),
            debug=config.get("debug", False),
            runmode=runmode,
        )

    output_path = Path(args.output) if args.output else data_dir / "preproc" / "static_recipe"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing static recipe to %s", output_path)

    preprocessor = PolarsClassificationPreprocessor(
        scaling=scaling,
        use_static_features=True,
        save_cache=output_path,
        use_100_features=use_100_features,
        use_post_imputation_stats=use_post_imputation_stats,
    )
    preprocessor._process_static(splits, vars_cfg)
    logging.info("Static recipe cache saved to %s", output_path)


if __name__ == "__main__":
    main()
