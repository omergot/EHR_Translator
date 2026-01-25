import argparse
from pathlib import Path

import joblib
import numpy as np


def _format_list(items, max_items=20):
    items = list(items)
    if len(items) <= max_items:
        return items
    return items[:max_items] + [f"...(+{len(items) - max_items} more)"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect linear_regression.pkl contents.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("/bigdata/omerg/Thesis/EHR_Translator/deep_pipeline/models/linear_regression.pkl"),
        help="Path to linear_regression.pkl",
    )
    parser.add_argument("--max-items", type=int, default=20, help="Max list items to print.")
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(f"File not found: {args.path}")

    obj = joblib.load(args.path)
    print(f"File: {args.path}")
    print(f"Type: {type(obj)}")

    if not isinstance(obj, dict):
        return

    print(f"Keys: {sorted(obj.keys())}")

    for key in sorted(obj.keys()):
        val = obj[key]
        print(f"\n[{key}] type={type(val)}")
        if isinstance(val, (list, tuple)):
            print(f"  len={len(val)}")
            print(f"  preview={_format_list(val, args.max_items)}")
        elif isinstance(val, np.ndarray):
            print(f"  shape={val.shape} dtype={val.dtype}")
            preview = _format_list(val.tolist(), args.max_items)
            print(f"  preview={preview}")
        else:
            print(f"  value={val}")


if __name__ == "__main__":
    main()
