#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import math


def compute_feature_ab(
    path_a: Path,
    path_b: Path,
    out_path: Path,
    chunksize: int = 200_000,
    missing_prefix: str = "MissingIndicator_",
):
    cols_a = pd.read_csv(path_a, nrows=0).columns.tolist()
    cols_b = pd.read_csv(path_b, nrows=0).columns.tolist()

    ignore = {"stay_id", "time"}
    missing_cols_a = {c for c in cols_a if c.startswith(missing_prefix)}
    missing_cols_b = {c for c in cols_b if c.startswith(missing_prefix)}
    ignore |= missing_cols_a
    ignore |= missing_cols_b

    common = [c for c in cols_a if c in cols_b and c not in ignore]
    if not common:
        raise ValueError("No common feature columns after filtering.")

    n_a = {c: 0 for c in common}
    n_b = {c: 0 for c in common}
    sum_a = {c: 0.0 for c in common}
    sum_b = {c: 0.0 for c in common}
    sum_a2 = {c: 0.0 for c in common}
    sum_b2 = {c: 0.0 for c in common}
    n_a_all = {c: 0 for c in common}
    n_b_all = {c: 0 for c in common}
    sum_a_all = {c: 0.0 for c in common}
    sum_b_all = {c: 0.0 for c in common}
    sum_a2_all = {c: 0.0 for c in common}
    sum_b2_all = {c: 0.0 for c in common}
    missing_a = {c: 0 for c in common}
    missing_b = {c: 0 for c in common}

    usecols_a = common + [c for c in missing_cols_a if c.replace(missing_prefix, "") in common]
    usecols_b = common + [c for c in missing_cols_b if c.replace(missing_prefix, "") in common]

    reader_a = pd.read_csv(path_a, usecols=usecols_a, chunksize=chunksize)
    for chunk_a in reader_a:
        a_vals = chunk_a.apply(pd.to_numeric, errors="coerce")
        for c in common:
            x = a_vals[c].to_numpy(dtype=float, copy=False)
            mask_all = ~np.isnan(x)
            if mask_all.any():
                x_all = x[mask_all]
                n_a_all[c] += x_all.size
                sum_a_all[c] += float(x_all.sum())
                sum_a2_all[c] += float(np.dot(x_all, x_all))
            mi_col = f"{missing_prefix}{c}"
            if mi_col in a_vals.columns:
                mi = a_vals[mi_col].to_numpy(dtype=float, copy=False)
                missing_a[c] += int((mi > 0.5).sum())
                mask_a = (~np.isnan(x)) & (mi <= 0.5)
            else:
                mask_a = ~np.isnan(x)
            if mask_a.any():
                x_a = x[mask_a]
                n_a[c] += x_a.size
                sum_a[c] += float(x_a.sum())
                sum_a2[c] += float(np.dot(x_a, x_a))

    reader_b = pd.read_csv(path_b, usecols=usecols_b, chunksize=chunksize)
    for chunk_b in reader_b:
        b_vals = chunk_b.apply(pd.to_numeric, errors="coerce")
        for c in common:
            y = b_vals[c].to_numpy(dtype=float, copy=False)
            mask_all = ~np.isnan(y)
            if mask_all.any():
                y_all = y[mask_all]
                n_b_all[c] += y_all.size
                sum_b_all[c] += float(y_all.sum())
                sum_b2_all[c] += float(np.dot(y_all, y_all))
            mi_col = f"{missing_prefix}{c}"
            if mi_col in b_vals.columns:
                mi = b_vals[mi_col].to_numpy(dtype=float, copy=False)
                missing_b[c] += int((mi > 0.5).sum())
                mask_b = (~np.isnan(y)) & (mi <= 0.5)
            else:
                mask_b = ~np.isnan(y)
            if mask_b.any():
                y_b = y[mask_b]
                n_b[c] += y_b.size
                sum_b[c] += float(y_b.sum())
                sum_b2[c] += float(np.dot(y_b, y_b))

    rows = []
    rows_all = []
    for c in common:
        total_a = n_a_all[c]
        total_b = n_b_all[c]
        miss_a = missing_a[c]
        miss_b = missing_b[c]
        miss_ratio_a = (miss_a / total_a) if total_a > 0 else float("nan")
        miss_ratio_b = (miss_b / total_b) if total_b > 0 else float("nan")
        if n_a[c] == 0 or n_b[c] == 0:
            rows.append(
                (
                    c,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    n_a[c],
                    n_b[c],
                    total_a,
                    total_b,
                    miss_a,
                    miss_b,
                    miss_ratio_a,
                    miss_ratio_b,
                )
            )
        else:
            mean_a = sum_a[c] / n_a[c]
            mean_b = sum_b[c] / n_b[c]
            var_a = (sum_a2[c] - (sum_a[c] ** 2) / n_a[c]) / n_a[c]
            var_b = (sum_b2[c] - (sum_b[c] ** 2) / n_b[c]) / n_b[c]
            std_a = math.sqrt(var_a) if var_a > 0 else 0.0
            std_b = math.sqrt(var_b) if var_b > 0 else 0.0
            if std_a == 0:
                a = 0.0
                b = mean_b
            else:
                a = std_b / std_a
                b = mean_b - a * mean_a
            rows.append(
                (
                    c,
                    a,
                    b,
                    mean_a,
                    mean_b,
                    n_a[c],
                    n_b[c],
                    total_a,
                    total_b,
                    miss_a,
                    miss_b,
                    miss_ratio_a,
                    miss_ratio_b,
                )
            )

        if n_a_all[c] == 0 or n_b_all[c] == 0:
            rows_all.append(
                (
                    c,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    n_a_all[c],
                    n_b_all[c],
                    total_a,
                    total_b,
                    miss_a,
                    miss_b,
                    miss_ratio_a,
                    miss_ratio_b,
                )
            )
        else:
            mean_a_all = sum_a_all[c] / n_a_all[c]
            mean_b_all = sum_b_all[c] / n_b_all[c]
            var_a_all = (sum_a2_all[c] - (sum_a_all[c] ** 2) / n_a_all[c]) / n_a_all[c]
            var_b_all = (sum_b2_all[c] - (sum_b_all[c] ** 2) / n_b_all[c]) / n_b_all[c]
            std_a_all = math.sqrt(var_a_all) if var_a_all > 0 else 0.0
            std_b_all = math.sqrt(var_b_all) if var_b_all > 0 else 0.0
            if std_a_all == 0:
                a_all = 0.0
                b_all = mean_b_all
            else:
                a_all = std_b_all / std_a_all
                b_all = mean_b_all - a_all * mean_a_all
            rows_all.append(
                (
                    c,
                    a_all,
                    b_all,
                    mean_a_all,
                    mean_b_all,
                    n_a_all[c],
                    n_b_all[c],
                    total_a,
                    total_b,
                    miss_a,
                    miss_b,
                    miss_ratio_a,
                    miss_ratio_b,
                )
            )

    columns = [
        "feature",
        "a",
        "b",
        "mean_a",
        "mean_b",
        "n_a",
        "n_b",
        "total_a",
        "total_b",
        "missing_a",
        "missing_b",
        "missing_ratio_a",
        "missing_ratio_b",
    ]
    out_df = pd.DataFrame(rows, columns=columns)
    out_all_df = pd.DataFrame(rows_all, columns=columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    out_all_df.to_csv(out_path.with_name(out_path.stem + "_all.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Compute per-feature affine mapping a,b from two CSVs.")
    ap.add_argument("--a", required=True, help="Path to source CSV (e.g., eicu features_full.csv)")
    ap.add_argument("--b", required=True, help="Path to target CSV (e.g., miiv features_full.csv)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk (default: 200000)")
    ap.add_argument("--missing-prefix", default="MissingIndicator_", help="Missing indicator prefix to ignore")
    args = ap.parse_args()

    compute_feature_ab(
        Path(args.a),
        Path(args.b),
        Path(args.out),
        chunksize=args.chunksize,
        missing_prefix=args.missing_prefix,
    )


if __name__ == "__main__":
    main()
