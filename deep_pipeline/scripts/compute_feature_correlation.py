#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd


def compute_correlations(
    path_a: Path,
    path_b: Path,
    out_path: Path,
    chunksize: int = 200_000,
    verbose: bool = True,
    out_stats_path: Path | None = None,
    bins: int = 200,
):
    # Read headers only
    cols_a = pd.read_csv(path_a, nrows=0).columns.tolist()
    cols_b = pd.read_csv(path_b, nrows=0).columns.tolist()

    ignore = {"stay_id", "time"}
    ignore |= {c for c in cols_a if c.startswith("MissingIndicator_")}
    ignore |= {c for c in cols_b if c.startswith("MissingIndicator_")}

    common = [c for c in cols_a if c in cols_b and c not in ignore]
    if not common:
        raise ValueError("No common feature columns after filtering.")

    # Accumulators for paired correlations
    n = {c: 0 for c in common}
    sum_x = {c: 0.0 for c in common}
    sum_y = {c: 0.0 for c in common}
    sum_x2 = {c: 0.0 for c in common}
    sum_y2 = {c: 0.0 for c in common}
    sum_xy = {c: 0.0 for c in common}

    # Accumulators for per-dataset stats
    n_a = {c: 0 for c in common}
    n_b = {c: 0 for c in common}
    sum_a = {c: 0.0 for c in common}
    sum_b = {c: 0.0 for c in common}
    sum_a2 = {c: 0.0 for c in common}
    sum_b2 = {c: 0.0 for c in common}

    # Optional pre-pass to get min/max per feature (for distribution metrics)
    edges = None
    if out_stats_path is not None:
        min_a = {c: float("inf") for c in common}
        max_a = {c: float("-inf") for c in common}
        min_b = {c: float("inf") for c in common}
        max_b = {c: float("-inf") for c in common}

        reader_a_minmax = pd.read_csv(path_a, usecols=common, chunksize=chunksize)
        for chunk_a in reader_a_minmax:
            a_vals = chunk_a.apply(pd.to_numeric, errors="coerce")
            for c in common:
                x = a_vals[c].to_numpy(dtype=float, copy=False)
                if np.isfinite(x).any():
                    min_a[c] = min(min_a[c], float(np.nanmin(x)))
                    max_a[c] = max(max_a[c], float(np.nanmax(x)))

        reader_b_minmax = pd.read_csv(path_b, usecols=common, chunksize=chunksize)
        for chunk_b in reader_b_minmax:
            b_vals = chunk_b.apply(pd.to_numeric, errors="coerce")
            for c in common:
                y = b_vals[c].to_numpy(dtype=float, copy=False)
                if np.isfinite(y).any():
                    min_b[c] = min(min_b[c], float(np.nanmin(y)))
                    max_b[c] = max(max_b[c], float(np.nanmax(y)))

        edges = {}
        for c in common:
            min_v = min(min_a[c], min_b[c])
            max_v = max(max_a[c], max_b[c])
            if not np.isfinite(min_v) or not np.isfinite(max_v) or min_v == max_v:
                edges[c] = None
            else:
                edges[c] = np.linspace(min_v, max_v, bins + 1)

    reader_a = pd.read_csv(path_a, usecols=common, chunksize=chunksize)
    reader_b = pd.read_csv(path_b, usecols=common, chunksize=chunksize)

    total_rows = 0
    chunk_idx = 0
    # Histograms for distribution metrics
    hist_a = {c: np.zeros(bins, dtype=np.float64) for c in common} if edges is not None else None
    hist_b = {c: np.zeros(bins, dtype=np.float64) for c in common} if edges is not None else None

    # Compute per-dataset stats on full CSVs (not paired)
    reader_a_stats = pd.read_csv(path_a, usecols=common, chunksize=chunksize)
    for chunk_a in reader_a_stats:
        a_vals = chunk_a.apply(pd.to_numeric, errors="coerce")
        for c in common:
            x = a_vals[c].to_numpy(dtype=float, copy=False)
            mask_a = ~np.isnan(x)
            if mask_a.any():
                x_a = x[mask_a]
                n_a[c] += x_a.size
                sum_a[c] += float(x_a.sum())
                sum_a2[c] += float(np.dot(x_a, x_a))
                if hist_a is not None and edges[c] is not None:
                    h, _ = np.histogram(x_a, bins=edges[c])
                    hist_a[c] += h

    reader_b_stats = pd.read_csv(path_b, usecols=common, chunksize=chunksize)
    for chunk_b in reader_b_stats:
        b_vals = chunk_b.apply(pd.to_numeric, errors="coerce")
        for c in common:
            y = b_vals[c].to_numpy(dtype=float, copy=False)
            mask_b = ~np.isnan(y)
            if mask_b.any():
                y_b = y[mask_b]
                n_b[c] += y_b.size
                sum_b[c] += float(y_b.sum())
                sum_b2[c] += float(np.dot(y_b, y_b))
                if hist_b is not None and edges[c] is not None:
                    h, _ = np.histogram(y_b, bins=edges[c])
                    hist_b[c] += h

    for chunk_a, chunk_b in zip(reader_a, reader_b):
        chunk_idx += 1
        # Align chunk lengths if last chunk differs
        if len(chunk_a) != len(chunk_b):
            min_len = min(len(chunk_a), len(chunk_b))
            chunk_a = chunk_a.iloc[:min_len]
            chunk_b = chunk_b.iloc[:min_len]

        total_rows += len(chunk_a)

        # Convert to float arrays
        # This will coerce non-numeric values to NaN
        a_vals = chunk_a.apply(pd.to_numeric, errors="coerce")
        b_vals = chunk_b.apply(pd.to_numeric, errors="coerce")

        for c in common:
            x = a_vals[c].to_numpy(dtype=float, copy=False)
            y = b_vals[c].to_numpy(dtype=float, copy=False)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.any():
                x_m = x[mask]
                y_m = y[mask]
                n[c] += x_m.size
                sum_x[c] += float(x_m.sum())
                sum_y[c] += float(y_m.sum())
                sum_x2[c] += float(np.dot(x_m, x_m))
                sum_y2[c] += float(np.dot(y_m, y_m))
                sum_xy[c] += float(np.dot(x_m, y_m))

        if verbose and chunk_idx % 10 == 0:
            print(f"Processed {chunk_idx} chunks, {total_rows} rows...", flush=True)

    # Build correlation results
    rows = []
    for c in common:
        n_c = n[c]
        if n_c < 2:
            corr = float("nan")
        else:
            num = sum_xy[c] - (sum_x[c] * sum_y[c]) / n_c
            den_x = sum_x2[c] - (sum_x[c] ** 2) / n_c
            den_y = sum_y2[c] - (sum_y[c] ** 2) / n_c
            denom = math.sqrt(den_x * den_y) if den_x > 0 and den_y > 0 else 0.0
            corr = num / denom if denom > 0 else float("nan")
        rows.append((c, corr, n_c))

    out_df = pd.DataFrame(rows, columns=["feature", "corr", "n_used"]).sort_values("corr", ascending=False)
    out_df.to_csv(out_path, index=False)

    if verbose:
        print(f"Saved {len(out_df)} correlations to {out_path}")

    if out_stats_path is not None:
        stats_rows = []
        for c in common:
            if n_a[c] > 0:
                mean_a = sum_a[c] / n_a[c]
                var_a = (sum_a2[c] - (sum_a[c] ** 2) / n_a[c]) / n_a[c]
                std_a = math.sqrt(var_a) if var_a > 0 else 0.0
            else:
                mean_a = float("nan")
                std_a = float("nan")

            if n_b[c] > 0:
                mean_b = sum_b[c] / n_b[c]
                var_b = (sum_b2[c] - (sum_b[c] ** 2) / n_b[c]) / n_b[c]
                std_b = math.sqrt(var_b) if var_b > 0 else 0.0
            else:
                mean_b = float("nan")
                std_b = float("nan")

            # Distribution metrics from histograms
            if edges is not None and edges[c] is not None and n_a[c] > 0 and n_b[c] > 0:
                ea = edges[c]
                ha = hist_a[c]
                hb = hist_b[c]
                cdf_a = np.cumsum(ha) / ha.sum()
                cdf_b = np.cumsum(hb) / hb.sum()
                ks_stat = float(np.max(np.abs(cdf_a - cdf_b)))

                # Approximate KS p-value
                en = math.sqrt((n_a[c] * n_b[c]) / (n_a[c] + n_b[c]))
                p_value = float(min(1.0, 2.0 * math.exp(-2.0 * (en * ks_stat) ** 2)))

                bin_widths = np.diff(ea)
                wasserstein = float(np.sum(np.abs(cdf_a - cdf_b) * bin_widths))

                def quantile_from_hist(q):
                    target = q * ha.sum()
                    cum = np.cumsum(ha)
                    idx = int(np.searchsorted(cum, target, side="left"))
                    idx = min(max(idx, 0), len(bin_widths) - 1)
                    prev = cum[idx - 1] if idx > 0 else 0.0
                    if ha[idx] == 0:
                        return float(ea[idx])
                    frac = (target - prev) / ha[idx]
                    return float(ea[idx] + frac * bin_widths[idx])

                def quantile_from_hist_b(q):
                    target = q * hb.sum()
                    cum = np.cumsum(hb)
                    idx = int(np.searchsorted(cum, target, side="left"))
                    idx = min(max(idx, 0), len(bin_widths) - 1)
                    prev = cum[idx - 1] if idx > 0 else 0.0
                    if hb[idx] == 0:
                        return float(ea[idx])
                    frac = (target - prev) / hb[idx]
                    return float(ea[idx] + frac * bin_widths[idx])

                q05_a = quantile_from_hist(0.05)
                q25_a = quantile_from_hist(0.25)
                q50_a = quantile_from_hist(0.50)
                q75_a = quantile_from_hist(0.75)
                q95_a = quantile_from_hist(0.95)

                q05_b = quantile_from_hist_b(0.05)
                q25_b = quantile_from_hist_b(0.25)
                q50_b = quantile_from_hist_b(0.50)
                q75_b = quantile_from_hist_b(0.75)
                q95_b = quantile_from_hist_b(0.95)
            else:
                ks_stat = float("nan")
                p_value = float("nan")
                wasserstein = float("nan")
                q05_a = q25_a = q50_a = q75_a = q95_a = float("nan")
                q05_b = q25_b = q50_b = q75_b = q95_b = float("nan")

            stats_rows.append(
                (
                    c,
                    mean_a,
                    std_a,
                    n_a[c],
                    q05_a,
                    q25_a,
                    q50_a,
                    q75_a,
                    q95_a,
                    mean_b,
                    std_b,
                    n_b[c],
                    q05_b,
                    q25_b,
                    q50_b,
                    q75_b,
                    q95_b,
                    ks_stat,
                    p_value,
                    wasserstein,
                )
            )

        stats_df = pd.DataFrame(
            stats_rows,
            columns=[
                "feature",
                "mean_a",
                "std_a",
                "n_a",
                "q05_a",
                "q25_a",
                "q50_a",
                "q75_a",
                "q95_a",
                "mean_b",
                "std_b",
                "n_b",
                "q05_b",
                "q25_b",
                "q50_b",
                "q75_b",
                "q95_b",
                "ks_stat",
                "ks_pvalue",
                "wasserstein",
            ],
        )
        stats_df.to_csv(out_stats_path, index=False)

        mean_corr = stats_df[["mean_a", "mean_b"]].corr().iloc[0, 1]
        std_corr = stats_df[["std_a", "std_b"]].corr().iloc[0, 1]

        if verbose:
            print(f"Saved per-feature stats to {out_stats_path}")
            print(f"Correlation of feature means (a vs b): {mean_corr:.6f}")
            print(f"Correlation of feature stds (a vs b): {std_corr:.6f}")


def main():
    ap = argparse.ArgumentParser(description="Compute column-wise Pearson correlation between two CSVs with matching columns.")
    ap.add_argument("--a", required=True, help="Path to first CSV")
    ap.add_argument("--b", required=True, help="Path to second CSV")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk (default: 200000)")
    ap.add_argument("--out-stats", help="Output CSV for per-feature means/stds (optional)")
    ap.add_argument("--bins", type=int, default=200, help="Histogram bins for distribution metrics (default: 200)")
    ap.add_argument("--quiet", action="store_true", help="Disable progress prints")
    args = ap.parse_args()

    compute_correlations(
        Path(args.a),
        Path(args.b),
        Path(args.out),
        chunksize=args.chunksize,
        verbose=not args.quiet,
        out_stats_path=Path(args.out_stats) if args.out_stats else None,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
