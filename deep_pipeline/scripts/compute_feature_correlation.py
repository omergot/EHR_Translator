#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pickle


def _is_parquet(path: Path) -> bool:
    return path.suffix.lower() in {".parquet", ".pq"}


def _read_columns(path: Path) -> list[str]:
    if _is_parquet(path):
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError("pyarrow is required to read parquet files.") from exc
        return pq.ParquetFile(path).schema.names
    return pd.read_csv(path, nrows=0).columns.tolist()


def _iter_table(path: Path, columns: list[str], chunksize: int):
    if _is_parquet(path):
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError("pyarrow is required to read parquet files.") from exc
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=columns, batch_size=chunksize):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, usecols=columns, chunksize=chunksize)


def _load_recipe_scaler_stats(recipe_path: Path) -> tuple[list[str], dict[str, float], dict[str, float]]:
    with recipe_path.open("rb") as f:
        recipe = pickle.load(f)
    scaler = None
    for step in getattr(recipe, "steps", []):
        transformer = getattr(step, "sklearn_transformer", None)
        if transformer is not None and transformer.__class__.__name__ == "StandardScaler":
            scaler = transformer
            break
    if scaler is None:
        raise RuntimeError("Could not find StandardScaler in recipe steps.")
    if hasattr(scaler, "feature_names_in_"):
        cols = list(scaler.feature_names_in_)
    else:
        cols = list(getattr(step, "columns", []))
    means = dict(zip(cols, scaler.mean_, strict=False))
    scales = dict(zip(cols, scaler.scale_, strict=False))
    return cols, means, scales


def _compute_temporal_delta(
    path: Path,
    columns: list[str],
    chunksize: int,
    means: dict[str, float],
    scales: dict[str, float],
    apply_denorm: bool = True,
) -> dict[str, float]:
    path_cols = set(_read_columns(path))
    if "stay_id" not in path_cols or "time" not in path_cols:
        raise RuntimeError(f"{path} must contain stay_id and time for temporal smoothness.")
    cols = [c for c in columns if c in path_cols]
    if not cols:
        return {}

    # Assumes rows are ordered by (stay_id, time) in the input file; this is true for our
    # cohort dyn.parquet outputs and avoids holding state for every stay_id.
    sum_abs = np.zeros(len(cols), dtype=np.float64)
    count = np.zeros(len(cols), dtype=np.float64)
    prev_sid: int | None = None
    prev_vals: np.ndarray | None = None

    mean_vec = np.array([means.get(c, 0.0) for c in cols], dtype=np.float64)
    scale_vec = np.array([scales.get(c, 1.0) for c in cols], dtype=np.float64)

    for chunk in _iter_table(path, ["stay_id", "time", *cols], chunksize):
        sid = pd.to_numeric(chunk["stay_id"], errors="coerce").to_numpy(dtype=np.int64, copy=False)
        vals = chunk[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=False)
        if apply_denorm:
            vals = vals * scale_vec + mean_vec

        diffs = np.full_like(vals, np.nan)
        if sid.size >= 2:
            same = sid[1:] == sid[:-1]
            diffs[1:][same] = vals[1:][same] - vals[:-1][same]
        if prev_sid is not None and sid.size and sid[0] == prev_sid and prev_vals is not None:
            diffs[0] = vals[0] - prev_vals

        abs_diff = np.abs(diffs)
        sum_abs += np.nansum(abs_diff, axis=0)
        count += np.sum(~np.isnan(abs_diff), axis=0)

        if sid.size:
            prev_sid = int(sid[-1])
            prev_vals = vals[-1].copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = sum_abs / count
    return dict(zip(cols, delta, strict=False))


def _load_patient_series(
    path: Path,
    stay_id: int,
    feature: str,
    means: dict[str, float],
    scales: dict[str, float],
    apply_denorm: bool = True,
) -> pd.DataFrame:
    if _is_parquet(path):
        try:
            import pyarrow.dataset as ds
        except Exception as exc:
            raise RuntimeError("pyarrow is required to read parquet files.") from exc
        dataset = ds.dataset(path)
        table = dataset.to_table(filter=ds.field("stay_id") == stay_id, columns=["stay_id", "time", feature])
        df = table.to_pandas()
    else:
        rows = []
        for chunk in pd.read_csv(path, usecols=["stay_id", "time", feature], chunksize=200_000):
            rows.append(chunk[chunk["stay_id"] == stay_id])
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stay_id", "time", feature])

    if df.empty:
        return df
    df = df.sort_values("time")
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    if apply_denorm and feature in means and feature in scales:
        df[feature] = df[feature] * scales[feature] + means[feature]
    return df


def _load_patient_series_multi(
    path: Path,
    stay_ids: list[int],
    feature: str,
    means: dict[str, float],
    scales: dict[str, float],
    apply_denorm: bool = True,
) -> dict[int, pd.DataFrame]:
    if not stay_ids:
        return {}
    stay_set = set(stay_ids)
    if _is_parquet(path):
        try:
            import pyarrow.dataset as ds
        except Exception as exc:
            raise RuntimeError("pyarrow is required to read parquet files.") from exc
        dataset = ds.dataset(path)
        table = dataset.to_table(
            filter=ds.field("stay_id").isin(list(stay_set)),
            columns=["stay_id", "time", feature],
        )
        df = table.to_pandas()
    else:
        rows = []
        for chunk in pd.read_csv(path, usecols=["stay_id", "time", feature], chunksize=200_000):
            rows.append(chunk[chunk["stay_id"].isin(stay_set)])
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stay_id", "time", feature])

    if df.empty:
        return {}
    df = df.sort_values(["stay_id", "time"])
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    if apply_denorm and feature in means and feature in scales:
        df[feature] = df[feature] * scales[feature] + means[feature]

    out: dict[int, pd.DataFrame] = {}
    for sid, group in df.groupby("stay_id"):
        out[int(sid)] = group[["stay_id", "time", feature]].copy()
    return out


def _sample_stay_ids(path: Path, k: int, seed: int, chunksize: int) -> list[int]:
    rng = np.random.default_rng(seed)
    seen: set[int] = set()
    sample: list[int] = []
    for chunk in _iter_table(path, ["stay_id"], chunksize):
        ids = pd.to_numeric(chunk["stay_id"], errors="coerce").dropna().astype(int).unique()
        for sid in ids:
            if sid in seen:
                continue
            seen.add(int(sid))
            if len(sample) < k:
                sample.append(int(sid))
            else:
                j = int(rng.integers(0, len(seen)))
                if j < k:
                    sample[j] = int(sid)
    return sample


def _collect_stay_ids(path: Path, chunksize: int) -> set[int]:
    stay_ids: set[int] = set()
    for chunk in _iter_table(path, ["stay_id"], chunksize):
        ids = pd.to_numeric(chunk["stay_id"], errors="coerce").dropna().astype(int).unique()
        stay_ids.update(int(x) for x in ids)
    return stay_ids


def run_smoothness(
    source_path: Path,
    translated_path: Path,
    recipe_path: Path,
    out_dir: Path,
    *,
    chunksize: int = 200_000,
    stay_id: int | None = None,
    feature: str | None = None,
    samples: int = 5,
    seed: int = 2222,
    source_space: str = "normalized",
    write_delta_csv: bool = False,
    verbose: bool = True,
) -> list[int]:
    dyn_cols, means, scales = _load_recipe_scaler_stats(recipe_path)
    if source_space not in {"normalized", "physical"}:
        raise ValueError("--smoothness-source-space must be 'normalized' or 'physical'.")
    denorm_source = source_space == "normalized"

    out_dir.mkdir(parents=True, exist_ok=True)

    if write_delta_csv:
        delta_source = _compute_temporal_delta(source_path, dyn_cols, chunksize, means, scales, apply_denorm=denorm_source)
        delta_trans = _compute_temporal_delta(translated_path, dyn_cols, chunksize, means, scales, apply_denorm=True)
        delta_rows = []
        for c in dyn_cols:
            ds = float(delta_source.get(c, float("nan")))
            dt = float(delta_trans.get(c, float("nan")))
            ratio = dt / ds if np.isfinite(ds) and ds != 0 and np.isfinite(dt) else float("nan")
            delta_rows.append((c, ds, dt, ratio))
        pd.DataFrame(delta_rows, columns=["feature", "delta_source", "delta_trans", "delta_ratio"]).to_csv(
            out_dir / "smoothness_delta.csv",
            index=False,
        )

    if stay_id is not None:
        stay_ids = [int(stay_id)]
    else:
        try:
            src_ids = _collect_stay_ids(source_path, chunksize)
            trans_ids = _collect_stay_ids(translated_path, chunksize)
            common_ids = list(src_ids & trans_ids)
        except Exception:
            common_ids = []

        rng = np.random.default_rng(seed)
        if common_ids:
            k = min(samples, len(common_ids))
            stay_ids = [int(x) for x in rng.choice(common_ids, size=k, replace=False)]
        else:
            stay_ids = _sample_stay_ids(translated_path, samples, seed, chunksize)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors

    def _maybe_denorm_df(df: pd.DataFrame, cols: list[str], do_denorm: bool) -> pd.DataFrame:
        if not do_denorm or df.empty:
            return df
        present = [c for c in cols if c in df.columns and c in means and c in scales]
        if not present:
            return df
        df = df.copy()
        for c in present:
            df[c] = pd.to_numeric(df[c], errors="coerce") * scales[c] + means[c]
        return df

    def _df_to_series_dict(df: pd.DataFrame, col: str) -> dict[int, pd.DataFrame]:
        out: dict[int, pd.DataFrame] = {}
        if df.empty or col not in df.columns:
            return out
        for sid, group in df[["stay_id", "time", col]].groupby("stay_id"):
            out[int(sid)] = group.sort_values("time").copy()
        return out

    src_all: pd.DataFrame | None = None
    trans_all: pd.DataFrame | None = None
    if _is_parquet(source_path) and _is_parquet(translated_path):
        try:
            import pyarrow.dataset as ds
        except Exception as exc:
            raise RuntimeError("pyarrow is required to read parquet files.") from exc
        src_ds = ds.dataset(source_path)
        trans_ds = ds.dataset(translated_path)
        src_cols = [c for c in dyn_cols if c in src_ds.schema.names]
        trans_cols = [c for c in dyn_cols if c in trans_ds.schema.names]
        src_table = src_ds.to_table(filter=ds.field("stay_id").isin(stay_ids), columns=["stay_id", "time", *src_cols])
        trans_table = trans_ds.to_table(filter=ds.field("stay_id").isin(stay_ids), columns=["stay_id", "time", *trans_cols])
        src_all = _maybe_denorm_df(src_table.to_pandas().sort_values(["stay_id", "time"]), src_cols, denorm_source)
        trans_all = _maybe_denorm_df(trans_table.to_pandas().sort_values(["stay_id", "time"]), trans_cols, True)

    for col in dyn_cols:
        if feature is not None and col != feature:
            continue

        if src_all is not None and trans_all is not None:
            src_series = _df_to_series_dict(src_all, col)
            trans_series = _df_to_series_dict(trans_all, col)
        else:
            src_series = _load_patient_series_multi(source_path, stay_ids, col, means, scales, apply_denorm=denorm_source)
            trans_series = _load_patient_series_multi(translated_path, stay_ids, col, means, scales, apply_denorm=True)
        if not src_series and not trans_series:
            continue

        # Export the per-hour samples used for the plot.
        rows = []
        for sid in stay_ids:
            s_df = src_series.get(sid)
            t_df = trans_series.get(sid)
            s = s_df[["time", col]].rename(columns={col: "source"}) if s_df is not None else pd.DataFrame(columns=["time", "source"])
            t = t_df[["time", col]].rename(columns={col: "translated"}) if t_df is not None else pd.DataFrame(columns=["time", "translated"])
            merged = pd.merge(s, t, on="time", how="outer").sort_values("time")
            merged.insert(0, "stay_id", sid)
            rows.append(merged)

        samples_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stay_id", "time", "source", "translated"])
        samples_df.to_csv(out_dir / f"smoothness_{col}.csv", index=False)

        plt.figure(figsize=(10, 4))
        for idx, sid in enumerate(stay_ids):
            color = colors[idx % len(colors)]
            s_df = src_series.get(sid)
            t_df = trans_series.get(sid)
            if t_df is not None and not t_df.empty:
                plt.plot(
                    t_df["time"],
                    t_df[col],
                    color=color,
                    linestyle="-",
                    linewidth=1.2,
                    marker="o",
                    markersize=2.0,
                    alpha=0.7,
                    label=f"{sid} translated",
                )
            if s_df is not None and not s_df.empty:
                plt.plot(
                    s_df["time"],
                    s_df[col],
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    marker="x",
                    markersize=2.5,
                    alpha=1.0,
                    zorder=3,
                    label=f"{sid} source",
                )

        plt.xlabel("time")
        plt.ylabel(col)
        plt.title(f"{col}: {len(stay_ids)} stays")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"smoothness_{col}.png")
        plt.close()

    (out_dir / "smoothness_stay_ids.txt").write_text("\n".join(str(x) for x in stay_ids) + "\n")
    if verbose:
        print(f"Saved smoothness plots and samples to {out_dir} (stays={stay_ids})")
    return stay_ids


def _corr_matrix_from_path(path: Path, columns: list[str], chunksize: int) -> np.ndarray:
    size = len(columns)
    n_pair = np.zeros((size, size), dtype=np.float64)
    sum_x = np.zeros((size, size), dtype=np.float64)
    sum_x2 = np.zeros((size, size), dtype=np.float64)
    sum_xy = np.zeros((size, size), dtype=np.float64)

    for chunk in _iter_table(path, columns, chunksize):
        vals = chunk.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=False)
        mask = ~np.isnan(vals)
        x = np.nan_to_num(vals, nan=0.0)
        m = mask.astype(np.float64)

        n_pair += m.T @ m
        sum_x += x.T @ m
        sum_x2 += (x * x).T @ m
        sum_xy += x.T @ x

    with np.errstate(divide="ignore", invalid="ignore"):
        num = sum_xy - (sum_x * sum_x.T) / n_pair
        den_x = sum_x2 - (sum_x ** 2) / n_pair
        den_y = sum_x2.T - (sum_x.T ** 2) / n_pair
        denom = np.sqrt(den_x * den_y)
        corr = num / denom

    corr[(n_pair < 2) | (denom <= 0)] = np.nan
    return corr


def compute_correlations(
    path_a: Path,
    path_b: Path,
    out_path: Path,
    chunksize: int = 200_000,
    verbose: bool = True,
    out_stats_path: Path | None = None,
    bins: int = 200,
    path_t: Path | None = None,
    out_corr_delta_path: Path | None = None,
    smoothness_recipe_path: Path | None = None,
    smoothness_plot_path: Path | None = None,
    smoothness_stay_id: int | None = None,
    smoothness_feature: str | None = None,
    smoothness_samples: int = 5,
    smoothness_seed: int = 2222,
    smoothness_source_path: Path | None = None,
    smoothness_source_space: str = "normalized",
):
    # Read headers only
    cols_a = _read_columns(path_a)
    cols_b = _read_columns(path_b)
    cols_t = _read_columns(path_t) if path_t is not None else None

    ignore = {"stay_id", "time"}
    ignore |= {c for c in cols_a if c.startswith("MissingIndicator_")}
    ignore |= {c for c in cols_b if c.startswith("MissingIndicator_")}
    if cols_t is not None:
        ignore |= {c for c in cols_t if c.startswith("MissingIndicator_")}

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
    n_t = {c: 0 for c in common}
    sum_a = {c: 0.0 for c in common}
    sum_b = {c: 0.0 for c in common}
    sum_t = {c: 0.0 for c in common}
    sum_a2 = {c: 0.0 for c in common}
    sum_b2 = {c: 0.0 for c in common}
    sum_t2 = {c: 0.0 for c in common}

    # Optional pre-pass to get min/max per feature (for distribution metrics)
    edges = None
    if out_stats_path is not None:
        min_a = {c: float("inf") for c in common}
        max_a = {c: float("-inf") for c in common}
        min_b = {c: float("inf") for c in common}
        max_b = {c: float("-inf") for c in common}
        min_t = {c: float("inf") for c in common}
        max_t = {c: float("-inf") for c in common}

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

        if path_t is not None:
            common_t = [c for c in common if c in cols_t]
            for chunk_t in _iter_table(path_t, common_t, chunksize):
                t_vals = chunk_t.apply(pd.to_numeric, errors="coerce")
                for c in common_t:
                    z = t_vals[c].to_numpy(dtype=float, copy=False)
                    if np.isfinite(z).any():
                        min_t[c] = min(min_t[c], float(np.nanmin(z)))
                        max_t[c] = max(max_t[c], float(np.nanmax(z)))

        edges = {}
        for c in common:
            min_v = min(min_a[c], min_b[c])
            max_v = max(max_a[c], max_b[c])
            if path_t is not None and np.isfinite(min_t[c]) and np.isfinite(max_t[c]):
                min_v = min(min_v, min_t[c])
                max_v = max(max_v, max_t[c])
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
    hist_t = {c: np.zeros(bins, dtype=np.float64) for c in common} if edges is not None else None

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

    if out_stats_path is not None and path_t is not None:
        common_t = [c for c in common if c in cols_t]
        for chunk_t in _iter_table(path_t, common_t, chunksize):
            t_vals = chunk_t.apply(pd.to_numeric, errors="coerce")
            for c in common_t:
                z = t_vals[c].to_numpy(dtype=float, copy=False)
                mask_t = ~np.isnan(z)
                if mask_t.any():
                    z_t = z[mask_t]
                    n_t[c] += z_t.size
                    sum_t[c] += float(z_t.sum())
                    sum_t2[c] += float(np.dot(z_t, z_t))
                    if hist_t is not None and edges[c] is not None:
                        h, _ = np.histogram(z_t, bins=edges[c])
                        hist_t[c] += h

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
        delta_source = None
        delta_trans = None
        if smoothness_recipe_path is not None and path_t is not None:
            src_path = smoothness_source_path or path_b
            dyn_cols, means, scales = _load_recipe_scaler_stats(smoothness_recipe_path)
            if smoothness_source_space not in {"normalized", "physical"}:
                raise ValueError("--smoothness-source-space must be 'normalized' or 'physical'.")
            denorm_source = smoothness_source_space == "normalized"
            delta_source = _compute_temporal_delta(src_path, dyn_cols, chunksize, means, scales, apply_denorm=denorm_source)
            delta_trans = _compute_temporal_delta(path_t, dyn_cols, chunksize, means, scales, apply_denorm=True)

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

            if n_t[c] > 0:
                mean_t = sum_t[c] / n_t[c]
                var_t = (sum_t2[c] - (sum_t[c] ** 2) / n_t[c]) / n_t[c]
                std_t = math.sqrt(var_t) if var_t > 0 else 0.0
            else:
                mean_t = float("nan")
                std_t = float("nan")

            # Distribution metrics from histograms
            if edges is not None and edges[c] is not None:
                ea = edges[c]
                bin_widths = np.diff(ea)

                if n_a[c] > 0 and n_b[c] > 0:
                    ha = hist_a[c]
                    hb = hist_b[c]
                    cdf_a = np.cumsum(ha) / ha.sum()
                    cdf_b = np.cumsum(hb) / hb.sum()
                    ks_stat = float(np.max(np.abs(cdf_a - cdf_b)))

                    # Approximate KS p-value
                    en = math.sqrt((n_a[c] * n_b[c]) / (n_a[c] + n_b[c]))
                    p_value = float(min(1.0, 2.0 * math.exp(-2.0 * (en * ks_stat) ** 2)))

                    wasserstein = float(np.sum(np.abs(cdf_a - cdf_b) * bin_widths))
                else:
                    ks_stat = float("nan")
                    p_value = float("nan")
                    wasserstein = float("nan")

                if n_a[c] > 0:
                    ha = hist_a[c]
                    cdf_a = np.cumsum(ha) / ha.sum()

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

                    p_001_a = quantile_from_hist(0.001)
                    p_05_a = quantile_from_hist(0.05)
                    p_25_a = quantile_from_hist(0.25)
                    p_50_a = quantile_from_hist(0.50)
                    p_75_a = quantile_from_hist(0.75)
                    p_95_a = quantile_from_hist(0.95)
                    p_999_a = quantile_from_hist(0.999)
                else:
                    p_001_a = p_05_a = p_25_a = p_50_a = p_75_a = p_95_a = p_999_a = float("nan")

                if n_b[c] > 0:
                    hb = hist_b[c]
                    cdf_b = np.cumsum(hb) / hb.sum()

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

                    p_001_b = quantile_from_hist_b(0.001)
                    p_05_b = quantile_from_hist_b(0.05)
                    p_25_b = quantile_from_hist_b(0.25)
                    p_50_b = quantile_from_hist_b(0.50)
                    p_75_b = quantile_from_hist_b(0.75)
                    p_95_b = quantile_from_hist_b(0.95)
                    p_999_b = quantile_from_hist_b(0.999)
                else:
                    p_001_b = p_05_b = p_25_b = p_50_b = p_75_b = p_95_b = p_999_b = float("nan")

                if n_t[c] > 0:
                    ht = hist_t[c]

                    def quantile_from_hist_t(q):
                        target = q * ht.sum()
                        cum = np.cumsum(ht)
                        idx = int(np.searchsorted(cum, target, side="left"))
                        idx = min(max(idx, 0), len(bin_widths) - 1)
                        prev = cum[idx - 1] if idx > 0 else 0.0
                        if ht[idx] == 0:
                            return float(ea[idx])
                        frac = (target - prev) / ht[idx]
                        return float(ea[idx] + frac * bin_widths[idx])

                    p_001_t = quantile_from_hist_t(0.001)
                    p_05_t = quantile_from_hist_t(0.05)
                    p_25_t = quantile_from_hist_t(0.25)
                    p_50_t = quantile_from_hist_t(0.50)
                    p_75_t = quantile_from_hist_t(0.75)
                    p_95_t = quantile_from_hist_t(0.95)
                    p_999_t = quantile_from_hist_t(0.999)
                else:
                    p_001_t = p_05_t = p_25_t = p_50_t = p_75_t = p_95_t = p_999_t = float("nan")

                if path_t is not None and n_t[c] > 0:
                    ht = hist_t[c]
                    cdf_t = np.cumsum(ht) / ht.sum()
                    wasserstein_a_t = float(np.sum(np.abs(cdf_a - cdf_t) * bin_widths)) if n_a[c] > 0 else float("nan")
                    wasserstein_b_t = float(np.sum(np.abs(cdf_b - cdf_t) * bin_widths)) if n_b[c] > 0 else float("nan")
                else:
                    wasserstein_a_t = float("nan")
                    wasserstein_b_t = float("nan")
            else:
                ks_stat = float("nan")
                p_value = float("nan")
                wasserstein = float("nan")
                p_001_a = p_05_a = p_25_a = p_50_a = p_75_a = p_95_a = p_999_a = float("nan")
                p_001_b = p_05_b = p_25_b = p_50_b = p_75_b = p_95_b = p_999_b = float("nan")
                p_001_t = p_05_t = p_25_t = p_50_t = p_75_t = p_95_t = p_999_t = float("nan")
                wasserstein_a_t = float("nan")
                wasserstein_b_t = float("nan")

            delta_source_c = float("nan")
            delta_trans_c = float("nan")
            if delta_source is not None:
                delta_source_c = delta_source.get(c, float("nan"))
            if delta_trans is not None:
                delta_trans_c = delta_trans.get(c, float("nan"))

            stats_rows.append(
                (
                    c,
                    mean_a,
                    std_a,
                    n_a[c],
                    p_001_a,
                    p_05_a,
                    p_25_a,
                    p_50_a,
                    p_75_a,
                    p_95_a,
                    p_999_a,
                    mean_b,
                    std_b,
                    n_b[c],
                    p_001_b,
                    p_05_b,
                    p_25_b,
                    p_50_b,
                    p_75_b,
                    p_95_b,
                    p_999_b,
                    mean_t,
                    std_t,
                    n_t[c],
                    p_001_t,
                    p_05_t,
                    p_25_t,
                    p_50_t,
                    p_75_t,
                    p_95_t,
                    p_999_t,
                    ks_stat,
                    p_value,
                    wasserstein,
                    wasserstein_a_t,
                    wasserstein_b_t,
                    delta_source_c,
                    delta_trans_c,
                )
            )

        stats_df = pd.DataFrame(
            stats_rows,
            columns=[
                "feature",
                "mean_a",
                "std_a",
                "n_a",
                "p_001_a",
                "p_05_a",
                "p_25_a",
                "p_50_a",
                "p_75_a",
                "p_95_a",
                "p_999_a",
                "mean_b",
                "std_b",
                "n_b",
                "p_001_b",
                "p_05_b",
                "p_25_b",
                "p_50_b",
                "p_75_b",
                "p_95_b",
                "p_999_b",
                "mean_t",
                "std_t",
                "n_t",
                "p_001_t",
                "p_05_t",
                "p_25_t",
                "p_50_t",
                "p_75_t",
                "p_95_t",
                "p_999_t",
                "ks_stat",
                "ks_pvalue",
                "wasserstein",
                "wasserstein_a_t",
                "wasserstein_b_t",
                "delta_source",
                "delta_trans",
            ],
        )
        stats_df.to_csv(out_stats_path, index=False)

        mean_corr = stats_df[["mean_a", "mean_b"]].corr().iloc[0, 1]
        std_corr = stats_df[["std_a", "std_b"]].corr().iloc[0, 1]

        if verbose:
            print(f"Saved per-feature stats to {out_stats_path}")
            print(f"Correlation of feature means (a vs b): {mean_corr:.6f}")
            print(f"Correlation of feature stds (a vs b): {std_corr:.6f}")

    if out_corr_delta_path is not None:
        if path_t is None:
            raise ValueError("--out-corr-delta requires --t.")
        common_corr = [c for c in common if c in cols_t]
        if not common_corr:
            raise ValueError("No common feature columns for correlation delta after filtering.")

        corr_a = _corr_matrix_from_path(path_a, common_corr, chunksize)
        corr_t = _corr_matrix_from_path(path_t, common_corr, chunksize)
        delta = np.abs(corr_a - corr_t)

        rows_delta = []
        for i, fi in enumerate(common_corr):
            for j in range(i + 1, len(common_corr)):
                rows_delta.append((fi, common_corr[j], corr_a[i, j], corr_t[i, j], delta[i, j]))

        delta_df = pd.DataFrame(
            rows_delta,
            columns=["feature_i", "feature_j", "corr_a", "corr_t", "delta"],
        )
        delta_df.to_csv(out_corr_delta_path, index=False)

        if verbose:
            print(f"Saved correlation delta matrix to {out_corr_delta_path}")

    if smoothness_recipe_path is not None and smoothness_plot_path is not None and path_t is not None:
        src_path = smoothness_source_path or path_b
        run_smoothness(
            source_path=src_path,
            translated_path=path_t,
            recipe_path=smoothness_recipe_path,
            out_dir=smoothness_plot_path,
            chunksize=chunksize,
            stay_id=smoothness_stay_id,
            feature=smoothness_feature,
            samples=smoothness_samples,
            seed=smoothness_seed,
            source_space=smoothness_source_space,
            verbose=verbose,
        )


def main():
    ap = argparse.ArgumentParser(description="Compute column-wise Pearson correlation between two CSVs with matching columns.")
    ap.add_argument("--a", help="Path to first dataset (CSV)")
    ap.add_argument("--b", help="Path to second dataset (CSV)")
    ap.add_argument("--out", help="Output CSV path")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk (default: 200000)")
    ap.add_argument("--out-stats", help="Output CSV for per-feature means/stds (optional)")
    ap.add_argument("--t", help="Path to translated dataset (parquet or CSV, optional)")
    ap.add_argument("--out-corr-delta", help="Output CSV for correlation delta matrix (requires --t)")
    ap.add_argument("--smoothness-recipe", help="Path to recipys recipe for denormalization (optional)")
    ap.add_argument("--smoothness-plot", help="Output directory for temporal smoothness plots (optional)")
    ap.add_argument("--smoothness-stay-id", type=int, help="Stay ID for temporal smoothness plot (optional)")
    ap.add_argument("--smoothness-feature", help="Feature name for temporal smoothness plot (optional)")
    ap.add_argument("--smoothness-samples", type=int, default=5, help="Number of random stays to plot (default: 5)")
    ap.add_argument("--smoothness-seed", type=int, default=2222, help="Random seed for stay sampling (default: 2222)")
    ap.add_argument("--smoothness-source", help="Source dataset for smoothness (default: --b)")
    ap.add_argument(
        "--smoothness-source-space",
        choices=["normalized", "physical"],
        default="normalized",
        help="Whether --smoothness-source values are already in physical units (default: normalized).",
    )
    ap.add_argument("--smoothness-only", action="store_true", help="Run only smoothness plots/samples and exit.")
    ap.add_argument("--bins", type=int, default=200, help="Histogram bins for distribution metrics (default: 200)")
    ap.add_argument("--quiet", action="store_true", help="Disable progress prints")
    args = ap.parse_args()

    if args.smoothness_only:
        if not args.t or not args.smoothness_recipe or not args.smoothness_plot:
            raise ValueError("--smoothness-only requires --t, --smoothness-recipe and --smoothness-plot.")
        src = args.smoothness_source or args.b
        if not src:
            raise ValueError("--smoothness-only requires --smoothness-source or --b.")
        run_smoothness(
            source_path=Path(src),
            translated_path=Path(args.t),
            recipe_path=Path(args.smoothness_recipe),
            out_dir=Path(args.smoothness_plot),
            chunksize=args.chunksize,
            stay_id=args.smoothness_stay_id,
            feature=args.smoothness_feature,
            samples=args.smoothness_samples,
            seed=args.smoothness_seed,
            source_space=args.smoothness_source_space,
            write_delta_csv=True,
            verbose=not args.quiet,
        )
        return

    if not args.a or not args.b or not args.out:
        raise ValueError("Missing required args: --a, --b, --out (unless --smoothness-only is used).")

    compute_correlations(
        Path(args.a),
        Path(args.b),
        Path(args.out),
        chunksize=args.chunksize,
        verbose=not args.quiet,
        out_stats_path=Path(args.out_stats) if args.out_stats else None,
        bins=args.bins,
        path_t=Path(args.t) if args.t else None,
        out_corr_delta_path=Path(args.out_corr_delta) if args.out_corr_delta else None,
        smoothness_recipe_path=Path(args.smoothness_recipe) if args.smoothness_recipe else None,
        smoothness_plot_path=Path(args.smoothness_plot) if args.smoothness_plot else None,
        smoothness_stay_id=args.smoothness_stay_id,
        smoothness_feature=args.smoothness_feature,
        smoothness_samples=args.smoothness_samples,
        smoothness_seed=args.smoothness_seed,
        smoothness_source_path=Path(args.smoothness_source) if args.smoothness_source else None,
        smoothness_source_space=args.smoothness_source_space,
    )


if __name__ == "__main__":
    main()
