import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re
import pysubgroup as ps
from sklearn.preprocessing import StandardScaler




# ============================================================
# ----------------- Utility Functions ------------------------
# ============================================================

def aggregate_eeg_if_needed(df):
    """
    Aggregate channel-level EEG band power features into region-level
    relative band power features.

    This function:
    - Detects whether EEG features are already region-aggregated
    - Converts absolute band power to relative power (normalized by total power)
    - Aggregates channels into predefined brain regions
    - Replaces channel-level features with region-level features

    Parameters
    ----------
    df : pd.DataFrame
        Input feature dataframe containing EEG band power columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with region-level EEG features.
    """

    df = df.copy()

    # Mapping from brain regions to EEG channels
    region_map = {
        "Frontal": ["AF7", "AF8", "Fp1", "Fp2"],
        "Parietal": ["PO7", "PO8"],
        "Occipital": ["O1", "O2"]
    }

    # Frequency bands considered
    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

    # Check whether EEG features have already been aggregated
    already_aggregated = any(
        col.startswith("EEG_Alpha_Frontal") for col in df.columns
    )

    if already_aggregated:
        print("EEG already aggregated. Skipping aggregation.")
        return df

    # Identify channel-level EEG band features
    eeg_channel_cols = [
        col for col in df.columns
        if col.startswith("EEG_")
        and any(band in col for band in bands)
        and not any(region in col for region in region_map.keys())
    ]

    if len(eeg_channel_cols) == 0:
        print("No channel-level EEG features found. Skipping aggregation.")
        return df

    # Extract EEG data and remove extremely small values
    eeg_data = df[eeg_channel_cols].copy()
    eeg_data[eeg_data < 1e-15] = 0.0

    # Compute total power per sample
    total_power = eeg_data.sum(axis=1)
    total_power[total_power == 0] = np.nan

    # Convert absolute power to relative power
    eeg_relative = eeg_data.div(total_power, axis=0)
    eeg_relative = eeg_relative.fillna(0.0)

    aggregated_features = {}

    # Aggregate channel-level features into region-level features
    for band in bands:
        for region, channels in region_map.items():

            cols = [
                f"EEG_{band}_{ch}"
                for ch in channels
                if f"EEG_{band}_{ch}" in eeg_relative.columns
            ]

            if len(cols) > 0:
                aggregated_features[f"EEG_{band}_{region}"] = (
                    eeg_relative[cols].mean(axis=1)
                )

    # Remove original channel-level features
    df.drop(columns=eeg_channel_cols, inplace=True)

    # Add aggregated region-level features
    for col_name, values in aggregated_features.items():
        df[col_name] = values

    print(f"EEG aggregated: {len(eeg_channel_cols)} → {len(aggregated_features)}")

    return df

def apply_mode_binary(s, mode_val):
    """
    Convert a near-constant feature into a binary categorical feature:
    eq_mode vs neq_mode.
    """
    if isinstance(mode_val, float):
        mode_str = f"{mode_val:.3f}"
    else:
        mode_str = str(mode_val)
    return np.where(s == mode_val, f"eq_{mode_str}", f"neq_{mode_str}")

def build_interval_labels(bins, n_bins):
    """
    Build human-readable interval labels.
    """
    if n_bins == 2:
        sem = ["low", "high"]
    elif n_bins == 3:
        sem = ["low", "mid", "high"]
    elif n_bins == 4:
        sem = ["q1", "q2", "q3", "q4"]
    else:
        sem = [f"bin{i}" for i in range(len(bins) - 1)]

    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        labels.append(f"{sem[i]}({lo:.3f}-{hi:.3f})")
    return labels

def adaptive_discretize_for_subgroups(
    df,
    label_col="label",
    feature_keywords=None,
    n_bins=4,
    low_info_filter=True,
    dominant_ratio=0.9,
    min_rare_count=3,
    min_valid_ratio=0.5,
    min_unique_values=5,
    max_zero_ratio=0.95,
    min_iqr=1e-8,
    keep_non_numeric=True,
    save_path=None
):
    """
    Adaptive discretization for subgroup discovery (feature-only version).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_col : str
        Target column name. This column will be kept unchanged.
    feature_keywords : list[str] or None
        Prefix list used to select candidate feature columns.
        If None, a default multimodal feature prefix list is used.
    n_bins : int
        Number of quantile bins for qcut-based discretization.
    low_info_filter : bool
        Whether to apply low-information filtering to numeric features.
    dominant_ratio : float
        Threshold above which a feature is treated as near-constant.
    min_rare_count : int
        Minimum number of non-mode samples required to keep a near-constant feature.
    min_valid_ratio : float
        Minimum ratio of non-missing values required to keep a feature.
    min_unique_values : int
        Minimum number of unique values required for numeric features.
    max_zero_ratio : float
        Maximum allowed ratio of zero values among valid numeric samples.
    min_iqr : float
        Minimum interquartile range required for numeric features.
    keep_non_numeric : bool
        Whether to keep non-numeric columns as nominal features.
    save_path : str or Path or None
        Optional directory to save discretized data and strategy files.

    Returns
    -------
    df_disc : pd.DataFrame
        Discretized dataframe containing label + selected feature columns.
    strategy_df : pd.DataFrame
        Per-column processing strategy summary.
    dropped_df : pd.DataFrame
        List of dropped columns.
    """

    df = df.copy()

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    if feature_keywords is None:
        feature_keywords = ["EEG_", "ECG_", "EDA_", "EYE_", "HEAD_", "PRE_", "SIM_"]

    # Select only feature columns matching prefixes
    feature_cols = [
        c for c in df.columns
        if c != label_col and any(c.startswith(k) for k in feature_keywords)
    ]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns matched feature_keywords.")

    df_disc = df[[label_col] + feature_cols].copy()

    dropped = []
    strategy = {}
    N = len(df_disc)

    for col in feature_cols:
        s = df_disc[col]

        # ---------------- Basic sanity ----------------
        if s.dropna().empty:
            dropped.append(col)
            strategy[col] = "dropped_all_nan"
            continue

        if s.nunique(dropna=True) < 2:
            dropped.append(col)
            strategy[col] = "dropped_constant"
            continue

        # ---------------- Non-numeric features ----------------
        if not pd.api.types.is_numeric_dtype(s):
            if not keep_non_numeric:
                dropped.append(col)
                strategy[col] = "dropped_non_numeric"
                continue

            s_obj = s.astype(str)
            s_obj = s_obj.replace({
                "nan": np.nan,
                "None": np.nan,
                "NaN": np.nan,
                "": np.nan
            })

            valid_ratio = s_obj.notna().mean()

            if low_info_filter and valid_ratio < min_valid_ratio:
                dropped.append(col)
                strategy[col] = "dropped_non_numeric_low_valid_ratio"
                continue

            unique_n = s_obj.nunique(dropna=True)
            if low_info_filter and unique_n < 2:
                dropped.append(col)
                strategy[col] = "dropped_non_numeric_low_unique"
                continue

            df_disc[col] = s_obj.fillna("Missing")
            strategy[col] = "kept_nominal"
            continue

        # ---------------- Numeric features ----------------
        valid_n = s.notna().sum()
        valid_ratio = valid_n / N

        if low_info_filter:
            if valid_ratio < min_valid_ratio:
                dropped.append(col)
                strategy[col] = "dropped_low_valid_ratio"
                continue

            unique_n = s.nunique(dropna=True)
            if unique_n < min_unique_values:
                dropped.append(col)
                strategy[col] = "dropped_low_unique"
                continue

            zero_ratio = (s == 0).sum() / max(valid_n, 1)
            if zero_ratio > max_zero_ratio:
                dropped.append(col)
                strategy[col] = "dropped_extreme_zero_sparse"
                continue

            q75, q25 = np.percentile(s.dropna(), [75, 25])
            iqr = q75 - q25
            if iqr < min_iqr:
                dropped.append(col)
                strategy[col] = "dropped_low_iqr"
                continue

        # ---------------- Near-constant handling ----------------
        vc = s.value_counts(dropna=True)
        mode_val = vc.index[0]
        mode_ratio = vc.iloc[0] / valid_n
        rare_count = valid_n - vc.iloc[0]

        if mode_ratio >= dominant_ratio:
            if rare_count < min_rare_count:
                dropped.append(col)
                strategy[col] = "dropped_too_rare"
                continue

            if (mode_val == 0) and (s.min(skipna=True) >= 0):
                df_disc[col] = np.where(s.fillna(0) == 0, "zero", "nonzero")
                strategy[col] = "zero_binary"
            else:
                df_disc[col] = apply_mode_binary(s, mode_val)
                strategy[col] = "mode_binary"
            continue

        # ---------------- qcut path ----------------
        try:
            _, bins = pd.qcut(s, q=n_bins, duplicates="drop", retbins=True)
            bins = np.unique(bins)

            if len(bins) < 3:
                raise ValueError("Degenerate bins")

            labels = build_interval_labels(bins, n_bins)

            df_disc[col] = pd.cut(
                s,
                bins=bins,
                labels=labels[:len(bins) - 1],
                include_lowest=True
            )

            # Convert category to string for easier downstream selector building
            df_disc[col] = df_disc[col].astype(str).replace("nan", np.nan)

            if df_disc[col].nunique(dropna=True) < 2:
                if rare_count >= min_rare_count:
                    df_disc[col] = apply_mode_binary(s, mode_val)
                    strategy[col] = "mode_binary_fallback"
                else:
                    dropped.append(col)
                    strategy[col] = "dropped_qcut_degenerate"
            else:
                strategy[col] = "qcut_interval"

        except Exception:
            if rare_count >= min_rare_count:
                df_disc[col] = apply_mode_binary(s, mode_val)
                strategy[col] = "mode_binary_fallback"
            else:
                dropped.append(col)
                strategy[col] = "dropped_qcut_error"

    # Drop rejected columns
    if dropped:
        df_disc = df_disc.drop(columns=dropped, errors="ignore")

    strategy_df = pd.DataFrame({
        "column": list(strategy.keys()),
        "strategy": list(strategy.values())
    }).sort_values("column").reset_index(drop=True)

    dropped_df = pd.DataFrame({"dropped_columns": dropped}) \
        if dropped else pd.DataFrame(columns=["dropped_columns"])

    # Save artifacts
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        df_disc.to_csv(save_path / "data_discretized_for_subgroups.csv", index=False)
        strategy_df.to_csv(save_path / "subgroup_discretization_strategy.csv", index=False)
        dropped_df.to_csv(save_path / "subgroup_discretization_dropped.csv", index=False)

    return df_disc, strategy_df, dropped_df

def build_searchspace_from_discretized(
    df_disc,
    label_col="label",
    ignore_cols=None,
    max_nominal_values=12,
    min_selector_coverage=3,
    drop_missing_like=True,
    verbose=False
):
    """
    Build subgroup selectors from an already discretized dataframe.

    All feature columns are treated as nominal after discretization.
    """

    if ignore_cols is None:
        ignore_cols = []

    invalid_tokens = {"nan", "None", "NaN", "Missing", ""}

    feature_cols = [
        c for c in df_disc.columns
        if c != label_col and c not in ignore_cols
    ]

    searchspace = []
    summary_rows = []
    selector_rows = []

    def sort_values_for_intervals(values):
        """
        Keep q1, q2, q3, q4 in natural order when possible.
        Otherwise return as-is.
        """
        def get_rank(v):
            m = re.match(r"q(\d+)\(", str(v))
            if m:
                return int(m.group(1))
            return 9999

        if all(re.match(r"q\d+\(", str(v)) for v in values):
            return sorted(values, key=get_rank)
        return values

    for col in feature_cols:
        s = df_disc[col].astype(str).replace("nan", np.nan)

        if s.dropna().empty:
            summary_rows.append((col, "empty_after_discretization", 0, 0))
            continue

        value_counts = s.value_counts(dropna=True)

        # Remove missing-like categories
        if drop_missing_like:
            value_counts = value_counts[
                ~value_counts.index.astype(str).isin(invalid_tokens)
            ]

        if value_counts.empty:
            summary_rows.append((col, "empty_after_missing_filter", 0, 0))
            continue

        raw_values = value_counts.index.tolist()
        sorted_values = sort_values_for_intervals(raw_values)

        # Reorder value_counts if interval-like
        value_counts = value_counts.reindex(sorted_values)

        selectors_this_col = []

        for val, cnt in value_counts.items():
            if cnt < min_selector_coverage:
                continue

            try:
                selector = ps.EqualitySelector(col, val)
                selectors_this_col.append(selector)
                selector_rows.append({
                    "column": col,
                    "value": val,
                    "coverage": int(cnt),
                    "selector_repr": str(selector)
                })
            except Exception:
                continue

        # Cap only after sorting/filtering
        selectors_this_col = selectors_this_col[:max_nominal_values]

        # Also cap selector_rows for this column consistently
        kept_values = {str(sel).split("==")[-1].strip("'") for sel in selectors_this_col}
        # Note: optional strict sync with selector_rows is not necessary for function behavior

        searchspace.extend(selectors_this_col)

        summary_rows.append((
            col,
            "nominal_selector",
            len(selectors_this_col),
            int(value_counts.max()) if len(value_counts) > 0 else 0
        ))

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["column", "selector_type", "n_selectors", "max_value_coverage"]
    )

    selector_df = pd.DataFrame(selector_rows)


    if verbose:
        print("\nSearchspace summary:")
        print(f"- feature columns used: {(summary_df['n_selectors'] > 0).sum()}")
        print(f"- total selectors: {len(searchspace)}")

        if not summary_df.empty:
            print("\nTop columns by number of selectors:")
            print(summary_df.sort_values("n_selectors", ascending=False).head(10))

        if len(searchspace) > 0:
            print("\nFirst 10 selectors:")
            for s in searchspace[:10]:
                print("  ", s)

    return searchspace, summary_df, selector_df

def clean_for_subgroup_discovery(
    df,
    label_col="label",
    drop_id_like=True,
    max_missing_ratio=0.4,
    min_unique_numeric=5,
):
    """
    Clean dataframe for subgroup discovery.

    Steps:
    - Drop obvious id/meta columns
    - Remove rows with missing label
    - Drop columns with too many missing values
    - Fill remaining missing values
    - Keep numeric and categorical columns only
    """
    df = df.copy()

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    # Remove rows with missing label
    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    # Drop id/meta columns
    if drop_id_like:
        drop_cols = []
        for col in df.columns:
            col_low = col.lower()
            if col == label_col:
                continue
            if (
                col_low.startswith("id")
                or col_low in {
                    "patient", "scenario", "group", "target",
                    "id_patient", "id_anonymat", "id_test", "id_scenario"
                }
            ):
                drop_cols.append(col)
        df = df.drop(columns=drop_cols, errors="ignore")

    # Drop columns with too many missing values
    keep_cols = []
    for col in df.columns:
        if col == label_col:
            keep_cols.append(col)
            continue
        missing_ratio = df[col].isna().mean()
        if missing_ratio <= max_missing_ratio:
            keep_cols.append(col)
    df = df[keep_cols].copy()

    # Convert object columns when possible
    for col in df.columns:
        if col == label_col:
            continue

        # Try numeric conversion first
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            # Keep as numeric if enough values convert
            if converted.notna().mean() > 0.8:
                df[col] = converted
            else:
                df[col] = df[col].astype(str).str.strip()

    # Drop constant columns
    constant_cols = []
    for col in df.columns:
        if col == label_col:
            continue
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    df = df.drop(columns=constant_cols, errors="ignore")

    # Fill missing values
    for col in df.columns:
        if col == label_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val).astype(str)

    # Remove very-low-information numeric columns
    low_info_numeric = []
    for col in df.columns:
        if col == label_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) < min_unique_numeric:
                low_info_numeric.append(col)

    # Do not force-remove them all; only remove if there are plenty of other features
    if len(low_info_numeric) > 0 and (len(df.columns) - 1 - len(low_info_numeric)) >= 10:
        df = df.drop(columns=low_info_numeric, errors="ignore")

    return df

def build_searchspace_manual(
    df,
    ignore_cols=None,
    nbins=4,
    max_nominal_values=10,
    min_bin_size=5
):
    """
    Build subgroup discovery search space manually.

    Numeric columns:
    - discretize by quantiles
    - create interval selectors manually

    Categorical columns:
    - create equality selectors manually
    """
    if ignore_cols is None:
        ignore_cols = []

    feature_cols = [c for c in df.columns if c not in ignore_cols]
    searchspace = []

    numeric_count = 0
    categorical_count = 0

    for col in feature_cols:
        s = df[col]

        # ---------------- Numeric columns ----------------
        if pd.api.types.is_numeric_dtype(s):
            s_nonan = s.dropna()

            if s_nonan.nunique() < 2:
                print(f"[Numeric][Skip constant] {col}")
                continue

            try:
                # Quantile bin edges
                quantiles = np.linspace(0, 1, nbins + 1)
                bins = np.unique(np.quantile(s_nonan, quantiles))

                if len(bins) < 2:
                    print(f"[Numeric][Skip degenerate bins] {col}")
                    continue

                selectors_this_col = []

                for i in range(len(bins) - 1):
                    lower = bins[i]
                    upper = bins[i + 1]

                    # Skip zero-width interval
                    if lower == upper:
                        continue

                    mask = (s >= lower) & (s <= upper) if i == len(bins) - 2 else (s >= lower) & (s < upper)
                    covered_n = int(mask.sum())

                    if covered_n < min_bin_size:
                        continue

                    try:
                        selector = ps.IntervalSelector(col, lower, upper)
                        selectors_this_col.append(selector)
                    except Exception as e:
                        print(f"[Numeric][Selector fail] {col} [{lower}, {upper}]: {e}")

                if len(selectors_this_col) > 0:
                    searchspace.extend(selectors_this_col)
                    numeric_count += 1
                    print(f"[Numeric] {col}: {len(selectors_this_col)} selectors")
                else:
                    print(f"[Numeric][No valid selectors] {col}")

            except Exception as e:
                print(f"[Numeric][Skip] {col}: {e}")

        # ---------------- Categorical columns ----------------
        else:
            try:
                s = s.astype(str).fillna("nan")
                value_counts = s.value_counts(dropna=False)

                if len(value_counts) == 0:
                    print(f"[Categorical][Skip empty] {col}")
                    continue

                # Only keep top categories to avoid explosion
                kept_values = value_counts.head(max_nominal_values).index.tolist()
                selectors_this_col = []

                for val in kept_values:
                    covered_n = int((s == val).sum())
                    if covered_n < min_bin_size:
                        continue

                    try:
                        selector = ps.EqualitySelector(col, val)
                        selectors_this_col.append(selector)
                    except Exception as e:
                        print(f"[Categorical][Selector fail] {col} == {val}: {e}")

                if len(selectors_this_col) > 0:
                    searchspace.extend(selectors_this_col)
                    categorical_count += 1
                    print(f"[Categorical] {col}: {len(selectors_this_col)} selectors")
                else:
                    print(f"[Categorical][No valid selectors] {col}")

            except Exception as e:
                print(f"[Categorical][Skip] {col}: {e}")

    print("\nSearchspace summary:")
    print(f"- numeric columns used: {numeric_count}")
    print(f"- categorical columns used: {categorical_count}")
    print(f"- total selectors: {len(searchspace)}")

    if len(searchspace) > 0:
        print("First 10 selectors:")
        for s in searchspace[:10]:
            print("  ", s)

    return searchspace

def subgroup_result_to_dataframe(result, target_label, total_n, positive_n):
    """
    Convert pysubgroup result object to a DataFrame.
    """
    rows = []

    # Baseline positive rate for one-vs-rest
    base_rate = positive_n / total_n if total_n > 0 else np.nan

    for rank, entry in enumerate(result.to_descriptions(), start=1):
        quality = entry[0]
        subgroup = entry[1]

        covered = subgroup.covers(result.task.data)
        subgroup_df = result.task.data[covered]

        subgroup_size = len(subgroup_df)
        positive_count = int((subgroup_df["target_binary"] == 1).sum())
        subgroup_precision = positive_count / subgroup_size if subgroup_size > 0 else np.nan
        lift = subgroup_precision / base_rate if base_rate > 0 else np.nan

        rows.append({
            "target_label": target_label,
            "rank": rank,
            "quality": float(quality) if quality is not None else np.nan,
            "subgroup": str(subgroup),
            "subgroup_size": subgroup_size,
            "positive_count": positive_count,
            "precision_in_subgroup": subgroup_precision,
            "baseline_positive_rate": base_rate,
            "lift": lift,
        })

    return pd.DataFrame(rows)

# ============================================================
# ---------------- Health base Prototype Deviation -----------
# ============================================================

class PrototypeDeviationAnalysis:
    """
    Prototype deviation analysis for interpretable class-wise pattern mining.

    Design
    ------
    This class separates two layers:

    1. Explanation layer
       - Prototypes are computed in the original feature space
       - Visualization uses raw deviations
       - Useful for clinical interpretation

    2. Mining layer
       - Deviations are also standardized
       - Ranking uses standardized deviation
       - Useful for feature discovery across heterogeneous scales

    Main workflow
    -------------
    1. Prepare cleaned multimodal feature table
    2. Compute one prototype per class in original scale
    3. Use reference class (default: Health) as baseline
    4. Compute raw deviation and scaled deviation
    5. Summarize:
       - global top deviating features
       - domain-level summary
       - domain representative features

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing labels and multimodal features.
    label_col : str
        Label column name.
    reference_label : str
        Reference healthy class label.
    feature_keywords : list[str] or None
        Prefixes used to detect feature domains.
    save_path : str or Path
        Directory to save results.
    center_method : str
        'mean' or 'median' for prototype computation.
    scaling_method : str
        'std' or 'iqr' for standardized deviation.
    min_scale : float
        Lower bound to avoid division explosion.
    domain_top_k_by_scaled : int
        Number of candidates retained per domain using scaled deviation.
    representative_per_domain : int
        Final number of representative features kept per domain.
    representative_abs_delta_ratio : float
        Relative threshold on abs_delta within domain candidates.
        Example: 0.7 means keeping features whose abs_delta is at least
        70% of the largest abs_delta among the domain candidates.
    """

    def __init__(
        self,
        df,
        label_col="label",
        reference_label="Health",
        feature_keywords=None,
        save_path=Path("./prototype_deviation"),
        center_method="mean",
        scaling_method="std",   # "std" or "iqr"
        min_scale=1e-6,
        domain_top_k_by_scaled=5,
        representative_per_domain=1,
        representative_abs_delta_ratio=0.7,
    ):
        self.df_raw = df.copy()
        self.label_col = label_col
        self.reference_label = reference_label
        self.feature_keywords = feature_keywords or [
            "EEG_", "ECG_", "EDA_", "EYE_", "HEAD_", "PRE_", "SIM_"
        ]
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.center_method = center_method
        self.scaling_method = scaling_method.lower()
        self.min_scale = min_scale

        self.domain_top_k_by_scaled = domain_top_k_by_scaled
        self.representative_per_domain = representative_per_domain
        self.representative_abs_delta_ratio = representative_abs_delta_ratio

        self.df = None
        self.feature_cols = None
        self.labels_ = None

        self.centroids_ = None
        self.reference_centroid_ = None
        self.reference_scale_ = None

        self.deviation_tables_ = {}
        self.global_top_features_ = {}
        self.domain_summary_ = {}
        self.domain_candidate_features_ = {}
        self.domain_representative_features_ = {}

    # ========================================================
    # -------------------- Utilities -------------------------
    # ========================================================

    def _save_json(self, obj, file_path):
        """Save Python object as JSON."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)

    def _extract_domain(self, feature_name):
        """Extract domain name from feature prefix."""
        for prefix in self.feature_keywords:
            if feature_name.startswith(prefix):
                return prefix.replace("_", "")
        return "OTHER"

    def _get_feature_columns(self, df):
        """
        Select valid numeric feature columns based on prefixes.
        """
        feature_cols = []
        for col in df.columns:
            if col == self.label_col:
                continue
            if any(col.startswith(k) for k in self.feature_keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    feature_cols.append(col)
        return feature_cols

    def _compute_center(self, sub_df):
        """
        Compute class centroid using mean or median.
        """
        if self.center_method == "median":
            return sub_df.median(axis=0)
        return sub_df.mean(axis=0)

    def _compute_reference_scale(self, ref_df):
        """
        Compute scaling denominator for standardized deviation.

        Options
        -------
        - std : standard deviation
        - iqr : interquartile range (robust)
        """
        if self.scaling_method == "iqr":
            q75 = ref_df.quantile(0.75)
            q25 = ref_df.quantile(0.25)
            scale = q75 - q25
        else:
            scale = ref_df.std(axis=0)

        scale = scale.fillna(0.0)
        scale = scale.clip(lower=self.min_scale)
        return scale

    def _rank_within_series_desc(self, s):
        """
        Return descending rank series.
        Highest value gets rank 1.
        """
        return s.rank(ascending=False, method="dense")

    # ========================================================
    # -------------------- Phase 0 ---------------------------
    # ========================================================

    def prepare_data(self):
        """
        Prepare dataframe for prototype analysis.

        Steps
        -----
        - keep rows with valid label
        - keep numeric multimodal feature columns
        - drop all-NaN columns
        - fill remaining missing values using column median
        """
        df = self.df_raw.copy()

        if self.label_col not in df.columns:
            raise ValueError(f"Missing label column: {self.label_col}")

        df = df[df[self.label_col].notna()].copy()
        df[self.label_col] = df[self.label_col].astype(str).str.strip()

        feature_cols = self._get_feature_columns(df)

        if len(feature_cols) == 0:
            raise ValueError("No valid numeric feature columns found for prototype analysis.")

        feature_cols = [c for c in feature_cols if not df[c].isna().all()]
        if len(feature_cols) == 0:
            raise ValueError("All selected feature columns are empty.")

        # Fill missing values
        for col in feature_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        self.df = df[[self.label_col] + feature_cols].copy()
        self.feature_cols = feature_cols
        self.labels_ = sorted(self.df[self.label_col].unique().tolist())

        if self.reference_label not in self.labels_:
            raise ValueError(
                f"Reference label '{self.reference_label}' not found in labels: {self.labels_}"
            )

        self.df.to_csv(self.save_path / "prototype_input_cleaned.csv", index=False)
        return self.df

    # ========================================================
    # -------------------- Phase 1 ---------------------------
    # ========================================================

    def compute_prototypes(self):
        """
        Compute one centroid per class in original scale.
        """
        if self.df is None:
            self.prepare_data()

        centroids = {}

        for label in self.labels_:
            sub_df = self.df[self.df[self.label_col] == label][self.feature_cols]
            centroids[label] = self._compute_center(sub_df)

        centroids_df = pd.DataFrame(centroids).T
        centroids_df.index.name = self.label_col

        ref_df = self.df[self.df[self.label_col] == self.reference_label][self.feature_cols]
        ref_centroid = centroids_df.loc[self.reference_label]
        ref_scale = self._compute_reference_scale(ref_df)

        centroids_df.to_csv(self.save_path / "class_centroids.csv")
        ref_centroid.to_csv(self.save_path / "reference_centroid.csv", header=["value"])
        ref_scale.to_csv(self.save_path / "reference_scale.csv", header=["scale"])

        self.centroids_ = centroids_df
        self.reference_centroid_ = ref_centroid
        self.reference_scale_ = ref_scale

        return centroids_df

    # ========================================================
    # -------------------- Phase 2 ---------------------------
    # ========================================================

    def compute_deviation(self):
        """
        Compute raw and scaled deviation for each non-reference class.

        Output columns
        --------------
        - feature
        - domain
        - reference_value
        - target_value
        - raw_delta
        - abs_delta
        - scaled_delta
        - abs_scaled_delta
        - raw_rank
        - scaled_rank
        - hybrid_rank_score
        """
        if self.centroids_ is None:
            self.compute_prototypes()

        for label in self.labels_:
            if label == self.reference_label:
                continue

            target_centroid = self.centroids_.loc[label]

            raw_delta = target_centroid - self.reference_centroid_
            abs_delta = raw_delta.abs()

            scaled_delta = raw_delta / self.reference_scale_
            abs_scaled_delta = scaled_delta.abs()

            result_df = pd.DataFrame({
                "feature": self.feature_cols,
                "domain": [self._extract_domain(c) for c in self.feature_cols],
                "reference_value": self.reference_centroid_.values,
                "target_value": target_centroid.values,
                "raw_delta": raw_delta.values,
                "abs_delta": abs_delta.values,
                "scaled_delta": scaled_delta.values,
                "abs_scaled_delta": abs_scaled_delta.values,
            })

            # Ranking columns
            result_df["raw_rank"] = self._rank_within_series_desc(result_df["abs_delta"])
            result_df["scaled_rank"] = self._rank_within_series_desc(result_df["abs_scaled_delta"])

            # Hybrid ranking score: smaller is better
            result_df["hybrid_rank_score"] = (
                result_df["raw_rank"] + result_df["scaled_rank"]
            ) / 2.0

            result_df = result_df.sort_values(
                by=["scaled_rank", "raw_rank", "feature"],
                ascending=[True, True, True]
            ).reset_index(drop=True)

            result_df.to_csv(
                self.save_path / f"prototype_deviation_{label}_vs_{self.reference_label}.csv",
                index=False
            )

            self.deviation_tables_[label] = result_df

        return self.deviation_tables_

    # ========================================================
    # -------------------- Phase 3 ---------------------------
    # ========================================================

    def summarize_global_top_features(self, top_k_global=10):
        """
        Mining layer:
        Get global top-k features ranked by abs_scaled_delta,
        while keeping raw values for explanation.
        """
        if len(self.deviation_tables_) == 0:
            self.compute_deviation()

        for label, df_dev in self.deviation_tables_.items():
            global_top = df_dev.sort_values(
                by=["scaled_rank", "raw_rank"],
                ascending=[True, True]
            ).head(top_k_global).copy()

            global_top.to_csv(
                self.save_path / f"top_global_features_{label}.csv",
                index=False
            )
            self.global_top_features_[label] = global_top

        return self.global_top_features_

    def summarize_domains(self):
        """
        Build domain-level summary.

        Explanation layer metrics
        -------------------------
        - mean_abs_delta
        - max_abs_delta

        Mining layer metrics
        --------------------
        - mean_abs_scaled_delta
        - max_abs_scaled_delta
        """
        if len(self.deviation_tables_) == 0:
            self.compute_deviation()

        for label, df_dev in self.deviation_tables_.items():
            summary = (
                df_dev.groupby("domain")
                .agg(
                    mean_abs_delta=("abs_delta", "mean"),
                    max_abs_delta=("abs_delta", "max"),
                    mean_abs_scaled_delta=("abs_scaled_delta", "mean"),
                    max_abs_scaled_delta=("abs_scaled_delta", "max"),
                    n_features=("feature", "count")
                )
                .reset_index()
                .sort_values(
                    by=["mean_abs_scaled_delta", "mean_abs_delta"],
                    ascending=False
                )
            )

            summary.to_csv(
                self.save_path / f"domain_summary_{label}.csv",
                index=False
            )
            self.domain_summary_[label] = summary

        return self.domain_summary_

    def summarize_domain_representatives(self):
        """
        Two-step representative feature selection per domain.

        Step 1
        ------
        Select top-k candidates by abs_scaled_delta within each domain.

        Step 2
        ------
        Among those candidates, keep features whose abs_delta is also large.
        A relative threshold is used:
            abs_delta >= max(abs_delta among candidates) * representative_abs_delta_ratio

        Finally, keep at most representative_per_domain features.

        Notes
        -----
        - This keeps mining sensitivity from scaled deviation
        - But final representative features remain interpretable in raw scale
        """
        if len(self.deviation_tables_) == 0:
            self.compute_deviation()

        for label, df_dev in self.deviation_tables_.items():
            candidate_rows = []
            representative_rows = []

            for domain, sub_df in df_dev.groupby("domain"):
                # Step 1: candidates by scaled deviation
                candidates = sub_df.sort_values(
                    by=["scaled_rank", "raw_rank"],
                    ascending=[True, True]
                ).head(self.domain_top_k_by_scaled).copy()

                if candidates.empty:
                    continue

                candidates["selection_stage"] = "domain_top_k_by_scaled"
                candidate_rows.append(candidates)

                # Step 2: filter by raw delta threshold
                max_abs_delta = candidates["abs_delta"].max()

                if pd.isna(max_abs_delta) or max_abs_delta <= 0:
                    selected = candidates.head(self.representative_per_domain).copy()
                else:
                    threshold = max_abs_delta * self.representative_abs_delta_ratio
                    selected = candidates[candidates["abs_delta"] >= threshold].copy()

                    if selected.empty:
                        selected = candidates.head(self.representative_per_domain).copy()
                    else:
                        selected = selected.sort_values(
                            by=["scaled_rank", "raw_rank"],
                            ascending=[True, True]
                        ).head(self.representative_per_domain).copy()

                selected["selection_stage"] = "domain_representative"
                selected["domain_abs_delta_threshold"] = (
                    max_abs_delta * self.representative_abs_delta_ratio
                    if max_abs_delta > 0 else np.nan
                )
                representative_rows.append(selected)

            if len(candidate_rows) > 0:
                candidate_df = pd.concat(candidate_rows, axis=0, ignore_index=True)
            else:
                candidate_df = pd.DataFrame()

            if len(representative_rows) > 0:
                representative_df = pd.concat(representative_rows, axis=0, ignore_index=True)
            else:
                representative_df = pd.DataFrame()

            candidate_df.to_csv(
                self.save_path / f"domain_candidate_features_{label}.csv",
                index=False
            )
            representative_df.to_csv(
                self.save_path / f"domain_representative_features_{label}.csv",
                index=False
            )

            self.domain_candidate_features_[label] = candidate_df
            self.domain_representative_features_[label] = representative_df

        return self.domain_candidate_features_, self.domain_representative_features_

    # ========================================================
    # -------------------- Phase 4 ---------------------------
    # ========================================================

    def plot_top_features_scaled_with_raw_labels(self, label, top_k=10, decimals=3):
        """
        Best visualization for prototype deviation.

        Strategy
        --------
        - Ranking and bar length use scaled_delta
        - Text annotations use raw values for interpretation

        This avoids visual distortion caused by different feature scales,
        while still keeping clinically interpretable raw values.

        Parameters
        ----------
        label : str
            Target class label to compare against the reference class.
        top_k : int
            Number of top features to display.
        decimals : int
            Number of decimals used in raw value annotations.
        """
        if label not in self.deviation_tables_:
            raise ValueError(f"Missing deviation table for label: {label}")

        df_dev = self.deviation_tables_[label].copy()

        # Select top features by mining-layer ranking
        plot_df = df_dev.sort_values(
            by=["scaled_rank", "raw_rank"],
            ascending=[True, True]
        ).head(top_k).copy()

        if plot_df.empty:
            return

        # Reverse order so the strongest feature is shown at the top
        plot_df = plot_df.iloc[::-1].reset_index(drop=True)

        y_pos = np.arange(len(plot_df))
        bar_values = plot_df["scaled_delta"].values

        # Dynamic figure height
        fig_height = max(4, 0.55 * len(plot_df) + 1.5)
        plt.figure(figsize=(12, fig_height))

        bars = plt.barh(y_pos, bar_values)

        plt.yticks(y_pos, plot_df["feature"])
        plt.axvline(0, color="black", linewidth=1)

        plt.title(
            f"Top {top_k} prototype deviations: {label} vs {self.reference_label}\n"
            f"(bar = scaled deviation, text = raw values)"
        )
        plt.xlabel(f"scaled_delta ({self.scaling_method})")
        plt.ylabel("Feature")

        # Adjust x-limits to leave space for annotations
        x_min = np.nanmin(bar_values)
        x_max = np.nanmax(bar_values)

        if np.isfinite(x_min) and np.isfinite(x_max):
            span = max(abs(x_min), abs(x_max))
            pad = max(0.5, span * 0.35)
            plt.xlim(x_min - pad, x_max + pad)

        # Add text annotations with raw values
        for i, row in plot_df.iterrows():
            scaled_val = row["scaled_delta"]
            ref_val = row["reference_value"]
            tgt_val = row["target_value"]
            raw_delta = row["raw_delta"]

            annotation = (
                f"{ref_val:.{decimals}f} → {tgt_val:.{decimals}f} "
                f"(Δ={raw_delta:.{decimals}f})"
            )

            # Put annotation slightly outside the bar
            offset = max(0.05, 0.03 * max(abs(x_min), abs(x_max), 1))

            if scaled_val >= 0:
                x_text = scaled_val + offset
                ha = "left"
            else:
                x_text = scaled_val - offset
                ha = "right"

            plt.text(
                x_text,
                i,
                annotation,
                va="center",
                ha=ha,
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(
            self.save_path / f"top_features_scaled_with_raw_labels_{label}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    def plot_domain_heatmap_raw(self):
        """
        Explanation-layer heatmap:
        domain-level mean absolute raw deviation.
        """
        if len(self.domain_summary_) == 0:
            self.summarize_domains()

        all_rows = []
        for label, summary in self.domain_summary_.items():
            tmp = summary.copy()
            tmp["target_label"] = label
            all_rows.append(tmp)

        if len(all_rows) == 0:
            return

        df_all = pd.concat(all_rows, axis=0, ignore_index=True)

        heat_df = df_all.pivot(
            index="target_label",
            columns="domain",
            values="mean_abs_delta"
        ).fillna(0.0)

        plt.figure(figsize=(8, 4 + 0.5 * len(heat_df)))
        plt.imshow(heat_df.values, aspect="auto")
        plt.colorbar(label="mean_abs_delta")

        plt.xticks(range(len(heat_df.columns)), heat_df.columns, rotation=45)
        plt.yticks(range(len(heat_df.index)), heat_df.index)

        for i in range(heat_df.shape[0]):
            for j in range(heat_df.shape[1]):
                val = heat_df.iloc[i, j]
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.title("Domain-level prototype deviation (raw)")
        plt.tight_layout()
        plt.savefig(self.save_path / "domain_deviation_heatmap_raw.png", dpi=300)
        plt.close()

        heat_df.to_csv(self.save_path / "domain_deviation_heatmap_raw_values.csv")

    def plot_domain_heatmap_scaled(self):
        """
        Mining-layer heatmap:
        domain-level mean absolute standardized deviation.
        """
        if len(self.domain_summary_) == 0:
            self.summarize_domains()

        all_rows = []
        for label, summary in self.domain_summary_.items():
            tmp = summary.copy()
            tmp["target_label"] = label
            all_rows.append(tmp)

        if len(all_rows) == 0:
            return

        df_all = pd.concat(all_rows, axis=0, ignore_index=True)

        heat_df = df_all.pivot(
            index="target_label",
            columns="domain",
            values="mean_abs_scaled_delta"
        ).fillna(0.0)

        plt.figure(figsize=(8, 4 + 0.5 * len(heat_df)))
        plt.imshow(heat_df.values, aspect="auto")
        plt.colorbar(label="mean_abs_scaled_delta")

        plt.xticks(range(len(heat_df.columns)), heat_df.columns, rotation=45)
        plt.yticks(range(len(heat_df.index)), heat_df.index)

        for i in range(heat_df.shape[0]):
            for j in range(heat_df.shape[1]):
                val = heat_df.iloc[i, j]
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.title(f"Domain-level prototype deviation ({self.scaling_method})")
        plt.tight_layout()
        plt.savefig(self.save_path / f"domain_deviation_heatmap_{self.scaling_method}.png", dpi=300)
        plt.close()

        heat_df.to_csv(
            self.save_path / f"domain_deviation_heatmap_{self.scaling_method}_values.csv"
        )

    # ========================================================
    # -------------------- Full Process ----------------------
    # ========================================================

    def process(
        self,
        top_k_global=10,
        plot_top_k=10,
        plot_scaled_heatmap=True
    ):
        """
        Run the full prototype deviation analysis pipeline.

        Returns
        -------
        dict
            Dictionary containing centroids, deviations, summaries,
            and representative feature tables.
        """
        print("\n========== Prototype Deviation Analysis START ==========")

        config = {
            "label_col": self.label_col,
            "reference_label": self.reference_label,
            "feature_keywords": self.feature_keywords,
            "center_method": self.center_method,
            "scaling_method": self.scaling_method,
            "min_scale": self.min_scale,
            "domain_top_k_by_scaled": self.domain_top_k_by_scaled,
            "representative_per_domain": self.representative_per_domain,
            "representative_abs_delta_ratio": self.representative_abs_delta_ratio,
        }
        self._save_json(config, self.save_path / "run_config.json")

        self.prepare_data()
        self.compute_prototypes()
        self.compute_deviation()
        self.summarize_global_top_features(top_k_global=top_k_global)
        self.summarize_domains()
        self.summarize_domain_representatives()

        # Explanation-layer visualizations
        for label in self.deviation_tables_.keys():
            self.plot_top_features_scaled_with_raw_labels(label, top_k=plot_top_k)

        self.plot_domain_heatmap_raw()

        # Optional mining-layer heatmap
        if plot_scaled_heatmap:
            self.plot_domain_heatmap_scaled()

        print("========== Prototype Deviation Analysis END ==========\n")

        return {
            "centroids": self.centroids_,
            "deviation_tables": self.deviation_tables_,
            "global_top_features": self.global_top_features_,
            "domain_summary": self.domain_summary_,
            "domain_candidate_features": self.domain_candidate_features_,
            "domain_representative_features": self.domain_representative_features_,
        }
    

# ============================================================
# ---------------- Multi-class Subgroup Discovery ------------
# ============================================================

class MultiClassSubgroupDiscovery:
    """
    Multi-class subgroup discovery using one-vs-rest strategy.

    For each class in label_col:
    - create binary target: target class vs all others
    - run subgroup discovery
    - prune redundant / nested subgroups
    - save top patterns
    - visualize top patterns
    """

    def __init__(
        self,
        df,
        label_col="label",
        save_path=Path("./subgroup_results"),
        result_set_size=20,
        depth=2,
        nbins=3,
        qf_name="WRAcc",
        min_label_count=3,
        prune_strategy="quality",   # "quality" / "precision" / "size"
        top_k_plot=5
    ):
        self.df_raw = df.copy()
        self.label_col = label_col
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.result_set_size = result_set_size
        self.depth = depth
        self.nbins = nbins
        self.qf_name = qf_name
        self.min_label_count = min_label_count
        self.prune_strategy = prune_strategy
        self.top_k_plot = top_k_plot

        self.df = None
        self.labels_ = None
        self.results_ = {}
        self.results_pruned_ = {}

    def prepare_data(self):
        """
        Clean input data for subgroup discovery.
        """
        self.df = clean_for_subgroup_discovery(
            self.df_raw,
            label_col=self.label_col
        )

        value_counts = self.df[self.label_col].value_counts()
        valid_labels = value_counts[value_counts >= self.min_label_count].index.tolist()

        if len(valid_labels) < 2:
            raise ValueError(
                f"Not enough valid labels after filtering. "
                f"Need at least 2 classes with >= {self.min_label_count} samples."
            )

        self.df = self.df[self.df[self.label_col].isin(valid_labels)].copy()
        self.labels_ = valid_labels

        self.df.to_csv(self.save_path / "subgroup_input_cleaned.csv", index=False)
        return self.df

    def _get_quality_function(self):
        """
        Select quality function for pysubgroup.
        """
        qf_name = self.qf_name.lower()

        if qf_name == "wracc":
            return ps.WRAccQF()
        elif qf_name == "lift":
            return ps.LiftQF()
        elif qf_name == "chi2":
            return ps.ChiSquaredQF()
        else:
            raise ValueError(f"Unsupported quality function: {self.qf_name}")

    def _parse_subgroup_conditions(self, subgroup_str):
        """
        Convert subgroup string into a set of atomic conditions.
        """
        if pd.isna(subgroup_str):
            return set()

        subgroup_str = str(subgroup_str).strip()

        if subgroup_str == "" or subgroup_str.lower() == "dataset":
            return set()

        parts = [x.strip() for x in subgroup_str.split("AND")]
        parts = [x for x in parts if x != ""]
        return set(parts)

    def _is_better_row(self, row_a, row_b, strategy="quality"):
        """
        Return True if row_a is better than row_b under the given strategy.
        """
        if strategy == "quality":
            if row_a["quality"] != row_b["quality"]:
                return row_a["quality"] > row_b["quality"]
        elif strategy == "precision":
            if row_a["precision_in_subgroup"] != row_b["precision_in_subgroup"]:
                return row_a["precision_in_subgroup"] > row_b["precision_in_subgroup"]
        elif strategy == "size":
            if row_a["subgroup_size"] != row_b["subgroup_size"]:
                return row_a["subgroup_size"] > row_b["subgroup_size"]

        # Tie-breakers
        if row_a["precision_in_subgroup"] != row_b["precision_in_subgroup"]:
            return row_a["precision_in_subgroup"] > row_b["precision_in_subgroup"]

        if row_a["subgroup_size"] != row_b["subgroup_size"]:
            return row_a["subgroup_size"] > row_b["subgroup_size"]

        # Prefer shorter / simpler subgroup if all above are equal
        len_a = len(row_a["conditions"])
        len_b = len(row_b["conditions"])
        if len_a != len_b:
            return len_a < len_b

        return False

    def _prune_subgroups(self, df):
        """
        Remove redundant nested subgroups.

        Rule:
        - If subgroup A is a subset of subgroup B (A ⊆ B), they are considered nested.
        - Keep the better one according to prune_strategy.
        """
        if df is None or df.empty:
            return df

        df_pruned = df.copy().reset_index(drop=True)
        df_pruned["conditions"] = df_pruned["subgroup"].apply(self._parse_subgroup_conditions)

        keep = [True] * len(df_pruned)

        for i in range(len(df_pruned)):
            if not keep[i]:
                continue

            for j in range(len(df_pruned)):
                if i == j or not keep[j]:
                    continue

                cond_i = df_pruned.loc[i, "conditions"]
                cond_j = df_pruned.loc[j, "conditions"]

                # Skip exact Dataset vs Dataset duplicates later handled naturally
                if cond_i == cond_j:
                    # Keep only the better one
                    if self._is_better_row(df_pruned.loc[j], df_pruned.loc[i], strategy=self.prune_strategy):
                        keep[i] = False
                        break
                    else:
                        keep[j] = False
                        continue

                # Check nested relationship
                if cond_i.issubset(cond_j) or cond_j.issubset(cond_i):
                    if self._is_better_row(df_pruned.loc[j], df_pruned.loc[i], strategy=self.prune_strategy):
                        keep[i] = False
                        break

        df_pruned = df_pruned[keep].copy().reset_index(drop=True)
        df_pruned = df_pruned.drop(columns=["conditions"], errors="ignore")

        # Final sorting
        df_pruned = df_pruned.sort_values(
            ["quality", "precision_in_subgroup", "subgroup_size"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        return df_pruned

    def _plot_top_patterns_for_label(self, df_label, target_label, top_k=5, metric="quality"):
        """
        Plot top-k subgroup patterns for one target label.
        """
        if df_label is None or df_label.empty:
            return

        plot_dir = self.save_path / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df_plot = df_label.sort_values(
            [metric, "precision_in_subgroup", "subgroup_size"],
            ascending=[False, False, False]
        ).head(top_k).copy()

        if df_plot.empty:
            return

        # Reverse order for horizontal bar chart (best at top visually after invert)
        df_plot = df_plot.iloc[::-1].copy()

        labels = df_plot["subgroup"].astype(str).tolist()
        values = df_plot[metric].tolist()

        # Dynamic figure height
        fig_height = max(4, 0.8 * len(df_plot) + 1.5)

        plt.figure(figsize=(12, fig_height))
        bars = plt.barh(range(len(df_plot)), values)

        plt.yticks(range(len(df_plot)), labels)
        plt.xlabel(metric)
        plt.title(f"Top {min(top_k, len(df_plot))} subgroup patterns for {target_label}")

        # Add annotation: lift + size + precision
        for i, (_, row) in enumerate(df_plot.iterrows()):
            text = (
                f"lift={row['lift']:.2f}, "
                f"size={int(row['subgroup_size'])}, "
                f"prec={row['precision_in_subgroup']:.2f}"
            )
            plt.text(
                row[metric],
                i,
                f"  {text}",
                va="center",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(plot_dir / f"top_{top_k}_{target_label}_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_all_top_patterns(self, metric="quality"):
        """
        Plot top subgroup charts for all labels.
        """
        for label, df_label in self.results_pruned_.items():
            self._plot_top_patterns_for_label(
                df_label=df_label,
                target_label=label,
                top_k=self.top_k_plot,
                metric=metric
            )

    def _run_one_vs_rest(self, target_label):
        df_target = self.df.copy()

        # Build binary target on original cleaned data
        df_target["target_binary"] = (
            df_target[self.label_col].astype(str).str.strip() == str(target_label).strip()
        ).astype(int)

        total_n = len(df_target)
        positive_n = int(df_target["target_binary"].sum())

        if positive_n < self.min_label_count:
            print(f"[Skip] Label '{target_label}' has too few positive samples: {positive_n}")
            return None

        # Step 1: adaptive discretization
        debug_dir = self.save_path / f"debug_{target_label}"
        debug_dir.mkdir(parents=True, exist_ok=True)

        df_disc, strategy_df, dropped_df = adaptive_discretize_for_subgroups(
            df=df_target,
            label_col="target_binary",
            feature_keywords=["EEG_", "ECG_", "EDA_", "EYE_", "HEAD_", "PRE_", "SIM_"],
            n_bins=4,
            low_info_filter=True,
            dominant_ratio=0.9,
            min_rare_count=3,
            min_valid_ratio=0.5,
            min_unique_values=5,
            max_zero_ratio=0.95,
            min_iqr=1e-8,
            keep_non_numeric=True,
            save_path=debug_dir
        )

        # Step 2: build searchspace from discretized data
        searchspace, summary_df, selector_df = build_searchspace_from_discretized(
            df_disc=df_disc,
            label_col="target_binary",
            ignore_cols=[self.label_col],
            max_nominal_values=12,
            min_selector_coverage=3,
            drop_missing_like=True,
            verbose=False
        )

        summary_df.to_csv(debug_dir / "searchspace_summary.csv", index=False)
        selector_df.to_csv(debug_dir / "searchspace_selectors.csv", index=False)

        if len(searchspace) == 0:
            print(f"[Error] Empty searchspace for target label: {target_label}")
            return None

        # Step 3: IMPORTANT — target and task must use df_disc
        target = ps.BinaryTarget("target_binary", 1)
        qf = self._get_quality_function()

        task = ps.SubgroupDiscoveryTask(
            data=df_disc,
            target=target,
            search_space=searchspace,
            result_set_size=self.result_set_size,
            depth=self.depth,
            qf=qf
        )

        result = ps.BeamSearch().execute(task)

        result_df = subgroup_result_to_dataframe(
            result=result,
            target_label=target_label,
            total_n=total_n,
            positive_n=positive_n
        )

        # Save raw results
        result_df.to_csv(
            self.save_path / f"subgroups_{str(target_label)}_vs_rest_raw.csv",
            index=False
        )

        # Prune redundant nested subgroups
        result_df_pruned = self._prune_subgroups(result_df)

        # Re-rank after pruning
        if result_df_pruned is not None and not result_df_pruned.empty:
            result_df_pruned = result_df_pruned.reset_index(drop=True)
            result_df_pruned["rank"] = np.arange(1, len(result_df_pruned) + 1)

        result_df_pruned.to_csv(
            self.save_path / f"subgroups_{str(target_label)}_vs_rest_pruned.csv",
            index=False
        )

        return result_df, result_df_pruned

    def run(self):
        """
        Run multi-class subgroup discovery using one-vs-rest.
        """
        self.prepare_data()

        """
        print("\nColumn dtypes after cleaning:")
        print(self.df.dtypes.sort_values())

        print("\nPreview of cleaned data:")
        print(self.df.head())

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        print(f"\nNumeric columns: {len(numeric_cols)}")
        print(numeric_cols[:20])

        print(f"\nCategorical columns: {len(object_cols)}")
        print(object_cols[:20])
        """

        all_results_raw = []
        all_results_pruned = []

        for label in self.labels_:
            print(f"\n[Subgroup Discovery] Running one-vs-rest for label: {label}")
            outputs = self._run_one_vs_rest(label)

            if outputs is None:
                continue

            result_df_raw, result_df_pruned = outputs

            if result_df_raw is not None and not result_df_raw.empty:
                self.results_[label] = result_df_raw
                all_results_raw.append(result_df_raw)

            if result_df_pruned is not None and not result_df_pruned.empty:
                self.results_pruned_[label] = result_df_pruned
                all_results_pruned.append(result_df_pruned)

        if len(all_results_raw) == 0:
            print("[Warning] No subgroup results found.")
            return {}

        # Save all raw results
        all_results_raw_df = pd.concat(all_results_raw, axis=0, ignore_index=True)
        all_results_raw_df.to_csv(self.save_path / "all_subgroup_results_raw.csv", index=False)

        # Save all pruned results
        if len(all_results_pruned) > 0:
            all_results_pruned_df = pd.concat(all_results_pruned, axis=0, ignore_index=True)
            all_results_pruned_df.to_csv(self.save_path / "all_subgroup_results_pruned.csv", index=False)

            summary_df = (
                all_results_pruned_df
                .sort_values(["target_label", "quality"], ascending=[True, False])
                .groupby("target_label", group_keys=False)
                .head(self.top_k_plot)
                .reset_index(drop=True)
            )
            summary_df.to_csv(self.save_path / f"top{self.top_k_plot}_subgroup_summary_pruned.csv", index=False)

            # Generate plots
            self._plot_all_top_patterns(metric="quality")

        return self.results_pruned_ if len(self.results_pruned_) > 0 else self.results_

# ============================================================
# ------------------- Run on your current data ---------------
# ============================================================

# Make sure your data has already been loaded and EEG aggregated:
# data = aggregate_eeg_if_needed(data)

if __name__ == "__main__":
    
    from feature_extraction.DatabaseConnection import get_latest_output_save_path
    latest_time, data_path = get_latest_output_save_path(output_root=Path("./output"))

    #data_name = 'all_features'
    data_name = 'scenario_2_features'

    feature_path = data_path / "data" / f"{data_name}.csv"
    data = pd.read_csv(feature_path, encoding='latin1')

    save_path = data_path / f'Pattern_{data_name}'

    data = aggregate_eeg_if_needed(data)

    prototype_runner = PrototypeDeviationAnalysis(
        df=data,
        label_col='label',
        reference_label='Health',
        feature_keywords=['EEG_', 'ECG_', 'EDA_', 'EYE_', 'HEAD_', 'PRE_', 'SIM_'],
        save_path=save_path / 'Prototype Deviation',
        center_method='mean',
        scaling_method='std',
        domain_top_k_by_scaled=5,
        representative_per_domain=1,
        representative_abs_delta_ratio=0.7,
    )

    prototype_results = prototype_runner.process(
    top_k_global=10,
    plot_top_k=10,
    plot_scaled_heatmap=True
    )

    subgroup_runner = MultiClassSubgroupDiscovery(
        df=data,
        label_col="label",
        save_path=save_path/ 'Subgroup Discovery',
        result_set_size=20,
        depth=3,
        nbins=4,
        qf_name="WRAcc",   # "WRAcc" / "Lift" / "Chi2"
        min_label_count=3
    )

    subgroup_results = subgroup_runner.run()



