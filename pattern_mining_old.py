import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from feature_extraction.DatabaseConnection import myDatabase
from itertools import combinations, product
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


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
    if isinstance(mode_val, float):
        mode_str = f"{mode_val:.3f}"
    else:
        mode_str = str(mode_val)
    return np.where(s == mode_val, f"eq_{mode_str}", f"neq_{mode_str}")

def build_interval_labels(bins, n_bins):
    """
    Build human-readable labels with semantic tags.
    """
    if n_bins == 2:
        sem = ["low", "high"]
    elif n_bins == 3:
        sem = ["low", "mid", "high"]
    else:
        sem = [f"bin{i}" for i in range(len(bins) - 1)]

    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        labels.append(f"{sem[i]}({lo:.3f}-{hi:.3f})")

    return labels

# ============================================================
# ---------------- Pattern Mining Pipeline -------------------
# ============================================================

class PatternMining:

    def __init__(
        self,
        ante_keywords,
        cons_keywords,
        label_col,
        label_map=None,
        label_mode="binary_health_vs_others",
        save_path=None,
    ):

        self.ante_keywords = ante_keywords
        self.cons_keywords = cons_keywords
        self.label_col = label_col
        self.label_map = label_map
        self.label_mode = label_mode

        self.save_path = Path(save_path) if save_path else Path("./tempPattern")
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.data = None
        self.data_disc = None
        self.df_trans = None
        self.labels = None
        self.freq_by_domain = None
        self.antecedents = None
        self.rules_phase3 = None
        self.rules_final = None


    # ========================================================
    # -------------------- Phase 0 ---------------------------
    # ========================================================

    def prepare_data(self, df):

        df = df.copy()

        # Default label strategy:
        #   Health -> 0
        #   all other classes -> 1
        #
        # Future extension:
        #   this block can be replaced by a multiclass mapping, e.g.
        #   Health -> 0, other labels -> 1, 2, 3, ...
        if self.label_map is not None:
            df[self.label_col] = df[self.label_col].replace(self.label_map)
        elif self.label_mode == "binary_health_vs_others":
            df[self.label_col] = df[self.label_col].apply(
                lambda x: 0 if str(x).strip() == "Health" else 1
            )
        else:
            raise ValueError(f"Unsupported label_mode: {self.label_mode}")

        ante_cols = [c for c in df.columns if any(k in c for k in self.ante_keywords)]
        cons_cols = [c for c in df.columns if any(k in c for k in self.cons_keywords)]

        ordered_cols = ante_cols + cons_cols + [self.label_col]
        ordered_cols = list(dict.fromkeys(ordered_cols))

        self.data = df[ordered_cols].copy()

        print(f"[Phase 0] Ante cols: {len(ante_cols)}, Cons cols: {len(cons_cols)}")
        print(f"[Phase 0] Label mode: {self.label_mode}")

        return self.data

    # ========================================================
    # -------------------- Phase 0.5 -------------------------
    # ========================================================

    def discretize(
        self,
        n_bins=3,
        dominant_ratio=0.9,
        min_rare_count=3,
        low_info_filter_ante=True,
        low_info_filter_cons=False,
        min_valid_ratio=0.5,
        min_unique_values=5,
        max_zero_ratio=0.95,
        min_iqr=1e-8,
    ):
        """
        Adaptive discretization with optional low-information filtering.

        If low_info_filter=True:
            Apply light sparsity / low-variance filtering (recommended for antecedents).

        - ANTE: low-info filtering enabled
        - CONS: low-info filtering disabled
        """

        print("\n[Phase 0.5] Advanced discretization...")

        df = self.data.copy()

        ante_cols = [c for c in df.columns if any(k in c for k in self.ante_keywords)]
        cons_cols = [c for c in df.columns if any(k in c for k in self.cons_keywords)]

        dropped = []
        strategy = {}
        N = len(df)

        # =====================================================
        # Internal adaptive discretization logic
        # =====================================================

        def process_columns(cols, low_info_filter):

            nonlocal df, dropped, strategy

            for col in cols:

                if col not in df.columns:
                    continue

                s = df[col]

                # ---------------- Basic sanity ----------------
                if s.dropna().empty:
                    dropped.append(col)
                    strategy[col] = "dropped_all_nan"
                    continue

                if s.nunique(dropna=True) < 2:
                    dropped.append(col)
                    strategy[col] = "dropped_constant"
                    continue

                if not pd.api.types.is_numeric_dtype(s):
                    strategy[col] = "skip_non_numeric"
                    continue

                valid_n = s.notna().sum()
                valid_ratio = valid_n / N

                # ---------------- Low information filtering ----------------
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

                    zero_ratio = (s == 0).sum() / valid_n
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
                        df[col] = np.where(s.fillna(0) == 0, "zero", "nonzero")
                        strategy[col] = "zero_binary"
                    else:
                        df[col] = apply_mode_binary(s, mode_val)
                        strategy[col] = "mode_binary"

                    continue

                # ---------------- qcut path ----------------
                try:
                    _, bins = pd.qcut(s, q=n_bins, duplicates="drop", retbins=True)
                    bins = np.unique(bins)

                    if len(bins) < 3:
                        raise ValueError("Degenerate bins")

                    labels = build_interval_labels(bins, n_bins)

                    df[col] = pd.cut(
                        s,
                        bins=bins,
                        labels=labels,
                        include_lowest=True
                    )

                    if df[col].nunique(dropna=True) < 2:
                        if rare_count >= min_rare_count:
                            df[col] = apply_mode_binary(s, mode_val)
                            strategy[col] = "mode_binary_fallback"
                        else:
                            dropped.append(col)
                            strategy[col] = "dropped_qcut_degenerate"
                    else:
                        strategy[col] = "qcut_interval"

                except Exception:
                    if rare_count >= min_rare_count:
                        df[col] = apply_mode_binary(s, mode_val)
                        strategy[col] = "mode_binary_fallback"
                    else:
                        dropped.append(col)
                        strategy[col] = "dropped_qcut_error"

        # =====================================================
        # Apply to ANTE and CONS separately
        # =====================================================

        process_columns(ante_cols, low_info_filter_ante)
        process_columns(cons_cols, low_info_filter_cons)

        # Drop columns at the end
        if dropped:
            df = df.drop(columns=dropped)

        self.data_disc = df

        # ---------------- Save artifacts ----------------
        df.to_csv(self.save_path / "data_discretized.csv", index=False)

        pd.DataFrame({
            "column": list(strategy.keys()),
            "strategy": list(strategy.values())
        }).to_csv(
            self.save_path / "discretization_strategy.csv",
            index=False
        )

        if dropped:
            pd.DataFrame({"dropped_columns": dropped}).to_csv(
                self.save_path / "discretization_dropped.csv",
                index=False
            )

        #print(f"[Phase 0.5] Finished. " f"Remaining columns: {df.shape[1]} | " f"Dropped: {len(dropped)}")

        return df


    # ========================================================
    # -------------------- Phase 0.6 -------------------------
    # ========================================================

    def build_transactions(self):

        transactions = []
        labels = []

        ante_cols = [c for c in self.data_disc.columns if any(k in c for k in self.ante_keywords)]
        cons_cols = [c for c in self.data_disc.columns if any(k in c for k in self.cons_keywords)]

        for _, row in self.data_disc.iterrows():

            items = []

            for c in ante_cols:
                items.append(f"ANTE::{c}={row[c]}")

            for c in cons_cols:
                items.append(f"CONS::{c}={row[c]}")

            transactions.append(sorted(items))
            labels.append(row[self.label_col])

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)

        df_trans = pd.DataFrame(te_ary, columns=te.columns_)
        df_trans = df_trans.reindex(sorted(df_trans.columns), axis=1)

        self.df_trans = df_trans
        self.labels = labels

        #print(f"[Phase 0.6] Transactions built: {df_trans.shape}")

        return df_trans, labels


    # ========================================================
    # -------------------- Phase 1 ---------------------------
    # ========================================================

    def mine_domain_itemsets(
        self,
        min_support=0.2,
        max_len=2,
        max_itemsets_per_domain=50,
    ):

        print("[Phase 1] Mining domain frequent itemsets...")

        domain_cols = {k: [] for k in self.ante_keywords}

        for col in self.df_trans.columns:
            if not col.startswith("ANTE::"):
                continue
            for k in self.ante_keywords:
                if f"ANTE::{k}" in col:
                    domain_cols[k].append(col)

        freq_by_domain = {}

        for domain, cols in domain_cols.items():

            if len(cols) == 0:
                continue

            freq = fpgrowth(
                self.df_trans[cols],
                min_support=min_support,
                max_len=max_len,
                use_colnames=True
            )

            if freq.empty:
                freq_by_domain[domain] = freq
                continue

            # -------------------------------------------------
            # Hard cap to control combinatorial explosion
            # -------------------------------------------------
            if max_itemsets_per_domain is not None:

                freq["length"] = freq["itemsets"].apply(len)

                freq_len2 = freq[freq["length"] == 2]
                freq_len1 = freq[freq["length"] == 1]

                # 70% 2-item, 30% 1-item 
                k2 = int(max_itemsets_per_domain * 0.7)
                k1 = max_itemsets_per_domain - k2

                freq = pd.concat([
                    freq_len2.sort_values(
                        ["support", "itemsets"],
                        ascending=[False, True]
                    ).head(k2),

                    freq_len1.sort_values(
                        ["support", "itemsets"],
                        ascending=[False, True]
                    ).head(k1)
                ])

            freq = freq.reset_index(drop=True)

            freq_by_domain[domain] = freq

            freq.to_csv(
                self.save_path / f"phase1_freq_{domain}.csv",
                index=False
            )

            print(
                f"  {domain}: kept {len(freq)} itemsets "
                f"(cap={max_itemsets_per_domain})"
            )

        self.freq_by_domain = freq_by_domain
        return freq_by_domain

    # ========================================================
    # -------------------- Phase 2 ---------------------------
    # ========================================================

    def build_antecedents(
        self,
        min_domains=1,
        max_domains=2,
        max_items=4,
        min_ante_support=0.15
    ):

        print("[Phase 2] Building antecedents...")

        antecedents = []

        domains = list(self.freq_by_domain.keys())

        if len(domains) == 0:
            print("[Phase 2] No valid domains found.")
            self.antecedents = []
            return []

        for n_dom in range(min_domains, max_domains + 1):

            for doms in combinations(domains, n_dom):

                # collect itemsets safely
                itemset_lists = []
                for d in doms:
                    if d not in self.freq_by_domain:
                        continue
                    df_dom = self.freq_by_domain[d]
                    if df_dom is None or df_dom.empty:
                        continue
                    itemset_lists.append(df_dom["itemsets"])

                if len(itemset_lists) != len(doms):
                    continue

                for combo in product(*itemset_lists):

                    union_set = frozenset().union(*combo)

                    if len(union_set) > max_items:
                        continue

                    sup = self.df_trans[list(union_set)].all(axis=1).mean()

                    if sup < min_ante_support:
                        continue

                    antecedents.append(union_set)

        self.antecedents = sorted(set(antecedents))

        print(f"[Phase 2] Total antecedents: {len(self.antecedents)}")

        ante_df = pd.DataFrame({
            "antecedent": [tuple(sorted(a)) for a in self.antecedents],
            "length": [len(a) for a in self.antecedents]
        })

        return self.antecedents,ante_df


    # ========================================================
    # -------------------- Phase 3 ---------------------------
    # ========================================================

    def evaluate_rules(
        self,
        target_label=1,
        ref_label=0,
        min_ante_count=5,
        min_conf=0.6,
        min_disc=0.3,
        prune_eps=0.05
    ):
        """
        Phase 3:
        1 Generate raw discriminative rules
        2 Sort
        3 Prune redundancy
        """

        print("[Phase 3] Evaluating discriminative rules...")

        raw_rules = self._generate_raw_rules(
            target_label=target_label,
            ref_label=ref_label,
            min_ante_count=min_ante_count,
            min_conf=min_conf,
            min_disc=min_disc
        )

        if raw_rules.empty:
            print("[Phase 3] No rules found.")
            self.rules_phase3 = raw_rules
            return raw_rules

        # Sort rules
        raw_rules = raw_rules.sort_values(
            ["disc_score", "conf_target", "ante_count_target"],
            ascending=False
        ).reset_index(drop=True)

        print(f"[Phase 3] Raw rules: {len(raw_rules)}")

        # Prune redundancy
        pruned_rules = self._prune_redundant_rules(
            raw_rules,
            eps=prune_eps
        )

        print(f"[Phase 3] After pruning: {len(pruned_rules)}")

        pruned_rules = self._tag_domain_complexity(pruned_rules)

        self.rules_phase3 = pruned_rules

        return pruned_rules


    # ============================================================
    # ----------- Core Raw Rule Generation -----------------------
    # ============================================================

    def _generate_raw_rules(
        self,
        target_label,
        ref_label,
        min_ante_count,
        min_conf,
        min_disc
    ):
        """
        Generate raw phy -> sim rules without pruning.
        """

        results = []

        labels = pd.Series(self.labels)
        mask_t = labels == target_label
        mask_r = labels == ref_label

        cons_items = [
            c for c in self.df_trans.columns
            if c.startswith("CONS::")
        ]

        # Cache consequent masks (performance optimization)
        cons_masks = {c: self.df_trans[c] for c in cons_items}

        for ante in self.antecedents:

            ante_cols = list(ante)
            mask_ante = self.df_trans[ante_cols].all(axis=1)

            mask_ante_t = mask_ante & mask_t
            mask_ante_r = mask_ante & mask_r

            ante_count_t = int(mask_ante_t.sum())
            if ante_count_t < min_ante_count:
                continue

            ante_count_r = int(mask_ante_r.sum())

            for cons, cons_mask in cons_masks.items():

                score = self._compute_rule_score(
                    mask_ante_t,
                    mask_ante_r,
                    cons_mask,
                    ante_count_t,
                    ante_count_r,
                    min_conf,
                    min_disc
                )

                if score is None:
                    continue

                results.append({
                    "antecedent": tuple(sorted(ante)),
                    "consequent": cons,
                    "ante_count_target": ante_count_t,
                    "conf_target": score["conf_target"],
                    "conf_ref": score["conf_ref"],
                    "disc_score": score["disc_score"],
                })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)


    # ============================================================
    # ----------- Core Rule Scoring Logic ------------------------
    # ============================================================

    def _compute_rule_score(
        self,
        mask_ante_t,
        mask_ante_r,
        cons_mask,
        ante_count_t,
        ante_count_r,
        min_conf,
        min_disc
    ):
        """
        Core discriminative scoring logic.

        This function can be modified in future:
        - Replace disc_score
        - Add lift
        - Add odds ratio
        - Add statistical testing
        """

        both_t = int((mask_ante_t & cons_mask).sum())
        conf_t = both_t / ante_count_t

        if conf_t < min_conf:
            return None

        if ante_count_r > 0:
            both_r = int((mask_ante_r & cons_mask).sum())
            conf_r = both_r / ante_count_r
        else:
            conf_r = 0.0

        disc = conf_t - conf_r

        if disc < min_disc:
            return None

        return {
            "conf_target": conf_t,
            "conf_ref": conf_r,
            "disc_score": disc
        }


    # ============================================================
    # -------------------- Redundancy Pruning --------------------
    # ============================================================

    def _prune_redundant_rules(self, rules, eps=0.05):
        """
        Remove redundant rules:
        If A ⊃ B and similar confidence → remove A
        """

        kept = []

        for _, r in rules.iterrows():

            A = set(r["antecedent"])
            conf = r["conf_target"]

            redundant = False

            for k in kept:
                B = set(k["antecedent"])
                if B.issubset(A) and abs(conf - k["conf_target"]) < eps:
                    redundant = True
                    break

            if not redundant:
                kept.append(r)

        return pd.DataFrame(kept).reset_index(drop=True)


    def visualize_heatmap(
        self,
        rules_df,
        filename="heatmap.png",
        alpha=0.7,
        annotate=True
    ):
        """
        Generate integrated heatmap:
        score = alpha * disc_score + (1-alpha) * ante_count_target

        Parameters
        ----------
        rules_df : pd.DataFrame
            Rules dataframe (must contain disc_score, ante_count_target, consequent, domains)

        filename : str
            Output image filename

        alpha : float
            Weight for disc_score (default=0.7)

        annotate : bool
            Whether to show values in heatmap
        """

        if rules_df is None or rules_df.empty:
            print("[Heatmap] No rules to visualize.")
            return

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        df = rules_df.copy()

        # ----------------------------
        # 1. Clean consequent label
        # ----------------------------
        df["consequent_clean"] = (
            df["consequent"]
            .str.replace("CONS::SIM_", "", regex=False)
            .str.split("=")
            .str[0]
        )

        # ----------------------------
        # 2. Domain combo
        # ----------------------------
        df["domain"] = df["domains"].apply(lambda x: "+".join(sorted(x)))

        # ----------------------------
        # 3. Normalize
        # ----------------------------
        d = df["disc_score"]
        c = df["ante_count_target"]

        df["disc_n"] = (d - d.min()) / (d.max() - d.min() + 1e-8)
        df["count_n"] = (c - c.min()) / (c.max() - c.min() + 1e-8)

        # ----------------------------
        # 4. Integrated score
        # ----------------------------
        df["score"] = alpha * df["disc_n"] + (1 - alpha) * df["count_n"]

        # ----------------------------
        # 5. Pivot table
        # ----------------------------
        heat = (
            df.groupby(["domain", "consequent_clean"])["score"]
            .mean()
            .unstack()
            .fillna(0)
        )

        # ----------------------------
        # 6. Plot
        # ----------------------------
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "white_red", ["white", "darkred"]
        )

        plt.figure(figsize=(8, 6))
        plt.imshow(heat.values, cmap=cmap, vmin=0, vmax=1)

        plt.xticks(range(len(heat.columns)), heat.columns, rotation=45)
        plt.yticks(range(len(heat.index)), heat.index)

        # annotate values
        if annotate:
            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    val = heat.iloc[i, j]
                    if abs(val) < 1e-6:
                        continue
                    plt.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color="black" if val < 0.5 else "white",
                        fontsize=8
                    )

        plt.colorbar()
        plt.tight_layout()

        save_file = self.save_path / filename
        plt.savefig(save_file, dpi=300)
        plt.close()

        print(f"[Heatmap] Saved: {save_file}")

    # ========================================================
    # -------------------- Phase 4 ---------------------------
    # ========================================================

    def heuristic_filtering(
        self,
        conf_thresholds=None,
        disc_thresholds=None,
        top_k_domain=3,
        top_k_consequent=5
    ):
        """
        Phase 4: Structural analysis and quality control of rules.

        This module performs:
        1. Domain structure tagging
        2. Domain-level statistics
        3. Layered filtering (adaptive to domain complexity)
        4. Representative rule selection
        """

        print("[Phase 4] Heuristic filtering rules...")

        if self.rules_phase3 is None or self.rules_phase3.empty:
            print("[Phase 4] No rules to process.")
            return None

        rules = self.rules_phase3.copy()

        self._compute_domain_stats(rules)

        rules = self._layered_filtering(
            rules,
            conf_thresholds=conf_thresholds,
            disc_thresholds=disc_thresholds
        )

        rules = self._select_representative_rules(
            rules,
            group_col="domain_combo",
            top_k=top_k_domain
        )

        rules = self._select_representative_rules(
            rules,
            group_col="consequent",
            top_k=top_k_consequent
        )

        self.rules_final = rules

        print(f"[Phase 4] Final rules: {len(rules)}")

        return rules

    # --------------------------------------------------------
    # Phase 4.1 — Domain Structure Tagging
    # --------------------------------------------------------
    def _tag_domain_complexity(self, rules):

        rules = rules.copy()

        def extract_domains(ante):
            return sorted(
                set(x.split("::")[1].split("_")[0] for x in ante)
            )

        rules["domains"] = rules["antecedent"].apply(extract_domains)
        rules["n_domains"] = rules["domains"].apply(len)
        rules["domain_combo"] = rules["domains"].apply(lambda x: "-".join(x))

        print(
            "[Phase 4.1] Domain levels:",
            sorted(rules["n_domains"].unique())
        )

        return rules

    # --------------------------------------------------------
    # Phase 4.2 — Domain Statistics
    # --------------------------------------------------------
    def _compute_domain_stats(self, rules):

        df_stats = (
            rules
            .groupby(["n_domains", "domain_combo"])
            .size()
            .reset_index(name="rule_count")
            .sort_values(["n_domains", "rule_count"], ascending=[True, False])
        )

        df_stats.to_csv(
            self.save_path / "phase4_domain_statistics.csv",
            index=False
        )

        print(f"[Phase 4.2] Domain combinations: {len(df_stats)}")

    # --------------------------------------------------------
    # Phase 4.3 — Layered Filtering
    # --------------------------------------------------------
    def _layered_filtering(
        self,
        rules,
        conf_thresholds=None,
        disc_thresholds=None
    ):

        print("[Phase 4.3] Layered filtering...")

        rules = rules.copy()

        domain_levels = sorted(rules["n_domains"].unique())

        filtered = []

        for n_dom in domain_levels:

            subset = rules[rules["n_domains"] == n_dom]

            if conf_thresholds and n_dom in conf_thresholds:
                subset = subset[
                    subset["conf_target"] >= conf_thresholds[n_dom]
                ]

            if disc_thresholds and n_dom in disc_thresholds:
                subset = subset[
                    subset["disc_score"] >= disc_thresholds[n_dom]
                ]

            print(f"  n_domains={n_dom}: {len(subset)}")

            filtered.append(subset)

        if filtered:
            return pd.concat(filtered).reset_index(drop=True)
        else:
            return pd.DataFrame()

    # --------------------------------------------------------
    # Phase 4.4 — Representative Rule Selection
    # --------------------------------------------------------
    def _select_representative_rules(
        self,
        rules,
        group_col,
        top_k=3,
        prioritize_length=True
    ):

        print(f"[Phase 4.4] Selecting by {group_col}...")

        rules = rules.copy()

        rules["ante_len"] = rules["antecedent"].apply(len)

        if prioritize_length:
            rules = rules.sort_values(
                ["ante_len", "disc_score", "conf_target"],
                ascending=[False, False, False]
            )
        else:
            rules = rules.sort_values(
                ["disc_score", "conf_target"],
                ascending=[False, False]
            )

        rules_rep = (
            rules
            .groupby(group_col)
            .head(top_k)
            .reset_index(drop=True)
        )

        print(f"[Phase 4.4] After selection: {len(rules_rep)}")

        return rules_rep


    # ========================================================
    # -------------------- Full Process ----------------------
    # ========================================================

    def process(self, df):

        print("\n========== Pattern Mining Pipeline START ==========")

        config = {
            "ante_keywords": self.ante_keywords,
            "cons_keywords": self.cons_keywords,
            "label_col": self.label_col
        }

        with open(self.save_path / "run_config.json", "w") as f:
            json.dump(config, f, indent=4)

        self.prepare_data(df)
        self.discretize()
        self.build_transactions()

        self.mine_domain_itemsets()

        antecedents,ante_df = self.build_antecedents()
        ante_df.to_csv(
            self.save_path / "phase2_antecedents.csv",
            index=False
        )

        pruned_rules = self.evaluate_rules()
        self.visualize_heatmap(pruned_rules,filename="phase3_heatmap.png")
        pruned_rules.to_csv(
            self.save_path / "phase3_rules_pruned.csv",
            index=False
        )

        rules_final = self.heuristic_filtering(
            conf_thresholds={1:0.7},
            disc_thresholds={2:0.4}
        )
        self.visualize_heatmap(rules_final,filename="final_heatmap.png")
        rules_final.to_csv(
            self.save_path / "rules_final.csv",
            index=False
        )

        print("\n========== Pattern Mining Pipeline END ==========\n")

        return self.rules_final

    def subprocess(
        self,
        sub_ante_keywords,
        sub_cons_keywords,
        min_domains=1,
        max_domains=2,
        max_items=4,
        min_ante_support=0.15,
        target_label=1,
        ref_label=0,
        min_ante_count=5,
        min_conf=0.6,
        min_disc=0.3,
    ):
        """
        Sub-process rule mining on selected antecedent and consequent domains.

        This reuses:
        - discretized data
        - built transactions
        - mined frequent itemsets (Phase 1)

        Only domain selection and rule evaluation are redone.
        """

        print("\n========== Subprocess START ==========")

        ante_name = "+".join(sub_ante_keywords).lower()
        cons_name = "+".join(sub_cons_keywords).lower()

        subfolder = self.save_path / f"{ante_name} to {cons_name}"
        subfolder.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------
        # Restrict frequent itemsets by domain
        # ---------------------------------------
        sub_freq = {
            d: self.freq_by_domain[d]
            for d in sub_ante_keywords
            if d in self.freq_by_domain
        }

        if len(sub_freq) == 0:
            print("No valid sub domains found.")
            return None

        # Backup original freq_by_domain
        original_freq = self.freq_by_domain
        self.freq_by_domain = sub_freq

        # ---------------------------------------
        # Rebuild antecedents (restricted domains)
        # ---------------------------------------
        sub_antecedents,ante_df = self.build_antecedents(
            min_domains=min_domains,
            max_domains=max_domains,
            max_items=max_items,
            min_ante_support=min_ante_support
        )


        if len(sub_antecedents) == 0:
            print("No antecedents generated.")
            self.freq_by_domain = original_freq
            return None
        
        ante_df.to_csv(
            subfolder / "phase2_antecedents.csv",
            index=False
        )

        # ---------------------------------------
        # Restrict consequents
        # ---------------------------------------
        sub_cons_items = [
            c for c in self.df_trans.columns
            if c.startswith("CONS::")
            and any(k in c for k in sub_cons_keywords)
        ]

        if len(sub_cons_items) == 0:
            print("No valid consequents found.")
            self.freq_by_domain = original_freq
            return None

        # Temporarily override consequent search
        original_df_trans = self.df_trans.copy()

        # Create a reduced df_trans with only allowed consequents
        cols_to_keep = [
            c for c in self.df_trans.columns
            if not c.startswith("CONS::")
            or c in sub_cons_items
        ]

        self.df_trans = self.df_trans[cols_to_keep]

        # ---------------------------------------
        # Evaluate rules
        # ---------------------------------------
        sub_rules = self.evaluate_rules(
            target_label=target_label,
            ref_label=ref_label,
            min_ante_count=min_ante_count,
            min_conf=min_conf,
            min_disc=min_disc
        )

        if sub_rules is None or sub_rules.empty:
            print("No rules found in subprocess.")
            self.freq_by_domain = original_freq
            self.df_trans = original_df_trans
            return None
        
        sub_rules.to_csv(
            subfolder / "phase3_rules_pruned.csv",
            index=False
        )


        # ---------------------------------------
        # Heuristic Filtering
        # ---------------------------------------

        rules_final = self.heuristic_filtering(
            conf_thresholds={1:0.7},
            disc_thresholds={2:0.4}
        )

        if rules_final is None or rules_final.empty:
            print("No rules found in final.")
            self.freq_by_domain = original_freq
            self.df_trans = original_df_trans
            return None

        rules_final.to_csv(
            subfolder / "rules_final.csv",
            index=False
        )

        # Restore original state
        self.freq_by_domain = original_freq
        self.df_trans = original_df_trans

        print("========== Subprocess END ==========\n")

        return sub_rules


# for test

if __name__ == "__main__":

    from feature_extraction.DatabaseConnection import get_latest_output_save_path
    latest_time, data_path = get_latest_output_save_path(output_root=Path("./output"))

    data_name = 'scenario_1.1_features'

    feature_path = data_path / "data" / f"{data_name}.csv"
    data = pd.read_csv(feature_path, encoding='latin1')

    save_path = data_path / f'Pattern_{data_name}'

    data = aggregate_eeg_if_needed(data)

    pipeline = PatternMining(
        ante_keywords=['EYE','HEAD','ECG','EEG','EDA'],
        cons_keywords=['SIM'],
        label_col='label',
        label_mode='binary_health_vs_others',
        save_path=save_path
    )

    rules = pipeline.process(data)

"""
    pipeline.subprocess(sub_ante_keywords=['EYE'],sub_cons_keywords=['SIM'])

    pipeline.subprocess(sub_ante_keywords=['ECG'],sub_cons_keywords=['SIM'])

    #pipeline.subprocess(sub_ante_keywords=['HEAD'],sub_cons_keywords=['SIM'])

    pipeline.subprocess(sub_ante_keywords=['EDA'],sub_cons_keywords=['SIM'])

    pipeline.subprocess(sub_ante_keywords=['EYE','HEAD'],sub_cons_keywords=['SIM'])

    #pipeline.subprocess(sub_ante_keywords=['EYE','ECG'],sub_cons_keywords=['SIM'])

    pipeline.subprocess(sub_ante_keywords=['EYE','EDA'],sub_cons_keywords=['SIM'])

    #pipeline.subprocess(sub_ante_keywords=['EYE','ECG','EDA'],sub_cons_keywords=['SIM'])

    pipeline.subprocess(sub_ante_keywords=['ECG','EDA'],sub_cons_keywords=['SIM'])

"""