from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.model_selection import ParameterGrid
from pandas import ExcelWriter

from sklearn.tree import DecisionTreeClassifier, export_text

import hdbscan

from feature_extraction.DatabaseConnection import myDatabase
from feature_selection import process_missing_values, analyze_features
from feature_selection import FeatureSelection

import json
import re

def _sanitize_filename(text):
    return re.sub(r'[^\w\-_.]', '_', text)


# Cluster label color sequence by frequency, including -1 as gray
CLUSTER_COLOR_SEQUENCE = {
    -1: '#aaaaaa',  # gray for noise
    0: '#1f77b4',   # blue
    1: '#d62728',   # red
    2: '#8c564b',   # brown
    3: '#2ca02c',
    4: '#ff7f0e',
    5: '#9467bd',
    6: '#e377c2',
    7: '#7f7f7f',
    8: '#bcbd22',
    9: '#17becf'
}

def _build_true_label_color_map(true_labels, used_cluster_colors=None):
    """
    Build a dynamic color map for true labels.
    Prefer colors not already used by clusters for better visual distinction.
    """
    if used_cluster_colors is None:
        used_cluster_colors = set()

    # Candidate colors from matplotlib tab palettes
    candidate_colors = (
        list(plt.cm.tab20.colors) +
        list(plt.cm.Set3.colors) +
        list(plt.cm.Paired.colors)
    )

    # Convert used cluster colors to lowercase strings if possible
    used_cluster_colors = {
        str(c).lower() for c in used_cluster_colors if c is not None
    }

    # Keep only colors not used by clusters
    available_colors = []
    for c in candidate_colors:
        hex_color = matplotlib.colors.to_hex(c).lower()
        if hex_color not in used_cluster_colors:
            available_colors.append(hex_color)

    # Fallback: if not enough colors, reuse full palette
    if len(available_colors) == 0:
        available_colors = [matplotlib.colors.to_hex(c).lower() for c in candidate_colors]

    # Sort true labels by frequency
    true_labels_series = pd.Series(true_labels)
    true_sorted_labels = (
        true_labels_series.value_counts()
        .index
        .tolist()
    )

    # Build mapping
    true_label_color_map = {}
    for i, lbl in enumerate(true_sorted_labels):
        true_label_color_map[lbl] = available_colors[i % len(available_colors)]

    return true_label_color_map, true_sorted_labels

def _cluster_size_stats(labels):
    """
    Compute cluster size statistics excluding noise label -1.
    """
    labels = np.asarray(labels)
    valid_labels = labels[labels != -1]

    if len(valid_labels) == 0:
        return np.array([]), np.array([]), np.array([])

    unique_clusters, counts = np.unique(valid_labels, return_counts=True)
    ratios = counts / counts.sum()

    return unique_clusters, counts, ratios


def _is_valid_clustering(
    labels,
    min_clusters=2,
    min_cluster_ratio=0.05,
    max_cluster_ratio=0.90,
    max_noise_ratio=0.40,
    min_cluster_size_abs=2
):
    """
    Check whether clustering result is structurally valid.
    """
    labels = np.asarray(labels)

    noise_ratio = np.mean(labels == -1)
    if noise_ratio > max_noise_ratio:
        return False

    unique_clusters, counts, ratios = _cluster_size_stats(labels)

    if len(unique_clusters) < min_clusters:
        return False

    if len(counts) == 0:
        return False

    if counts.min() < min_cluster_size_abs:
        return False

    if ratios.min() < min_cluster_ratio:
        return False

    if ratios.max() > max_cluster_ratio:
        return False

    return True



class ClusteringTrainer:
    def __init__(self, X, y=None, patient = None,
                 method='KMeans', 
                 reduce='PCA',
                 save_path=Path('./result_clustering'),
                 feature_weights=None,
                 random_state=42,
                 show=True,
                 param_search=False):
        """
        Unsupervised clustering pipeline with feature selection, visualization, and optional parameter search.
        """
        self.X_raw = X
        self.y_true = y.str.strip().str.capitalize()  # Optional ground-truth labels for 
        self.patient = patient
        self.method = method
        self.reduce = reduce
        self.save_path = Path(save_path)
        self.random_state = random_state
        self.show = show
        self.param_search = param_search

        self.feature_selector = FeatureSelection(self.X_raw, self.y_true, reduce=self.reduce, save_path=self.save_path,feature_weights=feature_weights)
        X_reduced, y_reduced = self.feature_selector.dimension_reduce()
        self.X_reduced = X_reduced
        self.y_reduced = y_reduced
        self.labels_ = None
        self.model = None

    def run(self):

        if self.X_reduced is None or self.X_reduced.shape[1] < 1:
            print(f"[{self.reduce}_{self.method}] No valid features after reduction.")
            self.labels_ = np.full(len(self.X_raw), -1)
            return self.labels_, {'silhouette': np.nan, 'ARI': 0, 'NMI': 0}

        if self.param_search:
            self.model, self.labels_, best_params, best_score = self._grid_search()
            if best_params:
                param_str = _sanitize_filename(json.dumps(best_params, sort_keys=True))
                self.method += f"({param_str})"
            else:
                self.method += "(None)"
        else:
            self.model = self._get_model()
            try:
                self.labels_ = self.model.fit_predict(self.X_reduced)
            except Exception as e:
                print(f"[{self.reduce}_{self.method}] clustering failed: {e}")
                self.labels_ = np.full(len(self.X_reduced), -1)
        


        if all(x is not None for x in [self.X_reduced, self.patient, self.labels_]):
            df_reduced = pd.DataFrame({
                'Patient': self.patient,
                'Cluster': self.labels_
            }).join(self.X_reduced.reset_index(drop=True))
            
            save_dir = self.save_path / 'Clustering_labels'
            save_dir.mkdir(parents=True, exist_ok=True)
            df_reduced.to_csv(save_dir / f"{self.reduce}_{self.method}_Clustering_labels.csv", index=False)

        try:
            if len(np.unique(self.labels_)) < 2:
                print(f"[{self.reduce}_{self.method}] clustering returned only one cluster or noise. Silhouette skipped.")
                score = np.nan
            else:
                score = silhouette_score(self.X_reduced, self.labels_)
        except:
            score = np.nan

        ari = adjusted_rand_score(self.y_true, self.labels_) if self.y_true is not None else 0
        nmi = normalized_mutual_info_score(self.y_true, self.labels_) if self.y_true is not None else 0

        metrics = {
            'silhouette': score,
            'ARI': ari,
            'NMI': nmi
        }

        if self.show:
            if self.y_true is not None:
                self._plot_comparison(self.X_reduced, self.labels_, self.y_true)
            else:
                self._plot_2d(self.X_reduced, self.labels_)

        # HDBSCAN-specific explanation
        if isinstance(self.model, hdbscan.HDBSCAN):
            self._explain_hdbscan()

        # Generic decision tree explanation for all clustering methods
        self._explain_with_decision_tree()

        return self.labels_, metrics

    def _get_model(self, params=None):
        if params is None:
            if self.method == 'KMeans':
                return KMeans(n_clusters=min(3, len(self.X_reduced)), init='k-means++', n_init=10, max_iter=300, random_state=self.random_state)
            elif self.method == 'Agglomerative':
                return AgglomerativeClustering(n_clusters=min(3, len(self.X_reduced)))
            elif self.method == 'Spectral':
                return SpectralClustering(n_clusters=min(3, len(self.X_reduced)), assign_labels='kmeans', random_state=self.random_state)
            elif self.method == 'HDBSCAN':
                return hdbscan.HDBSCAN(min_cluster_size=2)
            else:
                raise ValueError(f"Unsupported clustering method: {self.method}")
        else:
            return self._get_model_with_params(params)

    def _get_model_with_params(self, params):
        if self.method.startswith('KMeans'):
            return KMeans(**params, random_state=self.random_state)
        elif self.method.startswith('Agglomerative'):
            return AgglomerativeClustering(**params)
        elif self.method.startswith('Spectral'):
            return SpectralClustering(**params, assign_labels='kmeans', random_state=self.random_state)
        elif self.method.startswith('HDBSCAN'):
            return hdbscan.HDBSCAN(**params)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

    def _get_adaptive_constraints(self, labels=None):
        """
        Build adaptive clustering constraints based on sample size
        and current clustering method.

        Parameters
        ----------
        labels : array-like or None
            Optional clustering labels. Can be used in future if
            method-specific adaptive refinement is needed.

        Returns
        -------
        dict
            Constraint dictionary for clustering validation.
        """
        n = len(self.X_reduced)

        # Absolute minimum cluster size
        if n < 15:
            min_cluster_size_abs = 2
        elif n < 30:
            min_cluster_size_abs = 3
        else:
            min_cluster_size_abs = max(3, int(np.ceil(0.08 * n)))

        # Minimum cluster ratio
        min_cluster_ratio = min_cluster_size_abs / max(n, 1)

        # Relax dominance threshold for small datasets
        if n < 15:
            max_cluster_ratio = 0.95
        elif n < 30:
            max_cluster_ratio = 0.90
        else:
            max_cluster_ratio = 0.85

        # HDBSCAN can naturally produce more noise
        if self.method == 'HDBSCAN':
            max_noise_ratio = 0.50 if n < 20 else 0.40
        else:
            max_noise_ratio = 0.10  # effectively disallow noise for non-density methods

        return {
            "min_clusters": 2,
            "min_cluster_size_abs": min_cluster_size_abs,
            "min_cluster_ratio": min_cluster_ratio,
            "max_cluster_ratio": max_cluster_ratio,
            "max_noise_ratio": max_noise_ratio
        }

    def _grid_search(self):
        X = self.X_reduced
        best_score = -1
        best_labels = None
        best_model = None
        best_params = None

        fallback_score = -1
        fallback_labels = None
        fallback_model = None
        fallback_params = None

        if self.method == 'KMeans':
            param_grid = ParameterGrid({'n_clusters': list(range(2, min(6, len(X))))})
        elif self.method == 'Agglomerative':
            param_grid = ParameterGrid({'n_clusters': list(range(2, min(6, len(X))))})
        elif self.method == 'Spectral':
            param_grid = ParameterGrid({'n_clusters': list(range(2, min(6, len(X))))})
        elif self.method == 'HDBSCAN':
            param_grid = ParameterGrid({'min_cluster_size': [2, 3, 4, 5, 6]})
        else:
            raise ValueError(f"Unsupported clustering method for grid search: {self.method}")

        for params in param_grid:
            try:
                model = self._get_model_with_params(params)
                labels = model.fit_predict(X)

                valid_labels = labels[labels != -1]
                if len(np.unique(valid_labels)) < 2:
                    continue

                if np.any(labels == -1):
                    mask = labels != -1
                    if len(np.unique(labels[mask])) < 2:
                        continue
                    score = silhouette_score(X[mask], labels[mask])
                else:
                    score = silhouette_score(X, labels)

                # Save fallback candidate
                if score > fallback_score:
                    fallback_score = score
                    fallback_labels = labels
                    fallback_model = model
                    fallback_params = params

                # Adaptive constraints
                constraints = self._get_adaptive_constraints(labels)

                if _is_valid_clustering(
                    labels,
                    min_clusters=constraints["min_clusters"],
                    min_cluster_ratio=constraints["min_cluster_ratio"],
                    max_cluster_ratio=constraints["max_cluster_ratio"],
                    max_noise_ratio=constraints["max_noise_ratio"],
                    min_cluster_size_abs=constraints["min_cluster_size_abs"]
                ):
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_model = model
                        best_params = params

            except Exception:
                continue

        if best_labels is None:
            if fallback_labels is not None:
                print(f"[{self.reduce}_{self.method}] No balanced solution found. Using fallback best-silhouette solution.")
                best_labels = fallback_labels
                best_model = fallback_model
                best_params = fallback_params
                best_score = fallback_score
            else:
                print(f"[{self.reduce}_{self.method}] No valid clustering found during parameter search.")
                best_labels = np.full(len(X), -1)
                best_score = -1

        return best_model, best_labels, best_params, best_score

    def _plot_2d(self, X, cluster_labels):
        if cluster_labels is None or len(np.unique(cluster_labels)) < 1:
            print(f"[{self.reduce}_{self.method}] No valid clustering labels to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        reducers = [
            ('PCA', PCA(n_components=2, random_state=self.random_state)),
            ('t-SNE', TSNE(
                n_components=2,
                random_state=self.random_state,
                perplexity=min(30, max(5, len(X) // 3)),
                learning_rate='auto',
                init='pca'
            ))
        ]

        for i, (method, reducer) in enumerate(reducers):
            try:
                X_proj = reducer.fit_transform(X)
                # Get unique cluster labels by frequency
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                sorted_labels = [x for _, x in sorted(zip(counts, unique_labels), reverse=True)]
                label_to_color = {label: CLUSTER_COLOR_SEQUENCE[i % len(CLUSTER_COLOR_SEQUENCE)] for i, label in enumerate(sorted_labels)}
                cluster_colors = [label_to_color[l] for l in cluster_labels]

                scatter = axes[i].scatter(X_proj[:, 0], X_proj[:, 1], c=cluster_colors, s=50)
                axes[i].set_title(f"{self.reduce}_{self.method} Clustering ({method})")
                axes[i].set_xlabel(f"{method} 1")
                axes[i].set_ylabel(f"{method} 2")
                for label in sorted_labels:
                    axes[i].scatter([], [], color=label_to_color[label], label=f"Cluster {label}")
                axes[i].legend()
            except Exception as e:
                print(f"[{self.reduce}_{self.method}] {method} visualization failed: {e}")

        plt.tight_layout()
        save_path = self.save_path/'Visualization'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"{self.reduce}_{self.method}_Clusters_PCA_TSNE.png")
        plt.close(fig)

    def _plot_comparison(self, X, cluster_labels, true_labels):
        if cluster_labels is None or len(np.unique(cluster_labels)) < 1:
            print(f"[{self.reduce}_{self.method}] No valid clustering labels to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        reducers = [
            ('PCA', PCA(n_components=2, random_state=self.random_state)),
            ('t-SNE', TSNE(
                n_components=2,
                random_state=self.random_state,
                perplexity=min(30, max(5, len(X) // 3)),
                learning_rate='auto',
                init='pca'
            ))
        ]

        for i, (method, reducer) in enumerate(reducers):
            try:
                X_proj = reducer.fit_transform(X)

                # Cluster colors
                unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
                sorted_clusters = [x for _, x in sorted(zip(counts, unique_clusters), reverse=True)]
                label_to_color = {
                    label: CLUSTER_COLOR_SEQUENCE.get(label, '#777777')
                    for label in sorted_clusters
                }
                cluster_colors = [label_to_color[l] for l in cluster_labels]

                axes[i, 0].scatter(X_proj[:, 0], X_proj[:, 1], c=cluster_colors, s=50)
                axes[i, 0].set_title(f"{self.reduce}_{self.method} Clustering ({method})")
                axes[i, 0].set_xlabel(f"{method} 1")
                axes[i, 0].set_ylabel(f"{method} 2")
                for label in sorted_clusters:
                    axes[i, 0].scatter([], [], color=label_to_color[label], label=f"Cluster {label}")
                axes[i, 0].legend()

                # True label colors (dynamic and avoid cluster colors if possible)
                used_cluster_colors = set(cluster_colors)
                true_label_color_map, true_sorted_labels = _build_true_label_color_map(
                    true_labels=true_labels,
                    used_cluster_colors=used_cluster_colors
                )

                true_colors = [true_label_color_map.get(lbl, '#777777') for lbl in true_labels]

                axes[i, 1].scatter(X_proj[:, 0], X_proj[:, 1], c=true_colors, s=50)
                axes[i, 1].set_title(f"True Labels ({method})")
                axes[i, 1].set_xlabel(f"{method} 1")
                axes[i, 1].set_ylabel(f"{method} 2")
                for name in true_sorted_labels:
                    axes[i, 1].scatter([], [], color=true_label_color_map[name], label=str(name))
                axes[i, 1].legend()

            except Exception as e:
                print(f"[{self.reduce}_{self.method}] {method} visualization failed: {e}")

        plt.tight_layout()
        save_path = self.save_path / 'Visualization'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"{self.reduce}_{self.method}_Comparison_PCA_TSNE.png")
        plt.close(fig)

    def _explain_hdbscan(self):
        """
        HDBSCAN-specific analysis:
        - Cluster persistence and sizes
        - Outlier scores
        - Condensed tree and minimum spanning tree plots
        - Optional: decision tree explaining the smallest cluster vs others

        All outputs are saved under save_path / 'HDBSCAN_explain_test'.
        """
        if not isinstance(self.model, hdbscan.HDBSCAN):
            return

        if self.labels_ is None:
            print(f"[{self.reduce}_{self.method}] No labels for HDBSCAN explanation.")
            return

        save_dir = self.save_path / 'HDBSCAN_explain_test'
        save_dir.mkdir(parents=True, exist_ok=True)

        labels = self.labels_

        # -------- 1. Cluster persistence and sizes --------
        persistence = getattr(self.model, "cluster_persistence_", None)

        records = []
        unique_labels = np.unique(labels)

        for lab in unique_labels:

            pers = float(persistence[lab])


            size = int(np.sum(labels == lab))
            records.append({
                "cluster_label": int(lab),
                "size": size,
                "persistence": pers
            })

        df_clusters = pd.DataFrame(records)
        df_clusters.to_csv(
            save_dir / f"{self.reduce}_{self.method}_cluster_persistence_size.csv",
            index=False
        )

        # -------- 2. Outlier scores --------
        outlier_scores = getattr(self.model, "outlier_scores_", None)
        if outlier_scores is not None:
            df_out = pd.DataFrame({
                "outlier_score": outlier_scores,
                "cluster_label": labels
            })
            if self.patient is not None:
                df_out["Patient"] = self.patient

            df_out.to_csv(
                save_dir / f"{self.reduce}_{self.method}_outlier_scores.csv",
                index=False
            )

        # -------- 3. Condensed tree plot --------
        try:
            self.model.condensed_tree_.plot(select_clusters=True)
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            fig.tight_layout()
            fig.savefig(
                save_dir / f"{self.reduce}_{self.method}_condensed_tree.png",
                dpi=300
            )
            plt.close(fig)
        except Exception as e:
            print(f"[{self.reduce}_{self.method}] Condensed tree plotting failed: {e}")

        # -------- 4. Minimum spanning tree plot --------
        try:
            self.model.minimum_spanning_tree_.plot()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            fig.tight_layout()
            fig.savefig(
                save_dir / f"{self.reduce}_{self.method}_minimum_spanning_tree.png",
                dpi=300
            )
            plt.close(fig)
        except Exception as e:
            print(f"[{self.reduce}_{self.method}] Minimum spanning tree plotting failed: {e}")

        # -------- 5. Extra: decision tree for the smallest cluster vs others --------
        # Ignore noise (-1) when searching for the smallest cluster.
        non_noise_mask = labels != -1
        non_noise_labels = labels[non_noise_mask]

        if non_noise_labels.size > 0:
            cluster_ids, counts = np.unique(non_noise_labels, return_counts=True)
            smallest_cluster = cluster_ids[np.argmin(counts)]

            y_small = (labels == smallest_cluster).astype(int)
            X = self.X_reduced

            # Feature names as in the generic tree
            feature_names = self._get_feature_names()

            clf = DecisionTreeClassifier(
                max_depth=4,
                random_state=self.random_state,
                class_weight='balanced'
            )

            try:
                clf.fit(X, y_small)

                try:
                    rules_small = export_text(clf, feature_names=feature_names)
                except Exception:
                    rules_small = export_text(clf)

                rules_path = save_dir / f"{self.reduce}_{self.method}_smallCluster{smallest_cluster}_tree_rules.txt"
                with open(rules_path, 'w', encoding='utf-8') as f:
                    f.write(rules_small)

                print(f"[{self.reduce}_{self.method}] Smallest cluster {smallest_cluster} decision tree explanation saved to {rules_path}")
            except Exception as e:
                print(f"[{self.reduce}_{self.method}] Decision tree for smallest cluster failed: {e}")

    def _get_feature_names(self):
        """
        Retrieve feature names in the correct order used by X_reduced.
        Priority:
        1. Column names of X_reduced if it is a DataFrame
        2. Reduced / selected / original feature names from FeatureSelection
        3. Fallback: generic names ['f0', 'f1', ...]
        """
        # Case 1: use X_reduced DataFrame columns directly
        if isinstance(self.X_reduced, pd.DataFrame):
            # Ensure string type for all names
            return [str(c) for c in self.X_reduced.columns]

        fs = self.feature_selector

        # Case 2: reduced features (e.g. PCA/LDA components)
        if hasattr(fs, "reduced_feature_names_") and fs.reduced_feature_names_ is not None:
            names = list(fs.reduced_feature_names_)
            if len(names) == self.X_reduced.shape[1]:
                return names

        # Case 3: selected feature names
        if hasattr(fs, "selected_feature_names_") and fs.selected_feature_names_ is not None:
            names = list(fs.selected_feature_names_)
            if len(names) == self.X_reduced.shape[1]:
                return names

        # Case 4: original feature names if dimension was not changed
        if hasattr(fs, "feature_names_") and fs.feature_names_ is not None:
            names = list(fs.feature_names_)
            if len(names) == self.X_reduced.shape[1]:
                return names

        # Case 5: fallback
        return [f"f{i}" for i in range(self.X_reduced.shape[1])]

    def _explain_with_decision_tree(self, max_depth=5):
        """
        Train a decision tree to explain cluster assignments for any clustering method.
        The tree learns: X_reduced -> cluster_labels.

        Notes:
        - Only label -1 is treated as noise (HDBSCAN convention).
        - Label 0 is a normal cluster label, NOT noise.
        """
        # Check cluster labels
        if self.labels_ is None:
            print(f"[{self.reduce}_{self.method}] No labels for decision tree explanation.")
            return

        unique_labels = np.unique(self.labels_)

        # If all samples are labeled as -1, it is truly "all noise"
        if len(unique_labels) == 1 and unique_labels[0] == -1:
            print(f"[{self.reduce}_{self.method}] All samples are noise (-1). Decision tree explanation skipped.")
            return

        # Need at least two distinct labels to train a meaningful classifier
        if len(unique_labels) < 2:
            print(f"[{self.reduce}_{self.method}] Only one cluster label found. Decision tree explanation skipped.")
            return

        X = self.X_reduced
        y = self.labels_

        # Use DataFrame column names if available; otherwise fallback to helper
        if isinstance(X, pd.DataFrame):
            feature_names = [str(c) for c in X.columns]
        else:
            feature_names = self._get_feature_names()

        # Train decision tree
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.random_state,
            class_weight='balanced'  # handle unbalanced cluster sizes
        )

        try:
            clf.fit(X, y)
        except Exception as e:
            print(f"[{self.reduce}_{self.method}] Decision tree fit failed: {e}")
            return

        # Export rules as text (with feature names)
        try:
            tree_rules = export_text(clf, feature_names=feature_names)
        except Exception:
            # Fallback when feature_names length does not match
            tree_rules = export_text(clf)

        # Prepare save directory
        save_dir = self.save_path / 'tree_explain'
        save_dir.mkdir(parents=True, exist_ok=True)

        file_prefix = f"{self.reduce}_{self.method}"

        # Save rules
        rules_path = save_dir / f"{file_prefix}_tree_rules.txt"
        with open(rules_path, 'w', encoding='utf-8') as f:
            f.write(tree_rules)

        # Save feature importances
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            df_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            df_imp.sort_values('importance', ascending=False, inplace=True)
            df_imp.to_csv(save_dir / f"{file_prefix}_feature_importance.csv", index=False)

        print(f"[{self.reduce}_{self.method}] Decision tree explanation saved to {save_dir}")

def generate_clustering_report(X, y, patient,
                                save_path=Path('./result_clustering'),
                                feature_weights = None,
                                reduce_list=['EBA','PCA', 'HDF', 'L1SS'],
                                clustering_methods=['KMeans', 'Agglomerative', 'Spectral', 'HDBSCAN'],
                                param_search=False,
                                show=True):
    save_path = Path(save_path)
    result_dir = save_path / 'Clustering Report'
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = []
    sil_matrix, ari_matrix, nmi_matrix = [], [], []

    for reduce in reduce_list:
        if y is None and reduce in ['HDF','L1SS']:
            print(f"Skipping reduction method '{reduce}' due to lack of true labels.")
            continue
        sil_row, ari_row, nmi_row = [], [], []

        for method in clustering_methods:
            trainer = ClusteringTrainer(X, y, patient, method=method, reduce=reduce, save_path=result_dir,feature_weights = feature_weights, show=show, param_search=param_search)
            try:
                _, metrics = trainer.run()
            except Exception as e:
                print(f"[{method} - {reduce}] clustering failed in report generation: {e}")
                metrics = {'silhouette': np.nan, 'ARI': np.nan, 'NMI': np.nan}

            sil_row.append(metrics.get('silhouette', np.nan))
            ari_row.append(metrics.get('ARI', np.nan))
            nmi_row.append(metrics.get('NMI', np.nan))

            row = {'method': trainer.method, 'reduction': reduce} | metrics
            metrics_all.append(row)

        sil_matrix.append(sil_row)
        ari_matrix.append(ari_row)
        nmi_matrix.append(nmi_row)

    valid_reductions = [r for r in reduce_list if not (y is None and r in ['HDF','L1SS'])]
    sil_df = pd.DataFrame(sil_matrix, index=valid_reductions, columns=clustering_methods).T
    ari_df = pd.DataFrame(ari_matrix, index=valid_reductions, columns=clustering_methods).T
    nmi_df = pd.DataFrame(nmi_matrix, index=valid_reductions, columns=clustering_methods).T
    full_df = pd.DataFrame(metrics_all)

    with ExcelWriter(save_path / 'clustering_report.xlsx') as writer:
        full_df.to_excel(writer, sheet_name='Metrics Table', index=False)
        sil_df.to_excel(writer, sheet_name='Silhouette')
        ari_df.to_excel(writer, sheet_name='ARI')
        nmi_df.to_excel(writer, sheet_name='NMI')


if __name__ == "__main__":

    from feature_extraction.DatabaseConnection import get_latest_output_save_path
    latest_time, data_path = get_latest_output_save_path(output_root=Path("./output"))

    data_name = 'all_features'
    #data_name = 'scenario_2_features'

    #for data_name in ['all_features','scenario_1_features','scenario_2_features','scenario_3_features','scenario_4_features','scenario_5_features']:

    feature_path = data_path/str('data')/f'{data_name}.csv'
    data = pd.read_csv(feature_path)
    save_path = data_path/ f'results_{data_name}'

    #exclude_ids = ["P01-02", "P01-04"]
    #data = data[~data["id_patient"].str.contains("|".join(exclude_ids))]

    data = process_missing_values(data, save_path = save_path)


    X = data.loc[:, ~data.columns.str.match(r'^(id|label)', case=False)]
    y = data['label']

    #feature_weights = [1]*127+[0]*(len(X.columns)-127)
    feature_weights = None

    """
    analyze_features(X, y, save_path=save_path/ 'Feature Selection')
    for reduce in ['EBA','PCA','HDF','L1SS']:
        feature = FeatureSelection(X,y,reduce,save_path=save_path/ 'Feature Selection',feature_weights=feature_weights)
        X_processed,y_processed = feature.process()
    """

    patient = data['id_patient']
    generate_clustering_report(X,y,patient,param_search=True,save_path = save_path/'Clustering Analysis',feature_weights=feature_weights) 


