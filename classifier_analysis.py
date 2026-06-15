import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from pandas import ExcelWriter

from feature_extraction.DatabaseConnection import myDatabase
from feature_selection import show_pca_2d,process_missing_values, analyze_features
from feature_selection import FeatureSelection


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold,LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import shap

from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter


class PrototypicalClassifier:
    """
    A prototype-based classifier.

    This classifier represents each class by the mean feature vector
    (prototype) computed from the training samples. During prediction,
    each sample is assigned to the class whose prototype is closest
    according to a specified distance metric.
    """

    def __init__(self, distance_metric='euclidean'):
        """
        Initialize the classifier.

        Parameters
        ----------
        distance_metric : str
            Distance metric used to compute similarity between samples
            and class prototypes (default: 'euclidean').
            Any metric supported by scipy.spatial.distance.cdist can be used.
        """
        self.distance_metric = distance_metric
        self.prototypes = {}   # Dictionary: {class_label: prototype_vector}
        self.classes_ = None   # Array of unique class labels

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Compute class prototypes from the training data.
        """
        self.classes_ = np.unique(y)

        # Compute mean feature vector for each class
        self.prototypes = {
            cls: X[y == cls].mean(axis=0).values
            for cls in self.classes_
        }

    def predict(self, X):
        """
        Predict class labels for new samples.
        """

        # Ensure input is converted to NumPy array
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        elif isinstance(X, np.ndarray):
            X_np = X
        else:
            raise TypeError("Input X must be a pandas DataFrame or a numpy ndarray.")

        # Stack prototype vectors into matrix [n_classes, n_features]
        proto_mat = np.stack([
            self.prototypes[cls] for cls in self.classes_
        ])

        # Compute distance between samples and prototypes
        dists = cdist(
            X_np,
            proto_mat,
            metric=self.distance_metric
        )  # Shape: [n_samples, n_classes]

        # Assign class with minimum distance
        preds = np.argmin(dists, axis=1)

        return self.classes_[preds]

class NoiseAugmentationSampler:
    """
    Feature-level noise augmentation for minority classes only.

    This sampler mimics the interface of imbalanced-learn samplers
    (e.g., SMOTE) by providing a fit_resample method.

    The augmentation adds small Gaussian noise to minority class samples
    to approximate intra-class variability without interpolating across samples.
    """

    def __init__(
        self,
        alpha: float = 0.02,
        target_ratio: float = 1.0,
        clip: bool = True,
        random_state: int | None = None
    ):
        """
        Parameters
        ----------
        alpha : float
            Noise strength relative to feature standard deviation.
            Typical values: 0.01 ~ 0.05.
        target_ratio : float
            Target ratio minority / majority after augmentation.
            1.0 means fully balanced.
        clip : bool
            Whether to clip augmented features to original feature range.
        random_state : int or None
            Random seed for reproducibility.
        """
        self.alpha = alpha
        self.target_ratio = target_ratio
        self.clip = clip
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)

    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray
    ):
        """
        Perform noise-based augmentation on minority classes.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (training fold only).
        y : pd.Series or array-like
            Target labels.

        Returns
        -------
        X_res : pd.DataFrame
            Augmented feature matrix.
        y_res : pd.Series
            Augmented labels.
        """

        # Ensure pandas format
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="label")

        class_counts = Counter(y)
        max_count = max(class_counts.values())

        X_res_list = [X]
        y_res_list = [y]

        # Feature-wise std (avoid zero std)
        feature_std = X.std().replace(0, 1e-6)

        for cls, count in class_counts.items():
            target_count = int(self.target_ratio * max_count)

            if count >= target_count:
                continue  # not a minority class

            n_to_generate = target_count - count

            X_cls = X[y == cls]

            # Sample minority instances with replacement
            base_indices = self._rng.choice(
                X_cls.index,
                size=n_to_generate,
                replace=True
            )
            X_base = X_cls.loc[base_indices]

            # Generate Gaussian noise
            noise = self._rng.normal(
                loc=0.0,
                scale=self.alpha * feature_std.values,
                size=X_base.shape
            )

            X_noisy = X_base.values + noise
            X_noisy = pd.DataFrame(X_noisy, columns=X.columns)

            if self.clip:
                X_noisy = X_noisy.clip(
                    lower=X.min(),
                    upper=X.max(),
                    axis=1
                )

            y_noisy = pd.Series(
                [cls] * len(X_noisy),
                name=y.name
            )

            X_res_list.append(X_noisy)
            y_res_list.append(y_noisy)

        X_res = pd.concat(X_res_list, axis=0).reset_index(drop=True)
        y_res = pd.concat(y_res_list, axis=0).reset_index(drop=True)

        return X_res, y_res

class ClassifierTrainer:
    """
    Train a classifier with cross-validation, optional SMOTE, PCA visualization, and SLAP feature analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features).
    y : pd.Series or array-like
        Target labels.
    split: str
        Train and test split methpd used ('StratifiedKFold', 'RepeatedStratifiedKFold','LeaveOneOut')
    reduce : str
        Feature reduction method used ( 'EBA', 'PCA' 'HDF' or 'L1SS').
    imbalance_strategy :  None | "smote" | "borderline" | "noise_aug"
        imbalance_strategy (default: smote) None for not apply strategy, smote for SMOTE, borderline for Borderline-SMOTE.
    show : bool
        Whether to plot PCA visualizations and confusion matrix (default: True).
    classifier_type : str
        Classifier type: 'KNN','Prototypical', 'DecisionTree', 'RandomForest','LogisticRegression','SVM' or 'Voting'
    classifier_params : dict or None
        Hyperparameters for the classifier.
    title : str
        Custom title for plots and saved files. If empty, it's auto-generated.
    """
    def __init__(self, X, y,
                 split= 'RepeatedStratifiedKFold',
                 reduce='L1SS',
                 feature_weights = None,
                 classifier_type='KNN', 
                 classifier_params=None,
                 imbalance_strategy="smote",  
                 imbalance_params=None,
                 save_path='tempClassifier', 
                 show=True, 
                 random_state=42, 
                 title=''):
        

        class_counts = Counter(y)
        min_class_count = min(class_counts.values())

        if min_class_count < 4:
            #print(f"[Warning] Some classes too small ({class_counts}), duplicating all classes with <4 samples...")

            X = pd.DataFrame(X).reset_index(drop=True)
            y = pd.Series(y).reset_index(drop=True)

            X_list, y_list = [], []

            rng = np.random.default_rng(seed=random_state)  # for reproducible random sampling

            for cls, count in class_counts.items():
                X_cls = X[y == cls]
                y_cls = y[y == cls]

                if len(y_cls) < 4:
                    # Randomly replicate samples until at least 4
                    needed = 4 - len(y_cls)
                    idx_to_duplicate = rng.choice(X_cls.index, size=needed, replace=True)
                    X_dup = X_cls.loc[idx_to_duplicate]
                    y_dup = y_cls.loc[idx_to_duplicate]
                    X_cls = pd.concat([X_cls, X_dup], axis=0)
                    y_cls = pd.concat([y_cls, y_dup], axis=0)
                    #print(f"  [Class {cls}] duplicated from {count} → {len(y_cls)} samples.")

                X_list.append(X_cls)
                y_list.append(y_cls)

            # Combine all classes back
            X = pd.concat(X_list, axis=0).reset_index(drop=True)
            y = pd.concat(y_list, axis=0).reset_index(drop=True)

        self.X = X
        self.y = y
        self.reduce = reduce
        self.feature_weights = feature_weights
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params or {}
        self.imbalance_strategy = imbalance_strategy
        self.imbalance_params = imbalance_params
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.show = show
        self.random_state = random_state
        
        if title:
            self.title = title
        else:
            imbalance_str = f"with {imbalance_strategy}" if imbalance_strategy is not None else "without resampling"
            self.title = f"{classifier_type} for {reduce} features ({imbalance_str})"

        
        safe_n_splits = max(2, min(4, min(Counter(y).values())))
        splitters = {
            'StratifiedKFold': StratifiedKFold(n_splits=safe_n_splits, shuffle=True, random_state=random_state),
            'LeaveOneOut': LeaveOneOut(),
            'RepeatedStratifiedKFold': RepeatedStratifiedKFold(
                n_splits=max(2, min(3, min(Counter(y).values()))),
                n_repeats=4,
                random_state=random_state
            ),
        }

        if split not in splitters:
            raise ValueError(f"Unsupported split strategy: {split}")


        self.rkf = splitters[split]
        self.unique_labels = np.unique(y)
        self.total_folds = self.rkf.get_n_splits(X, y)

    def train(self):
        metric_rows = []
        confusion_matrices = []
        slap_matrix = pd.DataFrame()
        shap_matrix = pd.DataFrame()
        final_clf = None
        print(f"Training for {self.title} begin:")

        for current_fold, (train_index, test_index) in enumerate(self.rkf.split(self.X, self.y), start=1):
            
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            X_train, y_train = self._apply_imbalance_strategy(X_train, y_train)

            feature_selection = FeatureSelection(X_train,y_train,reduce=self.reduce,save_path=self.save_path, feature_weights=self.feature_weights)
            X_train,y_train = feature_selection.dimension_reduce()
            X_test = feature_selection.transform_test(X_test)

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)


            clf = self._get_classifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            confusion_matrices.append(confusion_matrix(y_test, y_pred, labels=self.unique_labels))
            slap = self._compute_slap(X_test, y_test, clf)
            slap_matrix = pd.concat([slap_matrix, slap], ignore_index=True)
            shap = self._compute_shap(X_test, clf)
            shap_matrix = pd.concat([shap_matrix, shap], ignore_index=True)


            if current_fold == self.total_folds and self.show:
                self._visualize_pca(X_train, y_train, "train set")

            metric_rows.append(self._evaluate(y_test, y_pred))
            final_clf = clf

        report_df = self._summarize_metrics(metric_rows)
        self._visualize_confusion_matrix(confusion_matrices)
        slap_scores = self._summarize_importance(slap_matrix, 'SLAP')
        shap_scores = self._summarize_importance(shap_matrix, 'SHAP')
        self._visualize_feature_importance(slap_scores, shap_scores)

        return final_clf, report_df, slap_scores, shap_scores

    def _apply_imbalance_strategy(self, X_train, y_train):
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values())

        # No resampling
        if self.imbalance_strategy is None:
            return X_train, y_train

        # SMOTE family requires at least 2 samples per class
        if self.imbalance_strategy in ["smote", "borderline"] and min_class_count < 2:
            print(
                f"[Warning] Skipping {self.imbalance_strategy} "
                f"due to too few samples: {class_counts}"
            )
            return X_train, y_train

        params = self.imbalance_params or {}

        # ----- SMOTE -----
        if self.imbalance_strategy == "smote":
            safe_k = max(1, min(5, min_class_count - 1))
            sampler = SMOTE(
                k_neighbors=safe_k,
                random_state=self.random_state,
                **params
            )

        # ----- Borderline-SMOTE -----
        elif self.imbalance_strategy == "borderline":
            safe_k = max(1, min(5, min_class_count - 1))
            sampler = BorderlineSMOTE(
                k_neighbors=safe_k,
                random_state=self.random_state,
                **params
            )

        # ----- Feature-level noise augmentation -----
        elif self.imbalance_strategy == "noise_aug":
            sampler = NoiseAugmentationSampler(
                alpha=params.get("alpha", 0.02),
                target_ratio=params.get("target_ratio", 1.0),
                clip=params.get("clip", True),
                random_state=self.random_state
            )

        else:
            raise ValueError(f"Unknown imbalance strategy: {self.imbalance_strategy}")

        X_res, y_res = sampler.fit_resample(X_train, y_train)

        X_res = pd.DataFrame(X_res, columns=X_train.columns)
        y_res = pd.Series(y_res, name=y_train.name)

        return X_res, y_res

    def _get_classifier(self):
        default_params = {
            'KNN': {
                'n_neighbors': 3,
                'weights': 'distance', 
                'metric': 'minkowski'
            },
            'RandomForest': {
                'n_estimators': 100,
                'max_depth': 4,              
                'class_weight': 'balanced',
                'random_state': self.random_state
            },
            'DecisionTree': {
                'max_depth': 4,
                'class_weight': 'balanced',
                'random_state': self.random_state
            },
            'LogisticRegression': {
                'solver': 'lbfgs',
                'penalty': 'l2',
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 5000,
                'random_state': self.random_state
            },
            'SVM': {
                'kernel': 'rbf',
                'C': 1.0,
                'class_weight': 'balanced',
                'probability': True,
                'random_state': self.random_state
            },
            'Prototypical': {},
        }

        def merge_params(model_key):
            return {**default_params.get(model_key, {}), **self.classifier_params}

        classifier_map = {
            'KNN': lambda: KNeighborsClassifier(**merge_params('KNN')),
            'RandomForest': lambda: RandomForestClassifier(**merge_params('RandomForest')),
            'DecisionTree': lambda: DecisionTreeClassifier(**merge_params('DecisionTree')),
            'LogisticRegression': lambda: LogisticRegression(**merge_params('LogisticRegression')),
            'Prototypical': lambda: PrototypicalClassifier(**merge_params('Prototypical')),  # Placeholder
            'SVM': lambda: SVC(**merge_params('SVM')),
            'Voting': self._build_voting_classifier
        }

        if self.classifier_type not in classifier_map:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")

        return classifier_map[self.classifier_type]()

    def _build_voting_classifier(self):
        voting_type = self.classifier_params.get('voting', 'hard')
        estimator_params = self.classifier_params.get('estimators', {})

        """
        clf1 = LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=self.random_state,
            **estimator_params.get('lr', {})
        )
        """
        clf1 = KNeighborsClassifier(**estimator_params.get('knn', {}))
        clf2 = SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=self.random_state,
            **estimator_params.get('svm', {})
        )
        clf3 = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=self.random_state,
            **estimator_params.get('rf', {})
        )

        voting_clf = VotingClassifier(
            estimators=[
                ('knn', clf1),
                ('svm', clf2),
                ('rf', clf3)
            ],
            voting=voting_type
        )
        return voting_clf

    def _evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def _compute_slap(self, X_test, y_test, clf):
        drops = pd.DataFrame()
        for col in X_test.columns:
            X_perturbed = X_test.copy()
            X_perturbed[col] = 0
            y_perturbed = clf.predict(X_perturbed)
            drop = 1.0 - accuracy_score(y_test, y_perturbed)
            drops.loc[0,col] = drop
        
        return drops

    def _summarize_metrics(self, metrics):
        df_metrics = pd.DataFrame(metrics)
        avg_metrics = df_metrics.mean().to_frame().T
        df_report = pd.concat([avg_metrics, df_metrics], ignore_index=True)
        df_report.index = ['avg'] + [f'fold{i+1}' for i in range(len(metrics))]
        df_report = df_report * 100
        df_report = df_report.round(2)
        save_path = self.save_path/'Classification Report'
        save_path.mkdir(parents=True, exist_ok=True)
        df_report.to_csv(save_path / f"{self.title}.csv")
        return df_report

    def _visualize_confusion_matrix(self, confusion_matrices):
        """Visualize averaged confusion matrix with large bold labels and numbers."""
        if not self.show:
            return

        import numpy as np
        import matplotlib.pyplot as plt
        from collections import Counter

        # Compute averaged confusion matrix
        mean_cm = np.mean(confusion_matrices, axis=0)

        # Sort labels by frequency
        label_counts = Counter(self.y)
        sorted_labels = [label for label, _ in label_counts.most_common()]
        label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        reorder_indices = [label_to_index[label] for label in sorted_labels]
        mean_cm_sorted = mean_cm[np.ix_(reorder_indices, reorder_indices)]

        # Normalize rows
        eps = 1e-12
        row_sums = mean_cm_sorted.sum(axis=1, keepdims=True)
        cm_normalized = mean_cm_sorted / np.clip(row_sums, eps, None)
        labels = [str(label) for label in sorted_labels]

        # Create compact figure
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")

        # Minimal colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cbar.ax.tick_params(labelsize=7)

        # Set axes and tick labels (same font style as numbers)
        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            ylabel="True label",
            xlabel="Predicted label",
            title=f"{self.title}",
        )

        # Bold + larger labels for axes and tick labels
        ax.set_xlabel("Predicted label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True label", fontsize=12, fontweight="bold")
        #ax.set_title(f"{self.title}", fontsize=13, fontweight="bold", pad=10)
        ax.set_title(f"{self.classifier_type} with {self.reduce}", fontsize=13, fontweight="bold", pad=10)

        # Bold tick labels (same style as numbers)
        ax.tick_params(axis="both", which="major", labelsize=12, width=0)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Make square cells
        ax.set_aspect("equal")

        # Large and bold numbers filling each cell
        fmt = ".0%"
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                val = cm_normalized[i, j]
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color="white" if val > 0.5 else "black",
                )

        # Remove spines and tighten layout
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.15)

        # Save figure
        save_path = self.save_path / "Confusion Matrix"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"{self.title}.png", dpi=300, bbox_inches="tight")
        #plt.savefig(save_path / f"{self.classifier_type} with {self.reduce}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _summarize_slap(self, slap_matrix):
        slap_scores = slap_matrix.mean().sort_values(ascending=False)
        save_path = self.save_path/'SLAP csv'
        save_path.mkdir(parents=True, exist_ok=True)
        slap_scores.to_csv(save_path / f"SLAP Feature Importance {self.title}.csv")
        return slap_scores

    def _visualize_slap(self, slap_scores):
        if not self.show:
            return

        top_scores = slap_scores.head(10).iloc[::-1]  # Reverse order: largest value on top

        # 1. Adjust figure width dynamically based on feature name length
        max_len = top_scores.index.str.len().max()
        fig_width = 8 + max_len * 0.3
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # 2. Plot horizontal bar chart and get bar container
        bars = ax.barh(top_scores.index, top_scores.values)

        # 3. Add title, axis labels, and grid
        ax.set_title(f"Top 10 SLAP Feature Importance - {self.title}")
        ax.set_xlabel("Importance (Δ accuracy)")
        ax.set_ylabel("Feature")
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        # 4. Set x-axis start point (only when figure is wide)
        if fig_width > 12:
            min_val = top_scores.min()-0.1
            x_start = math.floor(min_val * 10) / 10.0
            ax.set_xlim(left=x_start)

        # 5. Add value label to the right of each bar (2 decimal places)
        ax.bar_label(bars, fmt='%.2f', padding=3)

        plt.tight_layout()

        save_path = self.save_path / 'SLAP png'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"SLAP Feature Importance {self.title}.png")
        plt.close(fig)

    def _visualize_pca(self, X, y, title_suffix):
        save_path = self.save_path/'PCA png'
        save_path.mkdir(parents=True, exist_ok=True)
        show_pca_2d(X, y, label_column='permis', title=f"{self.title} ({title_suffix})", save_path=save_path)

    def _compute_shap(self, X_test: pd.DataFrame, clf):
        """
        Compute SHAP values for a classifier (including VotingClassifier).
        Returns mean absolute SHAP values per feature.
        """

        X_test_df = X_test.astype(float).copy()
        feature_names = X_test_df.columns
        background = shap.sample(X_test_df, 100) if len(X_test_df) > 100 else X_test_df
        class_map = {c: i for i, c in enumerate(np.unique(self.y))}

        def predict_fn(model, x):
            try:
                return model.predict_proba(x)
            except:
                y_pred = model.predict(x)
                if isinstance(y_pred[0], str):
                    return np.array([class_map[y] for y in y_pred]).reshape(-1, 1)
                return y_pred.reshape(-1, 1)

        def compute_single_shap(model):
            try:
                explainer = shap.Explainer(model, background)
                shap_vals = explainer(X_test_df).values
            except:
                explainer = shap.KernelExplainer(lambda x: predict_fn(model, pd.DataFrame(x, columns=feature_names)), background)
                shap_vals = explainer.shap_values(X_test_df, nsamples=100)

            # --- Normalize format of SHAP outputs ---
            if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                # shape = (n_samples, n_features, n_classes)
                shap_array = np.mean(np.abs(shap_vals), axis=2)
            elif isinstance(shap_vals, list) and isinstance(shap_vals[0], np.ndarray):
                # list of (n_samples, n_features)
                shap_tensor = np.stack([np.abs(sv) for sv in shap_vals], axis=0)
                shap_array = np.mean(shap_tensor, axis=0)
            else:
                shap_array = np.abs(shap_vals)

            return pd.DataFrame(shap_array, columns=feature_names).mean().to_frame().T

        # If VotingClassifier, average SHAP from sub-estimators
        if isinstance(clf, VotingClassifier):
            #print("SHAP: Detected VotingClassifier. Averaging SHAP from all sub-estimators.")
            shap_dfs = []
            for name, model in clf.named_estimators_.items():
                try:
                    check_is_fitted(model)
                    shap_dfs.append(compute_single_shap(model))
                except Exception as e:
                    print(f"[SHAP] Skipping '{name}' due to error: {e}")
            if shap_dfs:
                return pd.concat(shap_dfs).mean().to_frame().T
            else:
                raise RuntimeError("No valid sub-estimators for SHAP.")
        else:
            return compute_single_shap(clf)

    def _summarize_shap(self, shap_matrix):
        shap_scores = shap_matrix.mean().sort_values(ascending=False)
        save_path = self.save_path/'SHAP csv'
        save_path.mkdir(parents=True, exist_ok=True)
        shap_scores.to_csv(save_path / f"SHAP Feature Importance {self.title}.csv")
        return shap_scores

    def _visualize_shap(self, shap_scores):
        if not self.show:
            return

        top_scores = shap_scores.head(10).iloc[::-1]
        max_len = top_scores.index.str.len().max()
        fig_width = 8 + max_len * 0.3
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        bars = ax.barh(top_scores.index, top_scores.values)

        ax.set_title(f"Top 10 SHAP Feature Importance - {self.title}")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_ylabel("Feature")
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        if fig_width > 12:
            min_val = top_scores.min()-0.01
            x_start = math.floor(min_val * 10) / 10.0
            ax.set_xlim(left=x_start)

        ax.bar_label(bars, fmt='%.2f', padding=3)

        plt.tight_layout()
        save_path = self.save_path / 'SHAP png'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"SHAP Feature Importance {self.title}.png")
        plt.close(fig)

    def _summarize_importance(self, importance_matrix, method_name):
        """
        Summarize and save feature importance matrix (SLAP or SHAP).

        Parameters
        ----------
        importance_matrix : pd.DataFrame
            Accumulated feature importance scores over folds.
        method_name : str
            Name of the method, e.g., 'SLAP' or 'SHAP'.

        Returns
        -------
        pd.Series
            Averaged and sorted feature importance scores.
        """
        importance_scores = importance_matrix.mean().sort_values(ascending=False)

        save_dir = self.save_path / f'{method_name} csv'
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"{method_name} Feature Importance {self.title}.csv"
        importance_scores.to_csv(file_path)

        return importance_scores
    
    def _visualize_feature_importance(self, slap_scores, shap_scores):
        if not self.show:
            return

        # Get top 10 from each score and reverse for horizontal bar plot
        top_slap = slap_scores.head(10).iloc[::-1]
        top_shap = shap_scores.head(10).iloc[::-1]

        # Dynamically adjust figure size
        max_len = max(top_slap.index.str.len().max(), top_shap.index.str.len().max())
        fig_width = 8 + max_len * 0.3
        fig_height = 12  # More vertical space for two subplots

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height))

        # --- Plot SLAP ---
        bars1 = ax1.barh(top_slap.index, top_slap.values)
        ax1.set_title(f"Top 10 SLAP Feature Importance - {self.title}")
        ax1.set_xlabel("Importance (Δ accuracy)")
        ax1.set_ylabel("Feature")
        ax1.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax1.bar_label(bars1, fmt='%.2f', padding=3)

        # Set x-axis starting point for SLAP
        min_val_slap = top_slap.min()
        if min_val_slap > 0.1:
            ax1.set_xlim(left=min_val_slap - 0.1)

        # --- Plot SHAP ---
        bars2 = ax2.barh(top_shap.index, top_shap.values)
        ax2.set_title(f"Top 10 SHAP Feature Importance - {self.title}")
        ax2.set_xlabel("Mean |SHAP value|")
        ax2.set_ylabel("Feature")
        ax2.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax2.bar_label(bars2, fmt='%.2f', padding=3)

        # Set x-axis starting point for SHAP
        min_val_shap = top_shap.min()
        if min_val_shap > 0.01:
            ax2.set_xlim(left=min_val_shap - 0.01)

        plt.tight_layout()
        save_path = self.save_path / 'Feature Importance png'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"SLAP_SHAP Combined Feature Importance {self.title}.png")
        plt.close(fig)


def generate_summary_report(
    X, y,
    save_path='tempClassifier',
    reduce_list=['EBA', 'PCA', 'HDF', 'L1SS'],
    imbalance_list=[None, 'smote', 'borderline','noise_aug'],
    feature_weights=None,
    classifier_list=['KNN', 'Prototypical', 'DecisionTree',
                     'RandomForest', 'LogisticRegression', 'SVM', 'Voting']):
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for imbalance_strategy in imbalance_list:

        imbalance_name = (
            "no_resampling"
            if imbalance_strategy is None
            else imbalance_strategy
        )

        final_report = pd.DataFrame()
        report_f1 = []
        report_accuracy = []
        report_precision = []
        report_recall = []

        for reduce in reduce_list:
            f1_row, acc_row, prec_row, rec_row = [], [], [], []

            for classifier_type in classifier_list:

                filename = (
                    f"{classifier_type}"
                    f"_{reduce}"
                    f"_{imbalance_name}.csv"
                )
                filepath = save_path / filename

                if filepath.exists():
                    report_df = pd.read_csv(filepath, index_col=0)
                else:
                    classifier = ClassifierTrainer(
                        X, y,
                        save_path=save_path,
                        reduce=reduce,
                        feature_weights=feature_weights,
                        classifier_type=classifier_type,
                        imbalance_strategy=imbalance_strategy
                    )
                    _, report_df, _, _ = classifier.train()
                    print(
                        f"Generated result for {classifier_type} | "
                        f"{reduce} | {imbalance_name}"
                    )

                result = report_df.loc[['avg']]
                result.index = [f'{classifier_type} with {reduce}']
                final_report = pd.concat([final_report, result])

                f1_row.append(report_df.loc['avg', 'f1-score'])
                acc_row.append(report_df.loc['avg', 'accuracy'])
                prec_row.append(report_df.loc['avg', 'precision'])
                rec_row.append(report_df.loc['avg', 'recall'])

            report_f1.append(f1_row)
            report_accuracy.append(acc_row)
            report_precision.append(prec_row)
            report_recall.append(rec_row)

        # Restore original table structure
        f1_df = pd.DataFrame(report_f1, index=reduce_list, columns=classifier_list).T
        acc_df = pd.DataFrame(report_accuracy, index=reduce_list, columns=classifier_list).T
        prec_df = pd.DataFrame(report_precision, index=reduce_list, columns=classifier_list).T
        rec_df = pd.DataFrame(report_recall, index=reduce_list, columns=classifier_list).T

        excel_name = f"report_{imbalance_name}.xlsx"
        excel_path = save_path / excel_name

        with ExcelWriter(excel_path) as writer:
            final_report.to_excel(writer, sheet_name='Final Report')
            f1_df.to_excel(writer, sheet_name='F1-score')
            acc_df.to_excel(writer, sheet_name='Accuracy')
            prec_df.to_excel(writer, sheet_name='Precision')
            rec_df.to_excel(writer, sheet_name='Recall')

        print(f"[Saved] {excel_name}")


if __name__ == "__main__":
        
    from feature_extraction.DatabaseConnection import get_latest_output_save_path
    latest_time, data_path = get_latest_output_save_path(output_root=Path("./output"))

    #data_name = 'all_features'
    #data_name = 'scenario_2_features'

    for data_name in ['all_features','scenario_1_features','scenario_1.1_features','scenario_1.2_features','scenario_2_features','scenario_3_features','scenario_4_features','scenario_5_features']:

        feature_path = data_path/str('data')/f'{data_name}.csv'
        data = pd.read_csv(feature_path)
        save_path = data_path/ f'results_{data_name}'

        data = process_missing_values(data, save_path = save_path)

        X = data.loc[:, ~data.columns.str.match(r'^(id|label)', case=False)]
        y = data['label']

        #feature_weights = [1]*127+[0]*(len(X.columns)-127)
        feature_weights = None

        #analyze_features(X, y, save_path=save_path/ 'Feature Selection')
        #for reduce in ['EBA','PCA','HDF','L1SS']:
        #    feature = FeatureSelection(X,y,reduce,save_path=save_path/ 'Feature Selection',feature_weights=feature_weights)
        #    X_processed,y_processed = feature.process()

        generate_summary_report(X,y,save_path=save_path/'Test Classifiers',reduce_list=['L1SS'],imbalance_list=['noise_aug'], classifier_list=['KNN','SVM', 'Voting'],feature_weights=feature_weights) 

