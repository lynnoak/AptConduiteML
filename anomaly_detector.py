import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from pathlib import Path

from feature_extraction.DatabaseConnection import myDatabase
from feature_selection import show_pca_2d,process_missing_values, redundancy_feature_remove,grouping_by_prefix
from feature_selection import FeatureSelection


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold,LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import shap

from sklearn.ensemble import IsolationForest  
from sklearn.svm import OneClassSVM  
from sklearn.covariance import EllipticEnvelope  
from sklearn.neighbors import LocalOutlierFactor  


db = myDatabase()
df = db.get_collection('evaluation')
df = df.sort_values(by='eval_date')
latest_time = df['eval_date'].iloc[-1]
latest_time = latest_time.strftime('%Y-%m-%d_%H_%M_%S').split('.')[0]
save_path = Path("./output")/ str(latest_time)

feature_path = save_path/'all_features.csv'

header = pd.read_csv(feature_path, nrows=0)
unnamed_columns = [col for i, col in enumerate(header.columns) if col.startswith('Unna')]
data = pd.read_csv(feature_path, usecols=lambda column: column not in unnamed_columns)

data['permis'] = "Health"
#data = data.loc[data['scenario'] == 'scenario_2']
data.loc[data['patient'] == 'P01-04 P01-04', 'permis'] = "Disease"
data.loc[data['patient'] == 'P01-10 P01-10', 'permis'] = "Disease"
data.loc[data['patient'] == 'P01-14 P01-14', 'permis'] = "Disease"
data.loc[data['patient'] == 'P01-15 P01-15', 'permis'] = "Disease"
data.loc[data['patient'] == 'P01-FAKE-01 P01-FAKE-01', 'permis'] = "Disease"
data.loc[data['patient'] == 'P01-FAKE-02 P01-FAKE-02', 'permis'] = "Disease"
data.loc[data['patient'] == 'P01-FAKE-03 P01-FAKE-03', 'permis'] = "Disease"


data = process_missing_values(data, save_path = save_path)
X = data.iloc[:,3:-1]
y =data.iloc[:,-1]

X = redundancy_feature_remove(X)


class AnomalyDetectorTrainer:
    """
    Train an anomaly detector with optional cross-validation (if labels provided),
    dimensionality reduction, and SLAP/SHAP feature analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features).
    y : pd.Series or array-like
        Labels for evaluation (only used if use_labels=True).
    split : str
        Cross-validation strategy ('StratifiedKFold', 'RepeatedStratifiedKFold', 'LeaveOneOut').
    reduce : str
        Feature reduction method used ( 'Grouping', 'PCA' 'Selecting' or 'L1').
    detector_type : str
        Anomaly detector type: 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope', 'LOF'.
    detector_params : dict or None
        Parameters for the detector.
    use_labels : bool
        Whether to use labels for cross-validation (True = semi-supervised, False = unsupervised).
    save_path : Path
        Path to save results.
    show : bool
        Whether to visualize PCA and feature importances.
    random_state : int
        Random seed.
    title : str
        Custom title for result labeling.
    """
    def __init__(self, X, y=None, split='RepeatedStratifiedKFold', reduce='PCA',
                 detector_type='IsolationForest', detector_params=None,
                 use_labels=True, save_path=Path("results"), show=True,
                 random_state=42, title=''):

        self.X = X
        self.y = y
        self.split = split
        self.reduce = reduce
        self.detector_type = detector_type
        self.detector_params = detector_params or {}
        self.use_labels = use_labels
        self.save_path = save_path / 'anomaly_detection'
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.show = show
        self.random_state = random_state
        self.title = title or f"{detector_type} for {reduce} features ({'Supervised' if use_labels else 'Unsupervised'})"

        if self.use_labels and self.y is not None:
            y_series = pd.Series(self.y)
            most_common_label = y_series.value_counts().idxmax()
            self.y = y_series.apply(lambda x: 0 if x == most_common_label else 1)

        self.splitters = {
            'StratifiedKFold': StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state),
            'LeaveOneOut': LeaveOneOut(),
            'RepeatedStratifiedKFold': RepeatedStratifiedKFold(n_splits=3, n_repeats=4, random_state=random_state),
        }

        if use_labels and split not in self.splitters:
            raise ValueError(f"Unsupported split strategy: {split}")

    def train(self):
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        reducer = FeatureSelection(X_scaled, self.y, reduce=self.reduce, save_path=self.save_path)
        X_reduced, _ = reducer.dimension_reduce()
        print(f"Training for {self.title}")

        if self.use_labels:
            return self._train_with_cv(X_reduced)
        else:
            return self._train_without_labels(X_reduced)

    def _train_with_cv(self, X):
        rkf = self.splitters[self.split]
        metrics = []
        slap_matrix = pd.DataFrame()
        shap_matrix = pd.DataFrame()
        scores_series = pd.Series(index=self.X.index, dtype=float)
        final_model = None

        for fold, (train_idx, test_idx) in enumerate(rkf.split(X, self.y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            model = self._get_detector()
            model.fit(X_train)
            y_pred = model.predict(X_test)

            y_pred_bin = (y_pred == 1).astype(int)
            y_true_bin = (y_test != 0).astype(int)

            metrics.append(self._evaluate(y_true_bin, y_pred_bin))
            slap = self._compute_slap(X_test, model)
            slap_matrix = pd.concat([slap_matrix, slap], ignore_index=True)
            shap_ = self._compute_shap(X_test, model)
            shap_matrix = pd.concat([shap_matrix, shap_], ignore_index=True)

            try:
                fold_scores = model.decision_function(X_test)
            except:
                fold_scores = model.predict(X_test)
            scores_series.iloc[test_idx] = fold_scores

            final_model = model

        df_metrics = pd.DataFrame(metrics)
        avg_metrics = df_metrics.mean().to_frame().T
        df_report = pd.concat([avg_metrics, df_metrics], ignore_index=True)
        df_report.index = ['avg'] + [f'fold{i+1}' for i in range(len(metrics))]
        df_report = df_report * 100
        df_report = df_report.round(2)

        slap_scores = slap_matrix.mean()
        shap_scores = shap_matrix.mean()

        report_path = self.save_path / 'Performance Report'
        report_path.mkdir(parents=True, exist_ok=True)
        df_report.to_csv(report_path / f"{self.title}.csv")

        score_path = self.save_path / 'Supervised Scores'
        score_path.mkdir(parents=True, exist_ok=True)
        scores_series.name = 'score'
        scores_series.to_csv(score_path / f"{self.title}_all_scores.csv")
        score_stats = scores_series.describe().to_frame(name='Score')
        score_stats.to_csv(score_path / f"{self.title}_score_summary.csv")

        fig, ax = plt.subplots(figsize=(8, 4))
        scores_series.hist(bins=50, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f"Score Distribution - {self.title}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(score_path / f"{self.title}_score_distribution.png")
        plt.close()

        if self.show:
            show_pca_2d(X, y, label_column='label', title=self.title, save_path=score_path)
            show_pca_2d(X, scores_series, label_column='score', title=self.title, save_path=score_path)

        self._visualize_feature_importance(slap_scores, shap_scores)
        return final_model, df_report, slap_scores

    def _train_without_labels(self, X):
        model = self._get_detector()
        model.fit(X)

        try:
            scores = model.decision_function(X)
        except:
            scores = model.predict(X)

        score_series = pd.Series(scores, index=self.X.index, name='score')
        score_path = self.save_path / 'Unsupervised Scores'
        score_path.mkdir(parents=True, exist_ok=True)

        score_series.to_csv(score_path / f"{self.title}_all_scores.csv")
        score_series.describe().to_frame(name='Score').to_csv(score_path / f"{self.title}_score_summary.csv")

        fig, ax = plt.subplots(figsize=(8, 4))
        score_series.hist(bins=50, ax=ax, color='lightcoral', edgecolor='black')
        ax.set_title(f"Unsupervised Score Distribution - {self.title}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(score_path / f"{self.title}_score_distribution.png")
        plt.close()

        if self.show:
            show_pca_2d(X, y, label_column='label', title=self.title, save_path=score_path)
            show_pca_2d(X, score_series, label_column='score', title=self.title, save_path=score_path)

        return model, score_series, None
    
    def _get_detector(self):
        detectors = {
            'IsolationForest': IsolationForest(contamination='auto', random_state=self.random_state, **self.detector_params),
            'OneClassSVM': OneClassSVM(**self.detector_params),
            'EllipticEnvelope': EllipticEnvelope(random_state=self.random_state, **self.detector_params),
            'LOF': LocalOutlierFactor(novelty=True, **self.detector_params)
        }
        return detectors[self.detector_type]

    def _evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

    def _compute_slap(self, X, model):
        drops = {}
        y_pred = model.predict(X)
        for col in X.columns:
            X_perturbed = X.copy()
            X_perturbed[col] = 0
            y_perturbed = model.predict(X_perturbed)
            drops[col] = (y_pred != y_perturbed).mean()
        return pd.DataFrame([drops])

    def _compute_shap(self, X_test, model):
        X_test_df = X_test.astype(float).copy()
        feature_names = X_test_df.columns
        background = shap.sample(X_test_df, 100) if len(X_test_df) > 100 else X_test_df

        def predict_fn(x):
            try:
                return model.decision_function(pd.DataFrame(x, columns=feature_names))
            except Exception:
                return model.predict(pd.DataFrame(x, columns=feature_names))

        explainer = shap.KernelExplainer(predict_fn, background.values)
        shap_vals = explainer.shap_values(X_test_df.values, nsamples=100)

        if isinstance(shap_vals, np.ndarray):
            shap_array = np.abs(shap_vals)
        elif isinstance(shap_vals, list) and isinstance(shap_vals[0], np.ndarray):
            shap_array = np.mean(np.abs(np.stack(shap_vals, axis=0)), axis=0)
        else:
            raise ValueError("Unexpected SHAP output format")

        return pd.DataFrame(shap_array, columns=feature_names).mean().to_frame().T


    def _visualize_feature_importance(self, slap_scores, shap_scores):
        top_slap = slap_scores.sort_values(ascending=False).head(10)
        top_shap = shap_scores.sort_values(ascending=False).head(10)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        top_slap.iloc[::-1].plot.barh(ax=ax1, title="SLAP", legend=False)
        top_shap.iloc[::-1].plot.barh(ax=ax2, title="SHAP", legend=False)

        ax1.set_xlabel("SLAP Drop Rate")
        ax2.set_xlabel("Mean |SHAP|")
        plt.tight_layout()
        fig.savefig(self.save_path / f"FeatureImportance_{self.title}.png")
        plt.close()

    def _visualize_pca(self, X, labels):
        show_pca_2d(X, pd.DataFrame({'label': labels}), 'label', self.title, self.save_path)
    def _visualize_pca(self, X, labels):
        show_pca_2d(X, pd.DataFrame({'label': labels}), 'label', self.title, self.save_path)


for reduce in ['PCA','Grouping','Selecting','L1']:
    for detector_type in ['IsolationForest', 'OneClassSVM', 'EllipticEnvelope', 'LOF']:
        for use_label in [True,False]:
            trainer = AnomalyDetectorTrainer(X, y, 
                                 use_labels=use_label, 
                                 reduce=reduce, 
                                 detector_type= detector_type)
            model, report, slap = trainer.train()
