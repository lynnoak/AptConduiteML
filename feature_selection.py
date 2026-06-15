import matplotlib
matplotlib.use("TkAgg")

from feature_extraction.DatabaseConnection import myDatabase

from pathlib import Path

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.feature_selection import f_classif

from scipy.stats import f_oneway
from collections import defaultdict

import matplotlib.pyplot as plt

def show_pca_2d(X, y, label_column='label', title="", save_path=Path("./")):
    """
    Project high-dimensional data into 2D using PCA and plot using either category markers or continuous score color.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (samples x features) to be projected.
    y : array-like or pd.Series
        Class labels or score values corresponding to each sample.
    label_column : str
        Used for naming and legend titles.
    title : str
        Extra text in title and file name.
    save_path : Path or str
        Directory to save the image.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))

    y_series = pd.Series(y)

    if pd.api.types.is_numeric_dtype(y_series):
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                              c=y_series, cmap='coolwarm',
                              alpha=0.8, edgecolors='k')
        plt.colorbar(scatter, label=label_column)
    else:
        label_encoder = LabelEncoder()
        label_ids = label_encoder.fit_transform(y_series)
        unique_labels = label_encoder.classes_
        markers = ['o', 's', '^', 'D', 'P', 'X', '*']

        for i, label in enumerate(np.unique(label_ids)):
            mask = label_ids == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        marker=markers[i % len(markers)],
                        label=unique_labels[label],
                        alpha=0.8, edgecolors='k')
        plt.legend(title=label_column, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Projection Coded by {label_column} {title}")
    plt.grid(True)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"PCA Projection Coded by {label_column} {title}.png")
    plt.close()

def analyze_features(X, y, save_path='tempFeatureAnalyze'):
    """
    Analyze feature correlation and class-wise difference.

    Parameters:
    - X: pd.DataFrame, feature matrix
    - y: array-like, class labels
    - save_path: directory to save plots and results
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    y_series = pd.Series(y)
    if y_series.dtype == 'object' or str(y_series.dtype) == 'category':
        y_series = pd.Series(LabelEncoder().fit_transform(y_series), index=y_series.index)

    unique_labels = np.sort(y_series.dropna().unique())
    n_classes = len(unique_labels)

    # 1. Pearson correlation
    corr_matrix = X.corr(method='pearson')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, fmt=".2f")
    plt.title("Feature Pearson Correlation")
    plt.tight_layout()
    plt.savefig(save_path / 'feature_correlation_heatmap.png')
    plt.close()
    corr_matrix.to_csv(save_path / 'feature_correlation_matrix.csv')

    # 2. Binary classification: t-test
    if n_classes == 2:
        df = X.copy()
        df['label'] = y_series.values

        group0 = df[df['label'] == unique_labels[0]]
        group1 = df[df['label'] == unique_labels[1]]

        t_stats, p_values = [], []
        for col in X.columns:
            try:
                t_stat, p_val = ttest_ind(
                    group0[col],
                    group1[col],
                    nan_policy='omit'
                )
                t_stats.append(t_stat)
                p_values.append(p_val)
            except Exception:
                t_stats.append(np.nan)
                p_values.append(np.nan)

        diff_df = pd.DataFrame({
            'Feature': X.columns,
            'T-statistic': t_stats,
            'P-value': p_values
        }).sort_values(by='P-value', ascending=True)

        diff_df.to_csv(save_path / 'feature_group_difference_ttest.csv', index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='T-statistic',
            y='Feature',
            data=diff_df.head(15),
            color='skyblue'
        )
        plt.title("Top 15 Differentiating Features (t-test)")
        plt.axvline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(save_path / 'top_differentiating_features_ttest.png')
        plt.close()

    # 3. Multi-class classification: ANOVA
    elif n_classes > 2:
        F_values, p_vals = f_classif(X, y_series)
        anova_df = pd.DataFrame({
            'Feature': X.columns,
            'F-score': F_values,
            'P-value': p_vals
        }).sort_values(by='F-score', ascending=False)

        anova_df.to_csv(save_path / 'feature_group_difference_anova.csv', index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='F-score',
            y='Feature',
            data=anova_df.head(15),
            color='salmon'
        )
        plt.title("Top 15 Features by ANOVA F-score")
        plt.tight_layout()
        plt.savefig(save_path / 'top_differentiating_features_anova.png')
        plt.close()

    else:
        raise ValueError("At least two classes are required for feature analysis.")

    print(f"Analysis complete. Results saved in: {save_path}")

def process_missing_values(
    data,
    save_path=None,
    row_threshold=1/3,
    col_threshold=1/3,
    label_col="label"
):
    """
    Missing value handling.

    Strategy:
    1. Drop rows with too many missing values.
    2. Drop columns with too many missing values.
    3. Fill remaining missing values (mean/mode).
    """

    df = data.copy()

    # Row-level filtering
    feature_cols = [c for c in df.columns if c != label_col]

    row_missing_ratio = df[feature_cols].isna().mean(axis=1)

    df = df[row_missing_ratio < row_threshold]


    # Column-level filtering
    n_rows = len(df)
    dropped_cols = []

    for col in feature_cols:

        missing_ratio = df[col].isna().sum() / n_rows

        if missing_ratio >= col_threshold:
            dropped_cols.append(col)

    # Fill remaining missing values
    for col in df.columns:
        if col == label_col:
            continue

        if df[col].isna().sum() > 0:

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
            else:
                fill_value = df[col].mode().iloc[0]

            df[col] = df[col].fillna(fill_value)

    df = df.reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path / "data_cleaned.csv", index=False)

    return df

def redundancy_feature_remove(X, y=None, feature_corr_threshold=0.9, target_corr_threshold=None, method='pearson'):
    """
    Select features based on inter-feature and feature-target correlation.

    Parameters
    ----------
    y : array-like, optional
        Target labels for supervised selection (default: None).
    feature_corr_threshold : float
        Threshold above which two features are considered highly correlated (default: 0.9).
    target_corr_threshold : float or None
        Minimum correlation with target to retain features (default: None).
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall' (default: 'pearson').
    """
    X = X.copy()
    # Step 1: Optionally drop features weakly correlated with target
    if y is not None and target_corr_threshold is not None:
        target_corr = X.corrwith(pd.Series(y), method=method).abs()
        weak_features = target_corr[target_corr < target_corr_threshold].index.tolist()
        X.drop(columns=weak_features, inplace=True)

    # Step 2: Compute feature correlation matrix
    corr_matrix = X.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Step 3: Find highly correlated feature pairs and drop the less relevant one
    to_drop = set()
    if y is not None:
        target_corr = X.corrwith(pd.Series(y), method=method).abs()
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > feature_corr_threshold:
                # Compare correlation with target if available
                if y is not None:
                    if target_corr[col] >= target_corr[row]:
                        to_drop.add(row)
                    else:
                        to_drop.add(col)
                else:
                    # Arbitrarily drop one of the correlated features
                    to_drop.add(col)

    X_filtered = X.drop(columns=list(to_drop))
    return X_filtered

def grouping_by_prefix(df):
    """
    Group columns by the first part of their name before '_'.
    For each group, compute the row-wise mean if it contains multiple columns.
    If only one column is present, keep it unchanged.
    """
    prefix_groups = defaultdict(list)

    # Step 1: Build prefix groups
    for col in df.columns:
        prefix = col.split('_')[0]
        prefix_groups[prefix].append(col)

    #print(prefix_groups)
    # Step 2: Aggregate
    grouped_data = {}
    for prefix, columns in prefix_groups.items():
        if len(columns) > 1:
            grouped_data[prefix] = df[columns].mean(axis=1)
        else:
            grouped_data[prefix] = df[columns[0]] 

    return pd.DataFrame(grouped_data)

class FeatureSelection:
    """
    Handles data preprocessing, feature selection, and dimensionality reduction.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features).
    y : pd.Series or array-like
        Target labels.
    reduce : str, optional
        Dimensionality reduction method ['EBA','PCA','HDF','L1SS']. Default is 'None'.
    feature_weights : list, optional
        Weights for each of the column for the features.
    """
    def __init__(self, X,y, reduce = 'None', save_path = 'tempFeatureSelection', feature_weights=None):
        self.X = X
        self.y = y
        self.reduce = reduce
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.pca= None
        self.feature_weights = feature_weights
        self.scaler = None

    def apply_feature_weights(self, df):
        """
        Apply column-wise feature weights to the given DataFrame.
        Supports list and pandas Series.
        Columns with zero weight will be dropped.
        """
        if self.feature_weights is not None:
            if isinstance(self.feature_weights, list):
                if len(self.feature_weights) != df.shape[1]:
                    raise ValueError(f"Length of feature_weights list ({len(self.feature_weights)}) "
                                     f"does not match number of features ({df.shape[1]}).")
                weights = pd.Series(self.feature_weights, index=df.columns)

            elif isinstance(self.feature_weights, pd.Series):
                weights = self.feature_weights.copy()
                if not weights.index.equals(df.columns):
                    raise ValueError("feature_weights Series index must exactly match df.columns.")

            else:
                raise TypeError("feature_weights must be a list or pandas Series.")

            # --- Drop zero-weight columns ---
            zero_weight_cols = weights[weights == 0].index.tolist()
            if zero_weight_cols:
                #print(f"[apply_feature_weights] Removing {len(zero_weight_cols)} features with zero weight.")
                df = df.drop(columns=zero_weight_cols, errors='ignore')
                weights = weights.drop(index=zero_weight_cols)

            # --- Apply weighting ---
            df = df * weights.loc[df.columns]

        return df
    
    def process(self):

        self.scaler = StandardScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns, index=self.X.index)

        #print("Before correlation, X shape:", self.X.shape)
        #self.X = redundancy_feature_remove(self.X, y=None)
        #print("After correlation, X shape:", self.X.shape)

        self.X = self.apply_feature_weights(self.X)

        self.X, self.y = self.dimension_reduce(self.X)
        print(f"After reduction with {self.reduce}, X shape: {self.X.shape}")

        analyze_features(self.X, self.y, save_path=self.save_path / self.reduce)
        return self.X, self.y

    def dimension_reduce(self, df=None):
        if df is None:
            df = self.X.copy()

        reduce_methods = {
            'HDF': self.dimension_reduce_by_label,
            'EBA': self.dimension_reduce_by_group,
            'L1SS': self.dimension_reduce_by_L1SS,
            'PCA': self.dimension_reduce_by_PCA,
            'None': self.dimension_reduce_identity
        }

        reduce_method = reduce_methods.get(self.reduce, self.dimension_reduce_by_PCA)
        self.reduce = self.reduce if self.reduce in reduce_methods else 'PCA'
        df_reduced = reduce_method(df)

        save_path = self.save_path / self.reduce
        save_path.mkdir(parents=True, exist_ok=True)
        df_reduced.to_csv(save_path / f'{self.reduce}.csv')

        return df_reduced, self.y
    
    def dimension_reduce_identity(self, df):
        """No feature selection; return input as-is."""
        return df

    def transform_test(self, X_test):

        X_test = self.apply_feature_weights(X_test)

        if self.reduce == 'PCA':
            transformed = self.pca.transform(X_test)
            return pd.DataFrame(transformed, columns=self.pca_col, index=X_test.index)

        elif self.reduce == 'L1SS':
            return X_test[self.selected_columns]

        elif self.reduce == 'EBA':
            return grouping_by_prefix(X_test)

        elif self.reduce == 'HDF':
            return X_test[self.selected_columns]
        
        elif self.reduce == 'None':
            return X_test

        else:
            raise ValueError(f"Unsupported reduce strategy: {self.reduce}")

    def dimension_reduce_by_label(self, df, var_threshold=0.01, p_value_threshold=0.1, min_features=10):
        y = pd.Series(self.y).reindex(df.index)

        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)

        variances = df.var(axis=0)
        selected_var_features = variances[variances >= var_threshold].index.tolist()
        if len(selected_var_features) < min_features:
            top_k = min(min_features, len(variances))
            selected_var_features = variances.sort_values(ascending=False).head(top_k).index.tolist()

        df_var_filtered = df[selected_var_features]

        feature_p_values = {}
        for feature in selected_var_features:
            try:
                grouped_data = [df_var_filtered.loc[y == label, feature] for label in np.unique(y)]
                if any(len(group) < 2 for group in grouped_data):
                    continue
                f_stat, p_value = f_oneway(*grouped_data)
                feature_p_values[feature] = p_value
            except ValueError:
                continue

        sorted_features = sorted(feature_p_values.items(), key=lambda x: x[1])
        selected_anova_features = [f for f, p in sorted_features if p < p_value_threshold]

        if len(selected_anova_features) < min_features:
            if sorted_features:
                top_k = min(min_features, len(sorted_features))
                selected_anova_features = [f for f, _ in sorted_features[:top_k]]
            else:
                top_k = min(min_features, len(selected_var_features))
                selected_anova_features = selected_var_features[:top_k]

        df_selected = df_var_filtered[selected_anova_features]
        self.selected_columns = df_selected.columns
        return df_selected

    def dimension_reduce_by_group(self, df):
        df_grouped = grouping_by_prefix(df)
        return df_grouped

    def dimension_reduce_by_PCA(self, df, variance_threshold=0.9, top=3):
        save_path = self.save_path / 'PCA' / 'PCA_optimal'
        save_path.mkdir(parents=True, exist_ok=True)

        pca_full = PCA()
        pca_full.fit(df)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        component_labels = [f"PC{i+1}" for i in range(len(cumulative_variance))]

        plt.figure(figsize=(8, 5))
        plt.plot(component_labels, cumulative_variance, marker='o', linestyle='-', color='b')
        plt.axhline(y=variance_threshold, color='r', linestyle='--')
        plt.xlabel("Principal Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("Cumulative Explained Variance by Principal Components")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path / "Cumulative Explained Variance")
        plt.close()

        n_optimal = np.argmax(cumulative_variance >= variance_threshold) + 1
        n_optimal = max(n_optimal,3)
        pca = PCA(n_components=n_optimal)
        transformed_data = pca.fit_transform(df)

        descriptions = []
        for i, comp in enumerate(pca.components_):
            sorted_pairs = sorted(zip(comp, df.columns), key=lambda x: abs(x[0]), reverse=True)
            top_terms = [f"{w:+.2f}*{f}" for w, f in sorted_pairs[:top]]
            descriptions.append(f"PC{i+1} ≈ " + " ".join(top_terms))

        df_pca = pd.DataFrame(transformed_data, columns=descriptions, index=df.index)
        importance = np.abs(pca.components_).sum(axis=0)
        importance_df = pd.DataFrame({'Feature': df.columns, 'Importance': importance})
        importance_df.to_csv(save_path / 'Feature Importance for PCA.csv', index=False)

        self.pca = pca
        self.pca_col = descriptions
        return df_pca

    def dimension_reduce_by_L1SS(self, df):
        n_classes = pd.Series(self.y).nunique()

        if n_classes >= 3:
            l1_clf = LogisticRegression(
                penalty='l1',
                solver='saga',
                max_iter=5000,
                random_state=0
            )
        else:
            l1_clf = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=5000,
                random_state=0
            )

        selector = SelectFromModel(l1_clf, prefit=False, threshold='mean')
        selector.fit(df, self.y)
        selected_cols = df.columns[selector.get_support()]
        df_selected = df[selected_cols]
        self.selected_columns = selected_cols
        return df_selected

"""
if __name__ == "__main__":

    db = myDatabase()
    df = db.get_collection('evaluation')
    df = df.sort_values(by='eval_date')
    latest_time = df['eval_date'].iloc[-1]
    latest_time = latest_time.strftime('%Y-%m-%d_%H_%M_%S').split('.')[0]
    save_path = Path("./output")/ str(latest_time)

    feature_path = save_path/str('data')/'all_features.csv'

    header = pd.read_csv(feature_path, nrows=0)
    unnamed_columns = [col for i, col in enumerate(header.columns) if col.startswith('Unna')]
    data = pd.read_csv(feature_path, usecols=lambda column: column not in unnamed_columns)


    data['GroupLabel'] = "Health"
    #data.loc[data['id_patient'].str.contains('fake', case=False, na=False), 'GroupLabel'] = 'NV'
    data.loc[data['id_patient'].str.contains('fake', case=False, na=False), 'GroupLabel'] = 'Fake'
    #data = data[~data['id_patient'].str.contains('fake', case=False, na=False)].reset_index(drop=True)
    data = data[~data['id_patient'].str.contains('P01-02')].reset_index(drop=True)

    data.loc[data['id_patient'] == 'P01-04 P01-04', 'GroupLabel'] = "NV"
    data.loc[data['id_patient'] == 'P01-10 P01-10', 'GroupLabel'] = "NV"
    data.loc[data['id_patient'] == 'P01-14 P01-14', 'GroupLabel'] = "NV"
    data.loc[data['id_patient'] == 'P01-15 P01-15', 'GroupLabel'] = "NV"


    data.loc[data['id_patient'] == 'P01-18 P01-18', 'GroupLabel'] = "HD"
    data.loc[data['id_patient'] == 'P01-19 P01-19', 'GroupLabel'] = "HD"


    data = feature_selection.process_missing_values(data, save_path = save_path)
    data_ana = data.iloc[:,3:]
    save_path = save_path/ f'results_{data_name}'

    X = data.iloc[:,3:-1]
    y = data.iloc[:,-1]

    analyze_features(X, y, save_path=save_path)

    feature_weights = [0]*55+[1]*(len(X.columns)-55)

    for reduce in ['EBA','PCA','HDF','L1SS']:
        feature = FeatureSelection(X,y,reduce,save_path=save_path,feature_weights=feature_weights)
        X_processed,y_processed = feature.process()

"""   

