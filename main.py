# -*- coding: utf-8 -*-
# NOTE: Use module-level imports + importlib.reload to pick up edits automatically.

import importlib
from pathlib import Path
import pandas as pd

from feature_extraction.DatabaseConnection import myDatabase  # keep as-is
from feature_extraction.DatabaseConnection import get_latest_output_save_path

import feature_extract
import feature_selection
import classifier_analysis
import clustering_analysis
import pattern_mining_old

# ---- force reload to reflect latest edits when re-running in the same interpreter ----
# This is useful when you press the "Run" triangle in VS Code which may reuse the session.
importlib.reload(feature_extract)
importlib.reload(feature_selection)
importlib.reload(classifier_analysis)
importlib.reload(clustering_analysis)
importlib.reload(pattern_mining_old)


"""
if __name__ == "__main__":
# Prompt user for options
    show_plot_input = input("Display plots? (y/n): ").strip().lower()
    show_plot = True if show_plot_input == 'y' else False

    only_lastest = input("Lastest evalution? (y/n): ").strip().lower()
    only_lastest = True if only_lastest == 'y' else False
else:
    show_plot = False
    only_lastest = False
"""

show_plot = False
only_lastest = False

"""Reading data"""
feature_extract.extract_features_from_mongodb(show_plot, only_lastest)
feature_extract.save_extracted_features_to_csv()

latest_time, data_path = get_latest_output_save_path(output_root=Path("./output"))
#data_path = Path("./output/2025-07-24_18_38_54")

"""Selecting scenario"""
#data_name = 'scenario_2_features'

for data_name in ['all_features','scenario_1_features','scenario_2_features','scenario_3_features','scenario_4_features','scenario_5_features']:
    

    feature_path = data_path/str('data')/f'{data_name}.csv'
    data = pd.read_csv(feature_path, encoding='latin1')
    save_path = data_path/ f'results_{data_name}'

    data = feature_selection.process_missing_values(data, save_path = save_path)

    #data.to_csv('data_temp.csv',index=False)

    X = data.loc[:, ~data.columns.str.match(r'^(id|label)', case=False)]
    y = data['label']

    """Selecting features"""
    """

    def build_feature_weights(
        feature_names,
        exact_weights=None,
        keyword_weights=None,
        default_weight=0.0,
        case_sensitive=False):

        if exact_weights is None:
            exact_weights = {}

        if keyword_weights is None:
            keyword_weights = {}

        weights = []

        for feat in feature_names:
            feat_cmp = feat if case_sensitive else feat.lower()

            # 1. Exact match (highest priority)
            if feat in exact_weights:
                weights.append(exact_weights[feat])
                continue

            # 2. Keyword match
            assigned = False
            for key, w in keyword_weights.items():
                key_cmp = key if case_sensitive else key.lower()
                if key_cmp in feat_cmp:
                    weights.append(w)
                    assigned = True
                    break

            # 3. Default
            if not assigned:
                weights.append(default_weight)

        return weights

    exact_feature_weights = {
        "EEG_Delta_PO8": 1.500,
        "EEG_Theta_AF7": 1.464,
        "EYE_Saccade_rate_right_area_left": 1.429,
        "EEG_Beta_O2": 1.393,
        "EYE_Blink_rate_left_area_background": 1.357,
        "ECG_Rate_Mean": 1.321,
        "EYE_Pupil_dilation_left_area_upper": 1.286,
        "EYE_Blink_rate_right_area_center": 1.250,
        "HEAD_V_eulerY": 1.214,
        "ECG_HRV_pNN50": 1.179,
        "PRE_glasses_status": 1.143,
        "EEG_Beta_O1": 1.107,
        "EYE_Fixation_rate_left_area_background": 1.071,
        "ECG_HRV_HF": 1.036,
        "ECG_HRV_SD1": 1.000,
    }



    feature_weights = build_feature_weights(
        feature_names=X.columns.tolist(),
        exact_weights=exact_feature_weights,
        default_weight= 0.8
    )

    feature_selection.analyze_features(X, y, save_path=save_path/ 'Feature Selection')
    for reduce in ['EBA','PCA','HDF','L1SS']:
        feature = feature_selection.FeatureSelection(X,y,reduce,save_path=save_path/ 'Feature Selection',feature_weights=feature_weights)
        X_processed,y_processed = feature.process()

    classifier_analysis.generate_summary_report(X,y,reduce_list=['None'],save_path=save_path/'Classifiers Training',feature_weights=feature_weights)
    """

    feature_weights = None
    #classifier_analysis.generate_summary_report(X,y,save_path=save_path/'Classifiers Training') 

    patient = data['id_patient']
    #clustering_analysis.generate_clustering_report(X,y,patient,param_search=True,save_path = save_path/'Clustering Analysis')


    if data_name == 'scenario_5_features': continue

    data = pattern_mining_old.aggregate_eeg_if_needed(data)

    pipeline = pattern_mining_old.PatternMining(
        ante_keywords=['EYE','HEAD','ECG','EEG','EDA'],
        cons_keywords=['SIM'],
        label_col='label',
        label_mode='binary_health_vs_others',
        save_path=save_path/'Pattern Mining'
    )

    rules = pipeline.process(data)

