import feature_extraction.DataPhy as DPhy
import feature_extraction.DataSim as DSim
import feature_extraction.DataPre as DPre
from feature_extraction.DatabaseConnection import myDatabase, get_latest_output_save_path
from bson import ObjectId
from pathlib import Path
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import os


# =========================
# Scenario configuration
# =========================

SCENARIO_LIST = [
    'scenario_1',
    'scenario_2',
    'scenario_3',
    'scenario_4',
    'scenario_5'
]

SIMULATOR_SCENARIOS = {
    'scenario_1',
    'scenario_2',
    'scenario_3',
    'scenario_4'
}


# =========================
# Feature file names
# =========================

LABELS = "label.csv"

FEATURE_PHY = 'featurePhy.csv'
FEATURE_SIM_GENERAL = 'featureSim.csv'
FEATURE_SIM_SC = 'featureSimSC.csv'

FEATURES_GENERAL = [
    FEATURE_PHY,
    FEATURE_SIM_GENERAL
]

FEATURES_SIM_SCENARIO = [
    FEATURE_PHY,
    FEATURE_SIM_SC
]

FEATURES_PHY_ONLY = [
    FEATURE_PHY
]


DISEASE_PATIENTS = {
    'P01-04 P01-04',
    'P01-10 P01-10',
    'P01-14 P01-14',
    'P01-15 P01-15'
}

FAKE_PATIENTS = 'FAKE'


def extract_features_from_mongodb(show_plot=False, only_latest=False):
    """
    Retrieve raw multimodal evaluation data from MongoDB and perform
    scenario-wise feature extraction.

    For each participant and scenario:
    - Extract simulator features (DataSim) when applicable
    - Extract physiological features (DataPhy)
    - Store structured feature files in timestamped directories

    Parameters
    ----------
    show_plot : bool
        Whether to display signal plots during processing.
    only_latest : bool
        If True, process only the most recent evaluation session.
    """
    if show_plot == False:
        import matplotlib
        matplotlib.use("Agg")
    try:
        db = myDatabase()
        print("Successfully connected to MongoDB.")

        df = db.get_collection('evaluation').sort_values(by='eval_date')
        latest_time = df['eval_date'].iloc[-1].strftime('%Y-%m-%d_%H_%M_%S').split('.')[0]
        save_path = Path("./output") / latest_time/str('data')

        if only_latest:
            df = df.iloc[[-1]]

        for index, row in df.iterrows():
            patient_id = row['eval_pat']
            patient = db.get_collection('patient', query={"_id": ObjectId(patient_id)})
            name = patient['pat_nom_naissance'].iloc[0] + ' ' + patient['pat_prenom'].iloc[0]
            if name == 'Rabreau Olivier':
                continue

            patient_save_path = save_path / name
            print(f"Analysis the data from participant {name} at {row['eval_date']}")

            for scenario_n in SCENARIO_LIST:
                eval_scenario = db.get_collection(scenario_n, query={"sc_eval": ObjectId(row['_id'])})
                scenario_n_save_path = patient_save_path / scenario_n

                for _, scenario_row in eval_scenario.iterrows():
                    scenario_id = str(scenario_row['_id'])
                    sc_config = scenario_row['sc_config']['conf_name']
                    if "démo" in sc_config.lower():
                        continue
        
                    print(f"Data processing for {scenario_n} with config {sc_config} as {scenario_id}")

                    scenario_id_save_path = scenario_n_save_path / f"{sc_config} {scenario_id}"
                    if os.path.exists(scenario_id_save_path):
                        continue

                    if scenario_n in SIMULATOR_SCENARIOS:
                        SimData = DSim.DataSim(scenario_n, scenario_id, save_path=scenario_id_save_path)
                        SimData.process(show_plot=show_plot)

                    PhyData = DPhy.DataPhy(scenario_id, save_path=scenario_id_save_path)
                    PhyData.show(show_plot=show_plot)
                    PhyData.process(show_plot=show_plot)

        db.close()
    except Exception as e:
        print(e)
        with open('error_log.txt', 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] Unexpected error: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")

def save_extracted_features_to_csv(scenario="all", labels_csv=LABELS):
    """
    Aggregate previously extracted feature files and generate
    scenario-level structured datasets for machine learning.

    This function:
    - Loads featurePhy.csv and simulator feature files
    - Merges them with questionnaire features (DataPre)
    - Assigns labels (Health / Disease / Fake)
    - Generates scenario-specific feature CSV files

    Parameters
    ----------
    scenario : str
        "all", "general", or specific scenario name.
        Determines which datasets to aggregate.

    labels_csv : str or Path or None
        Optional path to an external CSV file containing labels.
        Supported formats:
        - columns: ['id', 'label']
        - columns: ['id_anonymat', 'label']

        If provided and valid, labels are assigned according to this file.
        Otherwise, fallback to the default rule-based labeling.
    """
    try:
        processor = DPre.DataPre(
            file_path=DPre.FILE_PATH,
            save_path=DPre.SAVE_PATH
        )
        data_pre = processor.run()

        latest_time, save_path = get_latest_output_save_path(output_root=Path("./output"))
        save_path = save_path / "data"

        if not save_path.exists():
            raise Exception("There is no data processed or not processing the latest data!")
    except Exception as e:
        print(e)

    # -------------------------------------------------
    # Optional external label mapping
    # -------------------------------------------------
    label_mode = None
    label_mapping = None

    if labels_csv is not None:
        labels_csv = Path(labels_csv)
        if labels_csv.is_file():
            df_labels = pd.read_csv(labels_csv)

            # Normalize column names
            df_labels.columns = [str(c).strip() for c in df_labels.columns]

            if "label" not in df_labels.columns:
                print("[WARNING] labels CSV exists but has no 'label' column. Fallback to default labeling.")
            elif "id" in df_labels.columns:
                df_labels["id"] = df_labels["id"].astype(str).str.strip()
                df_labels["label"] = df_labels["label"].astype(str).str.strip()
                label_mapping = dict(zip(df_labels["id"], df_labels["label"]))
                label_mode = "id"
                print(f"[INFO] External labels loaded by 'id' from: {labels_csv}")
            elif "id_anonymat" in df_labels.columns:
                df_labels["id_anonymat"] = df_labels["id_anonymat"].astype(str).str.strip()
                df_labels["label"] = df_labels["label"].astype(str).str.strip()
                label_mapping = dict(zip(df_labels["id_anonymat"], df_labels["label"]))
                label_mode = "id_anonymat"
                print(f"[INFO] External labels loaded by 'id_anonymat' from: {labels_csv}")
            else:
                print("[WARNING] labels CSV must contain either ['id', 'label'] or ['id_anonymat', 'label']. Fallback to default labeling.")
        else:
            print(f"[WARNING] labels CSV not found: {labels_csv}. Fallback to default labeling.")

    def get_label(id_value, id_anonymat_value, patient_name):
        """
        Determine label using:
        1. external labels by id
        2. external labels by id_anonymat
        3. fallback default rules
        """
        if label_mode == "id" and label_mapping is not None:
            label = label_mapping.get(str(id_value).strip(), None)
            if label is not None and label != "":
                return label

        if label_mode == "id_anonymat" and label_mapping is not None:
            label = label_mapping.get(str(id_anonymat_value).strip(), None)
            if label is not None and label != "":
                return label

        # Fallback to original logic
        if FAKE_PATIENTS in patient_name:
            return "Fake"
        elif patient_name in DISEASE_PATIENTS:
            return "Disease"
        else:
            return "Health"

    def process_one_scenario(scenario_n):
        # Decide which feature files to load for this scenario
        if scenario_n == "general":
            features_scenario_list = [
                FEATURE_PHY,
                FEATURE_SIM_GENERAL
            ]
        elif scenario_n in SIMULATOR_SCENARIOS:
            features_scenario_list = [
                FEATURE_PHY,
                FEATURE_SIM_SC
            ]
        else:
            features_scenario_list = [
                FEATURE_PHY
            ]

        features_all = []

        for patient_path in save_path.iterdir():
            if not patient_path.is_dir():
                continue

            if scenario_n == "general":
                scenario_dirs = [d for d in patient_path.iterdir() if d.is_dir()]
            else:
                scenario_dir = patient_path / scenario_n
                if not scenario_dir.exists() or not scenario_dir.is_dir():
                    continue
                scenario_dirs = [scenario_dir]

            for scenario_n_path in scenario_dirs:
                for scenario_id_path in scenario_n_path.iterdir():
                    if not scenario_id_path.is_dir():
                        continue

                    features = pd.DataFrame()
                    skip_folder = False

                    for feature_file in features_scenario_list:
                        features_path = scenario_id_path / feature_file
                        if not features_path.is_file():
                            skip_folder = True
                            break

                        temp = pd.read_csv(features_path)

                        if temp.empty or temp.shape[0] != 1:
                            skip_folder = True
                            break

                        features = temp if features.empty else pd.concat([features, temp], axis=1)

                    if skip_folder or features.empty:
                        continue

                    parts = scenario_id_path.name.rsplit(maxsplit=1)
                    id_value = str(parts[1]).strip()
                    id_anonymat = str(patient_path.name.split()[0]).strip()
                    patient_name = str(patient_path.name).strip()

                    label = get_label(
                        id_value=id_value,
                        id_anonymat_value=id_anonymat,
                        patient_name=patient_name
                    )

                    id_df = pd.DataFrame([{
                        'id': id_value,
                        'id_test': parts[0],
                        'id_patient': patient_name,
                        'id_scenario': scenario_n_path.name,
                        'id_anonymat': id_anonymat,
                        'label': label
                    }])

                    features = pd.concat([id_df, features], axis=1)
                    features_all.append(features)

        if len(features_all) == 0:
            print(f"[WARNING] No valid features found for scenario: {scenario_n}")
            return

        features_all = pd.concat(features_all, axis=0, ignore_index=True)
        features_all = features_all.merge(data_pre, on="id_anonymat", how="left")

        suffix = 'all_features.csv' if scenario_n == "general" else f"{scenario_n}_features.csv"
        features_all.to_csv(save_path / suffix, index=False)

        # -------------------------------------------------
        # Extra split for scenario_1 based on id_test
        # -------------------------------------------------
        if scenario_n == "scenario_1":

            for sub_id in ["1", "2"]:

                sub_df = features_all[
                    features_all["id_test"].astype(str).str.contains(sub_id, na=False)
                ].copy()

                if sub_df.empty:
                    continue

                # Remove numeric columns that are entirely NaN
                numeric_cols = sub_df.select_dtypes(include=[np.number]).columns
                cols_to_drop = [
                    col for col in numeric_cols
                    if sub_df[col].isna().all()
                ]

                sub_df.drop(columns=cols_to_drop, inplace=True)
                sub_df.drop(
                    columns=[
                        "SIM_Rea_cor_pos_5.0",
                        "SIM_Num_cor_pos_5"
                    ],
                    errors="ignore",
                    inplace=True
                )

                sub_suffix = f"scenario_1.{sub_id}_features.csv"
                sub_df.to_csv(save_path / sub_suffix, index=False)

    if scenario == "all":
        process_one_scenario("general")
        for scenario_n in SCENARIO_LIST:
            process_one_scenario(scenario_n)
    elif scenario == "general":
        process_one_scenario("general")
    elif scenario in SCENARIO_LIST:
        process_one_scenario(scenario)



if __name__ == "__main__":
    extract_features_from_mongodb()
    save_extracted_features_to_csv()
