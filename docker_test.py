# run_pattern_mining_docker.py
from pathlib import Path
import pandas as pd
import feature_selection
import pattern_mining_old

INPUT_CSV = Path("./input/scenario_1_features.csv")
OUTPUT_DIR = Path("./output/pattern_test")

data = pd.read_csv(INPUT_CSV, encoding="latin1")
data = feature_selection.process_missing_values(data, save_path=OUTPUT_DIR)
data = pattern_mining_old.aggregate_eeg_if_needed(data)

pipeline = pattern_mining_old.PatternMining(
    ante_keywords=['EYE', 'HEAD', 'ECG', 'EEG', 'EDA'],
    cons_keywords=['SIM'],
    label_col='label',
    label_mode='binary_health_vs_others',
    save_path=OUTPUT_DIR
)

rules = pipeline.process(data)
print("Pattern mining finished.")