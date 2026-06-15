from feature_extraction.DatabaseConnection import myDatabase
from bson import ObjectId, errors

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import neurokit2 as nk

from dataclasses import dataclass

# Channel name list for eeg
CHANNEL_NAMES = ["AF7", "Fp1", "Fp2", "AF8", "PO7", "O1", "O2", "PO8"]

# Dictionary to map signal columns to their respective y-axis labels
Y_AXIS_LABELS_WITH_UNITS = {
    'eeg': 'Voltage (µV)',
    'ecg': 'Voltage (mV)',
    'eda': 'Conductance (µS)',
    'euler': 'Angle (degrees)',
    'rate': 'Frequency (Hz)',
    'pupil': 'Velocity (mm/s)',
    'gaze': 'coordinate'
}

# selected features for ECG analysis.
ECG_FEATURE_ALIASES = {
    'ECG_Rate_Mean': 'ECG_Rate_Mean',              # Mean heart rate
    'ECG_HRV_SDNN': 'HRV_SDNN',                    # Standard deviation of NN intervals
    'ECG_HRV_pNN50': 'HRV_pNN50',                  # Percentage of successive RR intervals > 50 ms
    'ECG_HRV_LF': 'HRV_LF',                        # Low-frequency power
    'ECG_HRV_HF': 'HRV_HF',                        # High-frequency power
    'ECG_HRV_SampEn': 'HRV_SampEn',                # Sample entropy
}

#selected features for EDA analysis
EDA_FEATURE_ALIASES = {
    'EDA_SCR_Peaks_N': 'SCR_Peaks_N',                           # Number of detected skin conductance responses (SCRs)
    'EDA_SCR_Peaks_Amplitude_Mean': 'SCR_Peaks_Amplitude_Mean',# Mean amplitude of SCR peaks
    'EDA_Tonic_SD': 'EDA_Tonic_SD',                             # Standard deviation of tonic (baseline) EDA component
    'EDA_SympatheticN': 'EDA_SympatheticN'                     # Normalized Sympathetic nervous system activation (phasic)
}

#Compute mean absolute velocity of pupil ignoring NaN transitions.
def pupil_velocity(pupil_array):

    pupil_array = np.asarray(pupil_array)
    valid = ~np.isnan(pupil_array)

    if valid.sum() < 2:
        return np.nan

    valid_values = pupil_array[valid]
    valid_indices = np.where(valid)[0]

    diffs = np.abs(np.diff(valid_values))
    index_diffs = np.diff(valid_indices)
    diffs = diffs[index_diffs == 1]


    return np.nanmean(diffs) if len(diffs) > 0 else np.nan

@dataclass
class Config:
    # Sampling rate for each sensor
    sampling_rate_Bitbrain: int = 256
    sampling_rate_Bitalino: int = 100
    sampling_rate_Tobii: int = 60
    sampling_rate_Bno055: int = 10

    # Processing eye tracking slider parameters
    window_size: float = 1.0 #The size of the sliding window in seconds.
    step_size: float = 0.1 #The step size for sliding the window in seconds.

    # Screen size
    screen_x: int = 1170 # in mm 
    screen_y:int = 335 # in mm 
    screen_center:int = 120 # in mm 

    #Processing eye tracking thresholds 
    fixation_threshold: float=0.01 #The threshold for determining fixation (gaze stability).
    saccade_threshold: float=0.05 #The threshold for determining saccades (rapid eye movements).

# Create a Config instance
config = Config()

def clean_gaze_data(df):
    # All eye gaze columns (both x and y for left/right eyes)
    gaze_cols = [
        'sc_value_x_left_eye_gaze_point_display_area',
        'sc_value_y_left_eye_gaze_point_display_area',
        'sc_value_x_right_eye_gaze_point_display_area',
        'sc_value_y_right_eye_gaze_point_display_area'
    ]

    # Subset of y-direction columns
    gaze_y_cols = [
        'sc_value_y_left_eye_gaze_point_display_area',
        'sc_value_y_right_eye_gaze_point_display_area'
    ]

    # Identify invalid rows (any gaze column < 0 or > 1)
    invalid_mask = df[gaze_cols].lt(0) | df[gaze_cols].gt(1)
    invalid_rows = invalid_mask.any(axis=1)

    total_rows = len(df)
    invalid_count = invalid_rows.sum()
    invalid_ratio = invalid_count / total_rows

    if invalid_ratio <= 0.01:
        # Case 1: low invalid ratio, delete all invalid rows
        df_cleaned = df[~invalid_rows].copy()
    else:
        # Case 2: high invalid ratio, delete top 1% most deviated rows
        deviation = df[gaze_cols].apply(lambda x: np.maximum(0 - x, x - 1)).abs().max(axis=1)
        top_k = int(0.01 * total_rows)
        top_bad_indices = deviation.sort_values(ascending=False).head(top_k).index
        df_cleaned = df.drop(index=top_bad_indices).copy()

        # Only here, rescale y gaze columns to [0, 1]
        for col in gaze_y_cols:
            col_min = df_cleaned[col].min()
            col_max = df_cleaned[col].max()
            if col_max > col_min:
                df_cleaned[col] = (df_cleaned[col] - col_min) / (col_max - col_min)
            else:
                df_cleaned[col] = 0.5  # Fallback if constant

    return df_cleaned.reset_index(drop=True)


class DataPhy:
    """
    Data Collected from physiological sensors
    parameters: 
        sc_scenario: the id of the scenario being queried 
    """
    def __init__(self, sc_scenario,save_path = None):

        self.db_client = myDatabase()

        try:
            self.sc_scenario = ObjectId(sc_scenario)
        except errors.InvalidId:
            raise ValueError(f"Invalid ObjectId: {sc_scenario}")
        
        if not save_path:
            self.save_path = Path("./tempPhy") / str(sc_scenario)
        else:
            self.save_path = save_path

        self.save_path.mkdir(parents=True, exist_ok=True)

        # Load all data
        self.bitalino, self.ecg_eda = self._load_data("bitalino")
        if self.ecg_eda.empty:
            self.ecg = pd.DataFrame([])
            self.eda = pd.DataFrame([])
        else:
            self.ecg = self.bitalino['sc_ecg']
            self.eda = self.bitalino['sc_eda']
        self.bitbrain, self.eeg = self._load_data("bitbrain")
        self.bno055,self.euler = self._load_data("bno055")
        self.tobii, self.eye = self._load_data("tobii")

        self.db_client.close()
        self.features = {}

    def _load_data(self, collection_name):

        df = self.db_client.get_collection(collection_name,query={"sc_scenario": self.sc_scenario})
        
        if df.empty:
            print(f'No {collection_name} data for {self.sc_scenario}!')
            return pd.DataFrame([]), pd.DataFrame([])

        df.drop(columns=['_id', 'sc_scenario'], errors='ignore', inplace=True)

        if collection_name == "bitalino":
            # explode the ecg and eda values since they are in Array(10)
            df = df.explode(['sc_ecg','sc_eda']).reset_index(drop=True)

        if collection_name == "bitbrain":
            df = df[['sc_time','sc_eeg_values']]
            # explode the eeg values since they are in Array(64)
            for i in range(8):
                df['sc_eeg_'+CHANNEL_NAMES[i]] = df['sc_eeg_values'].apply(lambda x: x[i*8:i*8+7])
            df.drop(columns=['sc_eeg_values'], inplace=True)
            explode_columns = df.columns.tolist()[1:]
            df = df.explode(explode_columns).reset_index(drop=True)

        if collection_name == 'bno055':
            df = df

        print(f'There are {len(df)} rows of {collection_name} data loaded for {self.sc_scenario}.')
        df.to_csv(self.save_path / f'{collection_name}.csv', index=False)
        return df, df.drop(columns=['sc_time'], errors='ignore')
        
    def show_bitalino(self, show_plot = False):
        if self.bitalino.empty:
            print(f'There is no bitalino data to show for {self.sc_scenario} !')
        else:
            show_signals(self.ecg_eda, signal_type='ecg_eda', save_path=self.save_path/ "ecg_eda",show_plot= show_plot)

    def show_bitbrain(self, show_plot = False):
        if self.bitbrain.empty:
            print(f'There is no bitbrain data to show for {self.sc_scenario} !')
        else:
            show_signals(self.eeg, signal_type='euler', save_path=self.save_path/ "eeg",show_plot= show_plot)
    
    def show_bno055(self, show_plot = False):
        if self.bno055.empty:
            print(f'There is no bno055 data to show for {self.sc_scenario} !')
        else:
            show_signals(self.euler, signal_type='euler', save_path=self.save_path/ "euler",show_plot= show_plot)

    def show_tobii(self, show_plot = False):
        if self.tobii.empty:
            print(f'There is no tobii data to show for {self.sc_scenario} !')
        else:
            eye = self.eye.fillna(0)
            show_signals(eye, signal_type='eye', save_path=self.save_path/ "eye",show_plot= show_plot)

    def show(self, show_plot = False):
        self.show_bitalino(show_plot= show_plot)
        self.show_bitbrain(show_plot= show_plot)
        self.show_bno055(show_plot= show_plot)
        self.show_tobii(show_plot= show_plot)
        plt.close('all')

    def process_ecg(self, show_plot = False):
        if self.ecg.empty:
            print(f'There is no ecg data to process for {self.sc_scenario} !')
            return pd.DataFrame([]), {}
        # Preprocess ECG signal with nk
        ecg = np.array(self.ecg, dtype=np.float64)
        sampling_rate=config.sampling_rate_Bitalino

        try:
            ecg_processed, ecg_processed_info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
            ecg_processed.to_csv(self.save_path/ 'ecg_processed.csv')

            # Visualize
            nk.ecg_plot(ecg_processed, ecg_processed_info)
            fig = plt.gcf()
            fig.set_size_inches(10, 12, forward=True)
            fig.set_dpi(300)
            fig.savefig(self.save_path/ 'ecg_nk')
            if show_plot:
                plt.show(block=True)
            
            self.ecg_processed = ecg_processed
            self.ecg_processed_info = ecg_processed_info

            # Compute the features
            results = nk.ecg_analyze(ecg_processed, sampling_rate=sampling_rate)
            results.to_csv(self.save_path/ 'ecg_analyze.csv')
            # Store selected and renamed features
            for new_name, original_col in ECG_FEATURE_ALIASES.items():
                if original_col in results.columns:
                    value = results[original_col].iloc[0]
                    if isinstance(value, list):
                        value = value[0]
                    elif isinstance(value, (np.ndarray, pd.Series)):
                        value = value.item()
                    self.features[new_name] = value
                else:
                    self.features[new_name] = np.nan

            return ecg_processed
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            return pd.DataFrame([])

    def process_eda(self, show_plot = False):
        if self.eda.empty:
            print(f'There is no eda data to process for {self.sc_scenario} !')
            return pd.DataFrame([]), {}
        # Preprocess EDA signal with nk
        eda = np.array(self.eda, dtype=np.float64)
        sampling_rate=config.sampling_rate_Bitalino
        try:
            eda_processed, eda_processed_info = nk.eda_process(eda, sampling_rate=sampling_rate)
            eda_processed.to_csv(self.save_path/ 'eda_processed.csv')

            # Visualize
            nk.eda_plot(eda_processed, eda_processed_info)
            fig = plt.gcf()
            fig.set_size_inches(10, 12, forward=True)
            fig.set_dpi(300)
            fig.savefig(self.save_path/ 'eda_nk')
            if show_plot:
                plt.show(block=True)
            
            self.eda_processed = eda_processed
            self.eda_processed_info = eda_processed_info

            #Compute the features
            results = nk.eda_analyze(eda_processed, sampling_rate=sampling_rate)
            results.to_csv(self.save_path/ 'eda_analyze.csv')
            # Extract and rename features
            for new_name, original_col in EDA_FEATURE_ALIASES.items():
                if original_col in results.columns:
                    value = results[original_col].iloc[0]
                    if isinstance(value, list):
                        value = value[0]
                    elif isinstance(value, (np.ndarray, pd.Series)):
                        value = value.item()
                    self.features[new_name] = value
                else:
                    self.features[new_name] = np.nan
            return eda_processed
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            return pd.DataFrame([])
 
    def process_eeg(self, show_plot = False):
        if self.eeg.empty:
            print(f'There is no eeg data to process for {self.sc_scenario} !')
            return pd.DataFrame([]), {}
        # Preprocess EEG signal with nk
        eeg = self.eeg.copy()
        eeg = eeg.astype(float)

        sampling_rate = config.sampling_rate_Bitbrain

        psd = nk.eeg_power(eeg,sampling_rate=sampling_rate, frequency_band=["Gamma", "Beta", "Alpha", "Theta", "Delta"])
        psd['Channel'] = psd['Channel'].str.replace(r'^sc_eeg_', '', regex=True)
        psd.to_csv(self.save_path/ 'eeg_nk_psd.csv')
        self.eeg_psd = psd

        melted_df = psd.melt(id_vars='Channel', var_name='parameter', value_name='value')
        melted_df['key'] = 'EEG_'+melted_df['parameter'] + '_' + melted_df['Channel']
        result_dict = dict(zip(melted_df['key'], melted_df['value']))

        self.features.update(result_dict)
        return psd

    def process_eye(self, show_plot=False):

        if self.eye.empty:
            print(f'There is no eye data to process for {self.sc_scenario}!')
            return pd.DataFrame(), {}

        # Load configuration and clean gaze data
        data = clean_gaze_data(self.eye)

        sampling_rate = config.sampling_rate_Tobii
        window_samples = int(config.window_size * sampling_rate)
        step_samples = int(config.step_size * sampling_rate)

        fixation_threshold = config.fixation_threshold
        saccade_threshold = config.saccade_threshold

        screen_x = config.screen_x
        screen_y = config.screen_y
        screen_center_radius = config.screen_center

        # Extract pupil signals
        pupil_left = data['sc_value_left_pupil'].to_numpy()
        pupil_right = data['sc_value_right_pupil'].to_numpy()

        blink_left_events = np.diff(np.isnan(pupil_left).astype(int), prepend=0) == 1
        blink_right_events = np.diff(np.isnan(pupil_right).astype(int), prepend=0) == 1

        # Compute mean absolute pupil velocity ignoring NaN transitions
        def pupil_velocity(pupil_array):
            pupil_array = np.asarray(pupil_array)
            valid = ~np.isnan(pupil_array)

            if valid.sum() < 2:
                return np.nan

            valid_values = pupil_array[valid]
            valid_indices = np.where(valid)[0]

            diffs = np.abs(np.diff(valid_values))
            index_diffs = np.diff(valid_indices)
            diffs = diffs[index_diffs == 1]

            return np.nanmean(diffs) if len(diffs) > 0 else np.nan

        # Compute gaze shift magnitude
        def get_gaze_shift(eye):
            x = data[f'sc_value_x_{eye}_eye_gaze_point_display_area'].to_numpy()
            y = data[f'sc_value_y_{eye}_eye_gaze_point_display_area'].to_numpy()

            aspect_ratio = screen_x / screen_y
            x_scaled = x * aspect_ratio

            dx = np.diff(x_scaled, prepend=np.nan)
            dy = np.diff(y, prepend=np.nan)

            shift = np.sqrt(dx**2 + dy**2)
            shift /= np.sqrt(aspect_ratio**2 + 1)

            return np.nan_to_num(shift, nan=-1)

        gaze_shift_left = get_gaze_shift('left')
        gaze_shift_right = get_gaze_shift('right')

        saccade_left = gaze_shift_left > saccade_threshold
        saccade_right = gaze_shift_right > saccade_threshold

        fixation_left = (gaze_shift_left > 0) & (gaze_shift_left < fixation_threshold)
        fixation_right = (gaze_shift_right > 0) & (gaze_shift_right < fixation_threshold)

        fixation_left_changes = np.diff(fixation_left.astype(int), prepend=0) == 1
        fixation_right_changes = np.diff(fixation_right.astype(int), prepend=0) == 1

        n_samples = len(data)
        start_indices = np.arange(0, n_samples - window_samples + 1, step_samples)

        metrics = []
        center_flags = []
        left_flags = []
        upper_flags = []

        for start in start_indices:
            end = start + window_samples

            blink = blink_left_events[start:end].sum() + \
                    blink_right_events[start:end].sum()

            fixation_l = fixation_left_changes[start:end].sum()
            fixation_r = fixation_right_changes[start:end].sum()
            fixation = (fixation_l + fixation_r) / 2

            saccade_l = saccade_left[start:end].sum()
            saccade_r = saccade_right[start:end].sum()
            saccade = (saccade_l + saccade_r) / 2

            fixation_diff = fixation_l - fixation_r
            saccade_diff = saccade_l - saccade_r

            pupil_l = pupil_velocity(pupil_left[start:end])
            pupil_r = pupil_velocity(pupil_right[start:end])

            valid_values = [v for v in [pupil_l, pupil_r] if not np.isnan(v)]

            if len(valid_values) == 0:
                pupil = np.nan
            elif len(valid_values) == 1:
                pupil = valid_values[0]
            else:
                pupil = sum(valid_values) / len(valid_values)

            valid_ratio_left = np.mean(~np.isnan(pupil_left[start:end]))
            valid_ratio_right = np.mean(~np.isnan(pupil_right[start:end]))

            if valid_ratio_left < 0.2 and valid_ratio_right < 0.2:
                pupil = np.nan

            x = data['sc_value_x_left_eye_gaze_point_display_area'].iloc[start:end]
            y = data['sc_value_y_left_eye_gaze_point_display_area'].iloc[start:end]

            x_mean = np.nanmean(x) if not np.isnan(x).all() else np.nan
            y_mean = np.nanmean(y) if not np.isnan(y).all() else np.nan

            if np.isnan(x_mean) or np.isnan(y_mean):
                in_center = False
                is_left = False
                is_upper = False
            else:
                x_pos = x_mean * screen_x
                y_pos = y_mean * screen_y
                x_center = screen_x / 2
                y_center = screen_y / 2

                center_distance = np.sqrt((x_pos - x_center)**2 +
                                        (y_pos - y_center)**2)

                in_center = center_distance < screen_center_radius
                is_left = x_pos < x_center
                is_upper = y_pos < y_center

            metrics.append([
                blink,
                fixation,
                saccade,
                fixation_diff,
                saccade_diff,
                pupil
            ])

            center_flags.append(in_center)
            left_flags.append(is_left)
            upper_flags.append(is_upper)

        columns = [
            'EYE_Blink_rate',
            'EYE_Fixation_rate',
            'EYE_Saccade_rate',
            'EYE_Fixation_LR_diff',
            'EYE_Saccade_LR_diff',
            'EYE_Pupil_velocity'
        ]

        metrics_df = pd.DataFrame(metrics, columns=columns)

        metrics_df['in_center'] = center_flags
        metrics_df['is_left'] = left_flags
        metrics_df['is_upper'] = upper_flags

        for col in columns:
            self.features[col] = metrics_df[col].mean()

        self.frontal_eye_times = None

        metrics_df.to_csv(self.save_path / 'eye_processed.csv', index=False)

        show_signals(
            metrics_df[columns],
            signal_type='eye',
            save_path=self.save_path / 'eye_processed',
            show_plot=show_plot
        )

        try:
            raw_eye = self.tobii

            dx = raw_eye['sc_value_x_left_gaze_origin_in_user_coordinate_system'] - \
                raw_eye['sc_value_x_right_gaze_origin_in_user_coordinate_system']

            dy = raw_eye['sc_value_y_left_gaze_origin_in_user_coordinate_system'] - \
                raw_eye['sc_value_y_right_gaze_origin_in_user_coordinate_system']

            dz = raw_eye['sc_value_z_left_gaze_origin_in_user_coordinate_system'] - \
                raw_eye['sc_value_z_right_gaze_origin_in_user_coordinate_system']

            iod = np.sqrt(dx**2 + dy**2 + dz**2)
            valid_iod = iod[~np.isnan(iod)]

            if len(valid_iod) > 20:

                iod_median = np.median(valid_iod)

                if iod_median > 0:

                    iod_norm = iod / iod_median
                    threshold = np.nanpercentile(iod_norm, 90)

                    frontal_mask = iod_norm >= threshold

                    gaze_x = raw_eye['sc_value_x_left_eye_gaze_point_display_area']
                    gaze_y = raw_eye['sc_value_y_left_eye_gaze_point_display_area']

                    center_mask = (
                        (np.abs(gaze_x - 0.5) < 0.05) &
                        (np.abs(gaze_y - 0.5) < 0.05)
                    )

                    candidate_mask = frontal_mask & center_mask

                    candidate_times = raw_eye.loc[candidate_mask, 'sc_time']

                    if len(candidate_times) > 10:
                        self.frontal_eye_times = pd.to_datetime(candidate_times)
                        print("Frontal eye candidates stored.")

        except Exception as e:
            print("Frontal detection failed in eye processing.", e)

        return metrics_df

    def process_head(self, show_plot=False):
        """
        Head processing with:
        - Quaternion continuity correction
        - Euler unwrap
        - Eye-based frontal zero calibration (3D IOD + center fixation)
        - Time-aligned cross-modal synchronization
        """

        if self.bno055.empty:
            print(f'There is no head data to process for {self.sc_scenario}!')
            return pd.DataFrame(), {}

        # Use raw head data WITH time
        data = self.bno055.copy()

        sampling_rate = config.sampling_rate_Bno055  # Hz
        window_samples = int(config.window_size * sampling_rate)
        step_samples = int(config.step_size * sampling_rate)

        euler_cols = ['sc_eulerX', 'sc_eulerY', 'sc_eulerZ']
        quat_cols = ['sc_quatW', 'sc_quatX', 'sc_quatY', 'sc_quatZ']

        #  Quaternion continuity correction
        if all(col in data.columns for col in quat_cols):

            q = data[quat_cols].values.copy()

            for i in range(1, len(q)):
                if np.dot(q[i-1], q[i]) < 0:
                    q[i] *= -1.0

            data[quat_cols] = q

        # Euler unwrap (remove 360° wrapping)
        for col in euler_cols:
            if col in data.columns:
                rad = np.radians(data[col].values)
                rad_unwrapped = np.unwrap(rad)
                data[col] = np.degrees(rad_unwrapped)

        # Default zero position (first second)
        default_zero = data.iloc[0:sampling_rate][euler_cols].mean()
        zero_position = default_zero.copy()


        # Use eye-based frontal times if available
        if hasattr(self, 'frontal_eye_times') and self.frontal_eye_times is not None:

            head_times = pd.to_datetime(data['sc_time']).values.astype('datetime64[ns]')
            eye_times = self.frontal_eye_times.values.astype('datetime64[ns]')

            # Convert to int64 nanoseconds for fast math
            head_ns = head_times.astype('int64')
            eye_ns = eye_times.astype('int64')

            # Find insertion indices
            idx = np.searchsorted(head_ns, eye_ns)

            # Clip to valid range
            idx = np.clip(idx, 1, len(head_ns)-1)

            # Compare neighbor distances
            left_diff = np.abs(eye_ns - head_ns[idx - 1])
            right_diff = np.abs(eye_ns - head_ns[idx])

            nearest_idx = np.where(left_diff < right_diff, idx - 1, idx)

            # Apply tolerance (10Hz → 100ms)
            tolerance_ns = int(1e9 / config.sampling_rate_Bno055)

            valid_mask = np.abs(head_ns[nearest_idx] - eye_ns) <= tolerance_ns

            matched_head_indices = np.unique(nearest_idx[valid_mask])

            if len(matched_head_indices) > config.sampling_rate_Bno055:
                zero_position = data.iloc[matched_head_indices][euler_cols].median()
                print("Eye-based frontal zero calibration applied.")
            else:
                print("Frontal candidates insufficient for head alignment. Using default zero.")

        else:
            print("No frontal eye candidates found. Using default zero.")

        #  Sliding window computation
        deviation_list = []
        velocity_list = []

        for start in range(0, len(data) - window_samples + 1, step_samples):

            window = data.iloc[start:(start + window_samples)]

            # Mean deviation
            window_mean = window[euler_cols].mean()
            deviation = window_mean - zero_position
            deviation_list.append(deviation)

            # Mean absolute angular velocity (deg/s)
            velocity = window[euler_cols].diff().abs().mean() * sampling_rate
            velocity_list.append(velocity)

        deviation_df = pd.DataFrame(deviation_list)[euler_cols]
        velocity_df = pd.DataFrame(velocity_list)[euler_cols]

        deviation_df.columns = deviation_df.columns.str.replace('sc_', 'HEAD_', regex=False)
        velocity_df.columns = velocity_df.columns.str.replace('sc_', 'HEAD_V_', regex=False)

        result_df = pd.concat([deviation_df, velocity_df], axis=1)

        result_df.to_csv(self.save_path / 'head_processed.csv', index=False)

        show_signals(
            deviation_df,
            signal_type='euler',
            save_path=self.save_path / 'head_processed',
            show_plot=show_plot
        )

        # Save global mean features
        for col in result_df.columns:
            self.features[col] = result_df[col].mean()

        return result_df

    def process(self, show_plot = False):
            self.process_ecg(show_plot= show_plot)
            self.process_eda(show_plot= show_plot)
            self.process_eeg(show_plot= show_plot)
            self.process_eye(show_plot= show_plot)
            self.process_head(show_plot=show_plot)

            plt.close('all')

            data = self.features
            if len(data) == 0:
                return pd.DataFrame()

            # ensure scalar values
            for key, value in data.items():
                if isinstance(value, (np.ndarray, list)):
                    data[key] = np.nanmean(value)

            # ONE-ROW DataFrame
            df = pd.DataFrame([data])

            df.to_csv(self.save_path / "featurePhy.csv", index=False)
            return df


def show_signals(data, signal_type='eeg', save_path='signal_plot', sampling_rate=None, show_plot=False):
    """
    Directly show EEG, ECG, or EDA signals with matplotlib.
    
    Parameters:
    - data: DataFrame containing the signal data.
    - signal_type: 
            'eeg' for EEG signals, 
            'ecg_eda' for ECG and EDA signals.
            'eye' for eye tracking signals.
            'euler' for head tracking signals
    - save_path: Path to save the plotted figure.
    - sampling_rate: Sampling rate of the signal.
    """

    # Dictionary mapping signal types to their respective sampling rates
    sampling_rates = {
        'eeg': config.sampling_rate_Bitbrain,
        'ecg_eda': config.sampling_rate_Bitalino,
        'eye': config.sampling_rate_Tobii,
        'euler': config.sampling_rate_Bno055
    }

    # Set sampling rate based on signal_type
    if signal_type not in sampling_rates:
        print(f"Warning: Unknown signal_type '{signal_type}'.")
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            print(f"Error: Invalid sampling_rate '{sampling_rate}'. It must be a positive number.")
            return  # Exit the function if sampling_rate is invalid
    else:
        sampling_rate = sampling_rates[signal_type]

    # Extract time axis
    n_samples = len(data)
    times = np.arange(n_samples) / sampling_rate
    signal_columns = list(data.columns)

    # Safety limit for number of subplots
    max_channels_to_plot = 30
    n_channels = len(signal_columns)

    if n_channels > max_channels_to_plot:
        print(f"[WARNING] Too many channels ({n_channels}), only plotting first {max_channels_to_plot} to avoid overflow.")
        signal_columns = signal_columns[:max_channels_to_plot]
        n_channels = max_channels_to_plot

    # Dynamically determine figure height
    height_per_plot = 1.8
    max_fig_height = 40  # inches
    fig_height = min(height_per_plot * n_channels, max_fig_height)

    # Create figure with adaptive subplots
    fig, axs = plt.subplots(n_channels, 1, figsize=(12, fig_height), sharex=True)
    if n_channels == 1:
        axs = [axs]  # Ensure axs is iterable
    fig.subplots_adjust(hspace=0.3)

    for i, (ax, col) in enumerate(zip(axs, signal_columns)):
        signal = data[col]
        ax.plot(times, signal, lw=1)
        ax.set_title(f'{col}', pad=2, fontsize=10)
        ax.grid(True)

        # Set y-axis label based on signal type or column name
        for keyword, label in Y_AXIS_LABELS_WITH_UNITS.items():
            if keyword in col.lower():
                ax.set_ylabel(label)
                break
        else:
            ax.set_ylabel("")  # Default y-axis label

        # Set y-axis limits based on signal type
        if 'ecg' in col.lower():
            ax.set_ylim(-1.5, 1.5)  # ECG y-axis range
        elif 'eda' in col.lower():
            ax.set_ylim(0, 25)  # EDA y-axis range

    axs[-1].set_xlabel("Time (seconds)")

    # Add slider only for small number of subplots
    if n_channels <= 10:
        ax_slider = plt.axes([0.1, 0.02, 0.8, 0.02], facecolor='lightgoldenrodyellow')
        time_slider = Slider(ax_slider, 'Time', 0, times[-1], valinit=0)

        def update_time(val):
            start = time_slider.val
            end = start + 10  # Display a 10-second window
            for ax in axs:
                ax.set_xlim(start, end)
            fig.canvas.draw_idle()

        time_slider.on_changed(update_time)

        initial_end = min(30, times[-1])  # Show 30 seconds
        for ax in axs:
            ax.set_xlim(0, initial_end)
    else:
        print(f"[INFO] Skipping slider for {n_channels} channels")

    # Save figure
    fig.savefig(save_path, dpi=300)

    # Show plot
    if show_plot:
        plt.show()

    plt.close(fig)
 


#For test this file alone, should be deleted

if __name__ == "__main__":
    scenario_list = ['scenario_1','scenario_2','scenario_3','scenario_4','scenario_5']
    db = myDatabase()
    for scenario_n in scenario_list:
        df = db.get_collection(scenario_n)
        df["_id"] = df["_id"].astype(str)
        for sc_scenario in df['_id'][0:2]:
            data = DataPhy(sc_scenario,save_path=Path("./tempPhy") / str(scenario_n) /str(sc_scenario))
            data.show()
            data.process()
    db.close()


