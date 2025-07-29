import numpy as np
from wfdb import processing
import neurokit2 as nk
import pandas as pd
from tqdm import tqdm 
import custom_helper_code
from scipy.signal import resample


UNIFIED_FREQUENCY = 500
WINDOW_TIME = 10  # second, 10 seconds bc all ptb records are 10 seconds long
WINDOW_SIZE =  WINDOW_TIME * UNIFIED_FREQUENCY  
# Positive class weight for loss function, precalculated from training data
# This val was too high, so we divided it by 4 to balance the loss function
POS_WEIGHT = 43.711233 / 4 
# If True, use a single window when training/inference
# in the future, we might want to take multiple random windows from the same signal to augument
# the data set 
USE_ONE_WINDOW = True  
REFERENCE_POLARITY_LEAD_IDX = 1  # Lead II is the reference lead for polarity check
MIN_SIGNAL_DURATION = 2  # seconds, minimum duration of the signal to even be used for training
np.random.seed(42)  # For reproducibility

MEAN_AGE_TRAIN = 53.78590876196458
STD_AGE_TRAIN = 20.76942829983386

from biosppy.signals import ecg as biosppy_ecg
from biosppy.signals import hrv as biosppy_hrv

def preprocess_signal(record_path, windowing_method='entire_recording', is_training=False):
    """ Preprocess the ECG signal from the record file.
    Args:
        record_path (str): Path to the record file.
        windowing_method (str): Method for windowing the signal.
        is_training (bool): Whether the preprocessing is for training or inference. This is useful when 
        you want to disable checking for the minimum signal duration, as usually short records will be skipped, 
        and REQUIRED so that we don't get labels from held-out data
    Returns:
        signal (np.ndarray): Preprocessed signal of shape (num_leads, num_samples).
        wide_feats (np.ndarray): Wide features
    """
    signal, metadata = custom_helper_code.load_signals(record_path)
    header_text = custom_helper_code.load_header(record_path)
    signal = signal.T  # (num_leads, num_samples)
    orig_freq = metadata['fs']

    if is_training:
        if signal.shape[1] < orig_freq * MIN_SIGNAL_DURATION:
            return None, None

    age, sex, label = None, None, None
    if is_training:
        age, sex, label = custom_helper_code.get_patient_info(header_text, get_label=True)
    else:
        age, sex = custom_helper_code.get_patient_info(header_text, get_label=False)
    if age is not None and not np.isnan(age):
        age = (age - MEAN_AGE_TRAIN) / STD_AGE_TRAIN
    normalized_age = 0.0 if age is None or np.isnan(age) else age
    if sex is None or sex.lower() not in ['male','female']:
        numerical_sex = 0.5
    elif sex.lower() == 'male':
        numerical_sex = 0.0
    else:
        numerical_sex = 1.0

    # Resample
    if orig_freq != UNIFIED_FREQUENCY:
        ns = int(signal.shape[1] * (UNIFIED_FREQUENCY / orig_freq))
        signal = resample(signal, ns, axis=1)

    cleaned = [nk.ecg_clean(lead, sampling_rate=UNIFIED_FREQUENCY) for lead in signal]
    signal = np.stack(cleaned)
    # try:
    #     signal, _ = correct_12_lead_polarity_lead_II_ref(signal,UNIFIED_FREQUENCY)
    # except:
    #     print("Error in correcting polarity of the signal")

    # comp = calculate_ecg_features(signal, UNIFIED_FREQUENCY, REFERENCE_POLARITY_LEAD_IDX)
    # if comp is None:
    #     return None, None
    # comp = np.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0)
    wide_feats = np.array([normalized_age, numerical_sex])

    windows = get_windows(signal, method=windowing_method, window_size=WINDOW_SIZE)
    if windows is None or len(windows) == 0:
        print(f"No windows found for record {record_path}.")
        return None, None
    signal = windows[0]
    signal = normalize(signal, smooth=1e-8)

    return signal, wide_feats

def check_interval(waves_signals: dict) -> pd.DataFrame:
    # Define the relevant keys (also in the order they should be)
    keys = ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
    # Create a dataframe with the data from the dictionary only for the relevant keys
    df_waves = pd.DataFrame({key: waves_signals[key] for key in keys})
    rows = df_waves.shape[0]
    mask = np.zeros(rows)
    for i in range(1, rows):
        start_interval = df_waves.loc[i-1, 'ECG_R_Peaks']
        end_interval = df_waves.loc[i, 'ECG_R_Peaks']
        sliced_df = df_waves[df_waves.isin([start_interval, end_interval]).any(axis=1)]
        # if the sliced df has no missing values, then the interval is correct and the mask should be 1
        first_condition = sliced_df[keys].isnull().sum().sum() == 0
        second_condition = keys == sliced_df.iloc[0].sort_values(ascending=True).index.tolist()
        if first_condition and second_condition:
            mask[i] = 1
    return df_waves[mask == 1]

def ecg_signal_features(row: pd.Series, frequency: int = 400, milliseconds: bool = True) -> dict:
    ECG_P_Peaks, ECG_P_Onsets, ECG_P_Offsets, ECG_Q_Peaks, ECG_S_Peaks, ECG_T_Peaks, ECG_T_Onsets, ECG_T_Offsets, ECG_R_Peaks = row
    # Compute the P wave duration
    P_wave_duration = (ECG_P_Offsets - ECG_P_Onsets)
    # Compute the PR interval
    PR_interval = (ECG_Q_Peaks - ECG_P_Onsets)
    # Compute the PR segment
    PR_segment = (ECG_Q_Peaks - ECG_P_Offsets)
    # Compute the QRS duration
    QRS_duration = (ECG_S_Peaks - ECG_Q_Peaks)
    # Compute the QT interval
    QT_interval = (ECG_T_Offsets - ECG_Q_Peaks)
    # Compute the ST segment
    ST_segment = (ECG_T_Onsets - ECG_S_Peaks)
    
    if milliseconds:
        P_wave_duration = P_wave_duration * 1000 / frequency
        PR_interval = PR_interval * 1000 / frequency
        PR_segment = PR_segment * 1000 / frequency
        QRS_duration = QRS_duration * 1000 / frequency
        QT_interval = QT_interval * 1000 / frequency
        ST_segment = ST_segment * 1000 / frequency

    else:
        P_wave_duration = P_wave_duration / frequency
        PR_interval = PR_interval / frequency
        PR_segment = PR_segment / frequency
        QRS_duration = QRS_duration / frequency
        QT_interval = QT_interval / frequency
        ST_segment = ST_segment / frequency

    return {
        'P_wave_duration': P_wave_duration,
        'PR_interval': PR_interval,
        'PR_segment': PR_segment,
        'QRS_duration': QRS_duration,
        'QT_interval': QT_interval,
        'ST_segment': ST_segment
    }

def st_slope(signal_df: pd.DataFrame, s_peak: int, t_onset: int) -> float:
    with np.errstate(divide='ignore', invalid='ignore'):
        return (signal_df.iloc[t_onset]['ECG_Clean'] - signal_df.iloc[s_peak]['ECG_Clean']) / (t_onset - s_peak)


def calculate_ecg_features(signals, frequency, channel=1):
    # Extract the record file
    # Load the header and signals
    # Fill nan values with the mean of the signal
    signals = np.nan_to_num(signals, nan=np.nanmean(signals))
    if signals.shape[0] == 12:
        signals = signals.T  # Ensure the shape is (num_samples, num_leads) for NeuroKit2 compatibility
    
    try:
        ecg_signals, info = nk.ecg_process(signals[:, channel], sampling_rate=frequency)

        # Get HRV features - but handle potential issues
        try:
            hrv_features = nk.hrv_time(ecg_signals, sampling_rate=frequency)
        except Exception as e:
            print(f"Warning: HRV calculation failed: {e}")
            hrv_features = pd.DataFrame() # Empty DataFrame as fallback

        # Check the intervals (enforce the correct order)
        correct_waves = check_interval(info)

        # If no correct waves are found, return None
        if correct_waves.shape[0] == 0:
            return None

        # Compute the ECG features
        correct_waves[["P_wave_duration","PR_interval","PR_segment","QRS_duration","QT_interval","ST_segment"]] = correct_waves.apply(
            lambda x: ecg_signal_features(x, frequency), axis=1, result_type='expand')
        
        # Compute the ST slope
        correct_waves['ST_slope'] = correct_waves.apply(
            lambda x: st_slope(ecg_signals, int(x['ECG_S_Peaks']), int(x['ECG_T_Onsets'])), axis=1)
        
        # Drop the unnecessary columns
        correct_waves.drop(columns=['ECG_P_Peaks','ECG_P_Onsets','ECG_P_Offsets','ECG_Q_Peaks',
                                   'ECG_S_Peaks','ECG_T_Peaks','ECG_T_Onsets','ECG_T_Offsets','ECG_R_Peaks'], inplace=True)
        
        # Add the exam_id
        correct_waves['exam_id'] = 'placeholder'
        
        # Aggregate the features
        correct_waves = correct_waves.groupby('exam_id').aggregate(['mean', 'std', 'min', 'max']).reset_index()
        correct_waves.columns = ['_'.join(col).strip() for col in correct_waves.columns.values]
        correct_waves.drop(columns=['exam_id_'], inplace=True)

        # Define the desired features but only use those that are actually available
        desired_features = ['ST_slope_min', 'ST_slope_mean', 'ST_segment_max', 'ST_segment_mean', 
                          'QRS_duration_min', 'P_wave_duration_max', 'ST_segment_min', 
                          'PR_segment_mean', 'QRS_duration_mean', 'ST_segment_std', 
                          'PR_segment_min', 'P_wave_duration_mean']
                          
        hrv_desired_features = ['HRV_HTI', 'HRV_CVNN', 'HRV_TINN', 'HRV_MCVNN', 'HRV_SDSD', 'HRV_MaxNN']
        
        # Only include columns that exist in the DataFrame
        available_features = [col for col in desired_features if col in correct_waves.columns]
        available_hrv = [col for col in hrv_desired_features if col in hrv_features.columns]
        
        # Get only the available features
        correct_waves_subset = correct_waves[available_features] if available_features else pd.DataFrame()
        hrv_subset = hrv_features[available_hrv] if available_hrv else pd.DataFrame()
        
        # Combine available features
        final_df = pd.concat([correct_waves_subset, hrv_subset], axis=1)
        
        # Handle case where we have no features
        if final_df.empty:
            print("Warning: No features could be extracted")
            return None
            
        # Convert to numpy and return
        return final_df.to_numpy().flatten()  # Flatten to ensure 1D array
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None  # Return a default value

def correct_12_lead_polarity_lead_II_ref(multilead_signal, fs=UNIFIED_FREQUENCY, allow_padding=True):
    """
    Checks polarity of a 12-lead ECG using Lead II as reference and corrects all leads.
    Assumes multilead_signal is (num_leads, num_samples).

    allow_padding (bool): If True, allows padding of the signal to ensure it has enough samples.
    If False, raises an error if the signal is shorter than the required length.

    Returns (corrected_signal, was_inverted):
        corrected_signal (np.ndarray): The multilead signal with corrected polarity.
        was_inverted (bool): True if the signal was inverted, False otherwise.
    """
    # Ensure signal is 2D
    if not isinstance(multilead_signal, np.ndarray) or multilead_signal.ndim != 2:
        raise ValueError("multilead_signal must be a 2D numpy array (num_leads, num_samples).")

    # print(multilead_signal.shape, "Shape of multilead signal")
    # Extract the reference lead data
    if multilead_signal.shape[0] <= REFERENCE_POLARITY_LEAD_IDX:
        raise ValueError(f"Lead II (index {REFERENCE_POLARITY_LEAD_IDX}) is out of bounds "
                         f"for signal with {REFERENCE_POLARITY_LEAD_IDX.shape[0]} leads. Check your lead order.")
    
    reference_lead_data = multilead_signal[REFERENCE_POLARITY_LEAD_IDX, :]
    was_inverted = False
    try:
        _, was_inverted = nk.ecg_invert(reference_lead_data, sampling_rate=fs, show=False)
    except Exception as e:
        print(f"Error during polarity check: {e}")
        print(f"signal shape {multilead_signal.shape}, reference lead index {REFERENCE_POLARITY_LEAD_IDX}")
        raise e
    if was_inverted:
        # print(f"DEBUG: Lead II detected as inverted. Flipping all {multilead_signal.shape[0]} leads.")
        return -1 * multilead_signal, True
    else:
        # print(f"DEBUG: Lead II detected as normal polarity. No flip needed.")
        return multilead_signal, False
def detect_qrs_peaks(signal, fs, lead_index=1):
    """
    Detects QRS complex peaks (R-peaks) using the wfdb library's XQRS detector.

    This function uses the well-established `xqrs_detect` algorithm from wfdb,
    which is a robust method for finding R-peaks in an ECG signal.

    Args:
        signal (np.ndarray):
            Input signal of shape (num_leads, num_samples).
        fs (int):
            The sampling frequency of the signal in Hz.
        lead_index (int, optional):
            The index of the lead to use for QRS detection. Defaults to 0.

    Returns:
        np.ndarray: An array of indices corresponding to the detected R-peaks.
    """
    if signal.ndim == 1:
        # Handle single-lead signal passed as a 1D array
        ecg_lead = signal
    else:
        # Select the specified lead from the multi-lead signal
        ecg_lead = signal[lead_index, :]

    # --- Use wfdb's XQRS peak detector ---
    # The `xqrs_detect` function is a reliable, standard algorithm.
    # USE LEAD II
    _, rpeaks_info = nk.ecg_peaks(ecg_lead, sampling_rate=fs, method="neurokit")

    # The R-peak locations are in the 'ECG_R_Peaks' key of the info dictionary
    r_peaks = rpeaks_info["ECG_R_Peaks"]

    return r_peaks

def get_windows(signal, window_size, method='random', num_windows=1, stride=None, qrs_offset=None):
    """
    Extracts windows from a signal using one of several methods.

    Includes a 'qrs' method to extract windows aligned with QRS complexes and
    an 'entire_recording' method to handle the whole signal as one window.

    Args:
        signal (np.ndarray):
            Input signal of shape (num_leads, num_samples).
        window_size (int):
            The size of each window in samples.
        method (str, optional):
            The method for windowing. Defaults to 'random'.
            - 'random': Extracts N random windows. Ideal for training augmentation.
            - 'tiled': Extracts sequential, possibly overlapping windows. Ideal for inference.
            - 'qrs': Extracts `num_windows` windows, each centered on a randomly 
                     selected QRS peak. Requires `qrs_peaks`. If cannot find QRS peaks, uses random instead
            - 'entire_recording': Returns the entire signal as a single window,
                                  either truncated or padded to `window_size`.
        num_windows (int, optional):
            Number of windows for method='random' or method='qrs'. Defaults to 1.
        stride (int, optional):
            Step size for method='tiled'. Defaults to `window_size`.
        qrs_offset (int, optional):
            Offset from the R-peak to the start of the window, in samples.
            Defaults to `window_size // 2` (center-aligned).
            - `qrs_offset = 0`: Window starts at the R-peak.
            - `qrs_offset = window_size // 2`: Window is centered on the R-peak.

    Returns:
        np.ndarray: A batch of windows of shape (num_windows, num_leads, window_size).
    """
    num_leads, total_samples = signal.shape
    if qrs_offset is None:
        qrs_offset = window_size // 2

    if method == 'random':
        # Handle cases where the signal is shorter than the window size
        if total_samples < window_size:
            padding_amount = window_size - total_samples
            padded_signal = np.pad(signal, ((0, 0), (0, padding_amount)), 'constant')
            # Return multiple copies if requested, otherwise just one
            return np.repeat(np.expand_dims(padded_signal, axis=0), num_windows, axis=0)

        windows = []
        # Ensure there's a valid range for random sampling
        start_indices = np.random.randint(0, total_samples - window_size + 1, size=num_windows)
            
        for start_index in start_indices:
            window = signal[:, start_index : start_index + window_size]
            windows.append(window)
        return np.stack(windows)

    elif method == 'tiled':
        if stride is None:
            stride = window_size

        windows = []
        # Iterate through the signal with the specified stride
        for start in range(0, total_samples, stride):
            end = start + window_size
            if end > total_samples:
                # Pad the last window if it extends beyond the signal
                padding_amount = end - total_samples
                window = np.pad(signal[:, start:], ((0, 0), (0, padding_amount)), 'constant')
            else:
                window = signal[:, start:end]
            windows.append(window)
        
        # Handle case where signal is too short to produce any windows
        if not windows:
            padding_amount = max(0, window_size - total_samples)
            padded_signal = np.pad(signal, ((0,0), (0, padding_amount)), 'constant')
            return np.expand_dims(padded_signal, axis=0)
            
        return np.stack(windows)

    elif method == 'qrs':
        # This is a placeholder for your actual QRS detection logic
        # For example: qrs_peaks = detect_qrs_peaks(signal, fs=UNIFIED_FREQUENCY)
        # To make this runnable, we'll generate some dummy peaks
        qrs_peaks = np.linspace(qrs_offset, total_samples - (window_size - qrs_offset), 10, dtype=int)
        
        if qrs_peaks is None or len(qrs_peaks) == 0:
            # If no peaks, fall back to the 'random' method as a robust alternative
            print("Warning: No QRS peaks found. Falling back to 'random' windowing.")
            return get_windows(signal, window_size, 'random', num_windows)

        windows = []
        # Randomly select 'num_windows' peaks from the provided list, with replacement.
        selected_peaks = np.random.choice(qrs_peaks, size=num_windows, replace=True)

        for peak_index in selected_peaks:
            start_index = peak_index - qrs_offset
            end_index = start_index + window_size
            
            # Create an empty, zero-padded window
            window = np.zeros((num_leads, window_size), dtype=signal.dtype)

            # Determine the slice of the original signal to copy
            src_start = max(0, start_index)
            src_end = min(total_samples, end_index)
            
            # Determine where to place the slice in the new window
            dest_start = max(0, -start_index)
            dest_end = dest_start + (src_end - src_start)

            # Copy the data if there is a valid slice to copy
            if src_end > src_start and dest_end > dest_start:
                window[:, dest_start:dest_end] = signal[:, src_start:src_end]
            
            windows.append(window)
        
        if not windows:
            return np.empty((0, num_leads, window_size), dtype=signal.dtype)

        return np.stack(windows)
        
    elif method == 'entire_recording':
        if total_samples > window_size:
            # Truncate the signal if it's longer
            window = signal[:, :window_size]
        else:
            # Pad the signal if it's shorter
            padding_amount = window_size - total_samples
            window = np.pad(signal, ((0, 0), (0, padding_amount)), 'constant')
        
        # Add a batch dimension to match the standard output shape
        return np.expand_dims(window, axis=0)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'random', 'tiled', 'qrs', or 'entire_recording'.")



def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

def compute_challenge_score(labels, outputs, fraction_capacity = 0.05, num_permutations = 10**4, seed=12345):
    '''Compute the physionet 2025 challenge score based on the provided labels and outputs.
    NOTE: This function is modified to work with numpy 1.26 (as the original used a new np.argsort, specifically
    with the standalone stable keyword argument that wasn't present for versions before np 2.0))
    '''
    # Check the data.
    assert len(labels) == len(outputs)
    num_instances = len(labels)
    capacity = int(fraction_capacity * num_instances)

    # Convert the data to NumPy arrays, as needed, for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    # Permute the labels and outputs so that we can approximate the expected confusion matrix for "tied" probabilities.
    tp = np.zeros(num_permutations)
    fp = np.zeros(num_permutations)
    fn = np.zeros(num_permutations)
    tn = np.zeros(num_permutations)

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_permutations):
        permuted_idx = np.random.permutation(np.arange(num_instances))
        permuted_labels = labels[permuted_idx]
        permuted_outputs = outputs[permuted_idx]

        ordered_idx = np.argsort(permuted_outputs, kind='stable')[::-1]
        ordered_labels = permuted_labels[ordered_idx]

        tp[i] = np.sum(ordered_labels[:capacity] == 1)
        fp[i] = np.sum(ordered_labels[:capacity] == 0)
        fn[i] = np.sum(ordered_labels[capacity:] == 1)
        tn[i] = np.sum(ordered_labels[capacity:] == 0)

    tp = np.mean(tp)
    fp = np.mean(fp)
    fn = np.mean(fn)
    tn = np.mean(tn)

    # Compute the true positive rate.
    if tp + fn > 0:
        tpr = tp / (tp + fn)
    else:
        tpr = float('nan')

    return tpr