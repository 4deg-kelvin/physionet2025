import numpy as np
from wfdb import processing
import neurokit2 as nk


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


import numpy as np

# Assume these constants and functions are defined elsewhere in your project
# For example:
# WINDOW_SIZE = 5000
# UNIFIED_FREQUENCY = 100
# def detect_qrs_peaks(signal, fs):
#     # A placeholder for your QRS detection logic
#     # In a real implementation, this would return the indices of R-peaks
#     num_samples = signal.shape[1]
#     return np.linspace(100, num_samples - 100, 10, dtype=int)

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