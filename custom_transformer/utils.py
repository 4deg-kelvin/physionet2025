import numpy as np


UNIFIED_FREQUENCY = 500
WINDOW_TIME = 15  # seconds
WINDOW_SIZE =  WINDOW_TIME * UNIFIED_FREQUENCY  # 15 seconds
LABEL_SMOOTHING = 0.1
POS_WEIGHT = 42 # Positive class weight for loss function

np.random.seed(42)  # For reproducibility


import numpy as np

def get_windows(signal, window_size=WINDOW_SIZE, method='random', num_windows=1, stride=None):
    """
    Extracts windows from a signal using one of several methods.

    Args:
        signal (np.ndarray):
            Input signal of shape (num_leads, num_samples).
        window_size (int):
            The size of each window in samples.
        method (str, optional):
            The method to use for windowing. Defaults to 'random'.
            - 'random': Extracts N random windows. Ideal for training augmentation.
                        Requires the `num_windows` argument.
            - 'tiled': Extracts sequential windows. Ideal for inference/validation.
                       Requires the `stride` argument.
        num_windows (int, optional):
            The number of random windows to extract when method='random'. Defaults to 1.
        stride (int, optional):
            The step size between consecutive windows when method='tiled'.
            If stride == window_size, windows are non-overlapping.
            If stride < window_size, windows are overlapping.
            Defaults to `window_size` if not provided.

    Returns:
        np.ndarray: A batch of windows of shape (num_windows, num_leads, window_size).
    """
    num_leads, total_samples = signal.shape

    # --- Method 1: Random, Augmented Windowing (for Training) ---
    if method == 'random':
        windows = []
        for _ in range(num_windows):
            # Pad the signal on the right to allow for any start point
            padding_amount = max(0, window_size - 1) # Ensure non-negative padding
            padded_signal = np.pad(signal, ((0, 0), (0, padding_amount)), mode='constant', constant_values=0)

            # Pick a random starting index from the *original* signal's range
            start_index = np.random.randint(0, total_samples) if total_samples > 0 else 0
            
            window = padded_signal[:, start_index : start_index + window_size]
            windows.append(window)
        return np.stack(windows)

    # --- Method 2: Tiled, Strided Windowing (for Inference/Validation) ---
    elif method == 'tiled':
        # Default stride is non-overlapping if not specified
        if stride is None:
            stride = window_size

        windows = []
        start = 0
        while start + window_size <= total_samples:
            window = signal[:, start : start + window_size]
            windows.append(window)
            start += stride
        
        # If no full windows could be extracted (signal is too short), pad it
        if not windows:
            padding_amount = max(0, window_size - total_samples)
            padded_signal = np.pad(signal, ((0,0), (0, padding_amount)), mode='constant', constant_values=0)
            return np.expand_dims(padded_signal, axis=0)
            
        return np.stack(windows)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'random' or 'tiled'.")

