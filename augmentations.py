"""
ECG Data Augmentation Pipeline for Medical AI

This module provides comprehensive data augmentation techniques for ECG signals
to improve model robustness against out-of-distribution shifts, particularly
powerline interference and temporal variations.

Author: Andy Smithwick (Copilot Assisted)
Date: September 2025
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGAugmentations:
    """
    Comprehensive ECG signal augmentation pipeline.
    
    Applies various augmentation techniques to 12-lead ECG signals during training
    to improve model robustness and generalization.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing augmentation parameters
        
    Expected config structure:
    {
        'powerline': {
            'prob': 0.5,
            'frequencies': [50, 60],
            'freq_std': 1.0,
            'snr_range': [10, 30],
            'harmonics': True
        },
        'temporal': {
            'prob': 1.0,
            'crop_range': [0.8, 1.0],
            'shift_range': 0.1
        },
        'general': {
            'sample_rate': 500,
            'target_length': 5000,
            'amplitude_range': [-5.0, 5.0],
            'verbose': False
        }
    }
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get('general', {}).get('sample_rate', 500)
        self.target_length = config.get('general', {}).get('target_length', 5000)
        self.amplitude_range = config.get('general', {}).get('amplitude_range', [-5.0, 5.0])
        self.verbose = config.get('general', {}).get('verbose', False)
        
        # Powerline interference config
        self.powerline_config = config.get('powerline', {})
        self.powerline_prob = self.powerline_config.get('prob', 0.5)
        self.powerline_freqs = self.powerline_config.get('frequencies', [50, 60])
        self.freq_std = self.powerline_config.get('freq_std', 1.0)
        self.snr_range = self.powerline_config.get('snr_range', [10, 30])
        self.use_harmonics = self.powerline_config.get('harmonics', True)
        
        # Temporal augmentation config
        self.temporal_config = config.get('temporal', {})
        self.temporal_prob = self.temporal_config.get('prob', 1.0)
        self.crop_range = self.temporal_config.get('crop_range', [0.8, 1.0])
        self.shift_range = self.temporal_config.get('shift_range', 0.1)
        
        if self.verbose:
            logger.info(f"ECGAugmentations initialized with config: {config}")
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline to ECG signal.
        
        Args:
            signal (torch.Tensor): Input ECG signal of shape [12, seq_len]
            
        Returns:
            torch.Tensor: Augmented ECG signal of shape [12, 5000]
            
        Raises:
            ValueError: If input signal has invalid shape or contains invalid values
        """
        # Input validation
        self._validate_input(signal)
        
        # Convert to float32 for processing
        signal = signal.float()
        
        if self.verbose:
            logger.info(f"Input signal shape: {signal.shape}")
        
        # Apply powerline interference augmentation
        if torch.rand(1).item() < self.powerline_prob:
            signal = self._add_powerline_interference(signal)
            if self.verbose:
                logger.info("Applied powerline interference augmentation")
        
        # Apply temporal augmentations (crop and shift) - MUST BE LAST
        if torch.rand(1).item() < self.temporal_prob:
            signal = self._random_crop_and_shift(signal)
            if self.verbose:
                logger.info("Applied temporal augmentation")
        else:
            # Even if no temporal augmentation, ensure fixed length
            signal = self._ensure_fixed_length(signal)
        
        # Final validation and clamping
        signal = self._final_validation_and_clamp(signal)
        
        return signal
    
    def _validate_input(self, signal: torch.Tensor) -> None:
        """Validate input signal format and values."""
        if not isinstance(signal, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(signal)}")
        
        if signal.dim() != 2:
            raise ValueError(f"Expected 2D tensor [leads, samples], got shape {signal.shape}")
        
        if signal.shape[0] != 12:
            raise ValueError(f"Expected 12 leads, got {signal.shape[0]}")
        
        if signal.shape[1] == 0:
            raise ValueError("Signal cannot have zero length")
        
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            raise ValueError("Input signal contains NaN or infinity values")
    
    def _add_powerline_interference(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Add synthetic powerline interference to ECG signal.
        
        Adds noise at 50/60 Hz ±1 Hz with harmonics, applied identically across all leads.
        
        Args:
            signal (torch.Tensor): Input ECG signal [12, seq_len]
            
        Returns:
            torch.Tensor: Signal with added powerline interference
        """
        device = signal.device
        dtype = signal.dtype
        seq_len = signal.shape[1]
        
        # Choose random powerline frequency
        base_freq = np.random.choice(self.powerline_freqs)
        freq_variation = np.random.normal(0, self.freq_std)
        actual_freq = base_freq + freq_variation
        
        # Random SNR for main frequency
        snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Calculate signal power (RMS)
        signal_power = torch.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_amplitude = torch.sqrt(noise_power)
        
        # Time vector
        t = torch.arange(seq_len, device=device, dtype=dtype) / self.sample_rate
        
        # Generate main frequency noise with random phase
        main_phase = np.random.uniform(0, 2 * np.pi)
        main_noise = noise_amplitude * torch.sin(2 * np.pi * actual_freq * t + main_phase)
        
        # Total noise starts with main frequency
        total_noise = main_noise.clone()
        
        # Add harmonics if enabled
        if self.use_harmonics:
            # Second harmonic (2x frequency, half amplitude)
            harmonic2_phase = np.random.uniform(0, 2 * np.pi)
            harmonic2_amp = noise_amplitude / 2
            harmonic2_noise = harmonic2_amp * torch.sin(2 * np.pi * (2 * actual_freq) * t + harmonic2_phase)
            total_noise += harmonic2_noise
            
            # Third harmonic (3x frequency, one-third amplitude)
            harmonic3_phase = np.random.uniform(0, 2 * np.pi)
            harmonic3_amp = noise_amplitude / 3
            harmonic3_noise = harmonic3_amp * torch.sin(2 * np.pi * (3 * actual_freq) * t + harmonic3_phase)
            total_noise += harmonic3_noise
        
        # Apply noise identically to all 12 leads
        noise_matrix = total_noise.unsqueeze(0).expand(12, -1)
        
        return signal + noise_matrix
    
    def _random_crop_and_shift(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply random cropping and temporal shifting, then ensure fixed output length.
        
        This method MUST be the final step in the augmentation pipeline as it
        guarantees the output has exactly 5000 samples.
        
        Args:
            signal (torch.Tensor): Input ECG signal [12, seq_len]
            
        Returns:
            torch.Tensor: Signal with shape [12, 5000] after crop/shift and padding
        """
        seq_len = signal.shape[1]
        
        # Step 1: Random cropping
        crop_factor = np.random.uniform(self.crop_range[0], self.crop_range[1])
        crop_length = int(seq_len * crop_factor)
        crop_length = max(1, crop_length)  # Ensure at least 1 sample
        
        # Random crop start position
        if seq_len > crop_length:
            start_idx = np.random.randint(0, seq_len - crop_length + 1)
            signal = signal[:, start_idx:start_idx + crop_length]
        
        # Step 2: Temporal shifting
        max_shift = int(self.shift_range * signal.shape[1])
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        
        if shift_amount != 0:
            signal = self._apply_temporal_shift(signal, shift_amount)
        
        # Step 3: Ensure exactly target_length samples (CRITICAL FINAL STEP)
        signal = self._ensure_fixed_length(signal)
        
        return signal
    
    def _apply_temporal_shift(self, signal: torch.Tensor, shift_amount: int) -> torch.Tensor:
        """Apply temporal shift with zero padding."""
        device = signal.device
        dtype = signal.dtype
        seq_len = signal.shape[1]
        
        if shift_amount > 0:
            # Positive shift: add zeros at the beginning
            padding = torch.zeros(12, shift_amount, device=device, dtype=dtype)
            signal = torch.cat([padding, signal], dim=1)
        elif shift_amount < 0:
            # Negative shift: add zeros at the end
            padding = torch.zeros(12, abs(shift_amount), device=device, dtype=dtype)
            signal = torch.cat([signal, padding], dim=1)
        
        return signal
    
    def _ensure_fixed_length(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Ensure signal has exactly target_length samples through truncation or padding.
        
        This is the FINAL operation that guarantees output shape consistency.
        """
        current_length = signal.shape[1]
        
        if current_length > self.target_length:
            # Truncate to target length
            signal = signal[:, :self.target_length]
        elif current_length < self.target_length:
            # Zero-pad to target length
            padding_needed = self.target_length - current_length
            padding = torch.zeros(12, padding_needed, device=signal.device, dtype=signal.dtype)
            signal = torch.cat([signal, padding], dim=1)
        
        return signal
    
    def _final_validation_and_clamp(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Final validation and clamping of the augmented signal.
        
        Ensures output meets all requirements before returning.
        """
        # Shape validation
        if signal.shape != (12, self.target_length):
            raise ValueError(
                f"Output shape mismatch: expected (12, {self.target_length}), "
                f"got {signal.shape}"
            )
        
        # Check for invalid values
        if torch.isnan(signal).any():
            warnings.warn("NaN values detected in augmented signal, replacing with zeros")
            signal = torch.nan_to_num(signal, nan=0.0)
        
        if torch.isinf(signal).any():
            warnings.warn("Infinite values detected in augmented signal, replacing with zeros")
            signal = torch.nan_to_num(signal, posinf=0.0, neginf=0.0)
        
        # Clamp to valid ECG range (FINAL STEP)
        signal = torch.clamp(signal, self.amplitude_range[0], self.amplitude_range[1])
        
        if self.verbose:
            logger.info(f"Final signal shape: {signal.shape}, range: [{signal.min():.3f}, {signal.max():.3f}]")
        
        return signal


def create_default_augmentation_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary for ECG augmentations.
    
    Returns:
        Dict[str, Any]: Default augmentation configuration
    """
    return {
        'powerline': {
            'prob': 0.5,
            'frequencies': [50, 60],
            'freq_std': 1.0,
            'snr_range': [10, 30],
            'harmonics': True
        },
        'temporal': {
            'prob': 1.0,
            'crop_range': [0.8, 1.0],
            'shift_range': 0.1
        },
        'general': {
            'sample_rate': 500,
            'target_length': 5000,
            'amplitude_range': [-5.0, 5.0],
            'verbose': False
        }
    }


# Example usage
if __name__ == "__main__":
    # Create example ECG signal
    torch.manual_seed(42)
    sample_signal = torch.randn(12, 4800)  # Slightly shorter than target
    
    # Create augmentation pipeline
    config = create_default_augmentation_config()
    config['general']['verbose'] = True
    augmenter = ECGAugmentations(config)
    
    # Apply augmentations
    print("Original signal shape:", sample_signal.shape)
    augmented_signal = augmenter(sample_signal)
    print("Augmented signal shape:", augmented_signal.shape)
    print("Signal range:", f"[{augmented_signal.min():.3f}, {augmented_signal.max():.3f}]")
