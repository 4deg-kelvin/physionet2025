# ECG Data Augmentation Integration Guide

This document explains the implementation and integration of ECG data augmentation techniques for improving model robustness in medical AI applications.

## Overview

The augmentation pipeline has been implemented in a modular, clean way that integrates seamlessly with the existing PyTorch Lightning framework. The augmentations are applied only during training to improve model robustness against out-of-distribution shifts, particularly powerline interference and temporal variations.

## Files Added/Modified

### New Files
1. **`augmentations.py`** - Main augmentation pipeline implementation
2. **`test_augmentations.py`** - Comprehensive unit tests (requires pytest)
3. **`test_simple.py`** - Simple tests without external dependencies
4. **`test_dataloader_integration.py`** - Integration tests with mock dataloader pattern

### Modified Files
1. **`dataloader.py`** - Added augmentation support to `ECGDataset` and `ECGDataModule`
2. **`fm.py`** - Added augmentation configuration and integration

## Architecture

### Core Components

#### 1. ECGAugmentations Class (`augmentations.py`)
- **Input**: ECG tensors of shape `[12, seq_len]` (12 leads, variable length)
- **Output**: Augmented tensors of shape `[12, 5000]` (fixed 10-second duration at 500 Hz)
- **Features**:
  - Powerline interference simulation (50/60 Hz + harmonics)
  - Random cropping and temporal shifting
  - Amplitude clamping to valid ECG ranges (-5 to 5 mV)
  - Comprehensive input validation and error handling

#### 2. Integration Pattern
```python
# In ECGDataset.__init__
self.augmenter = None
if self.is_training and aug_config:
    from augmentations import ECGAugmentations
    self.augmenter = ECGAugmentations(config=aug_config)

# In ECGDataset.__getitem__
if self.augmenter:
    signal = self.augmenter(signal)
```

### Augmentation Techniques

#### 1. Powerline Interference
- **Purpose**: Simulate realistic powerline noise to improve robustness
- **Implementation**:
  - Randomly selects 50 Hz or 60 Hz base frequency
  - Adds ±1 Hz frequency variation
  - Includes 2nd and 3rd harmonics with appropriate amplitude ratios
  - SNR range: 10-30 dB
  - Applied identically across all 12 leads
  - Probability: 50% (configurable)

#### 2. Temporal Augmentation
- **Purpose**: Handle temporal variations and ensure fixed output length
- **Implementation**:
  - Random cropping to 80-100% of original length
  - Temporal shifting (±10% of signal length) with zero padding
  - Final padding/truncation to exactly 5000 samples
  - Applied to all leads consistently
  - Probability: 100% during training (ensures fixed length)

## Configuration

### Augmentation Configuration Structure
```python
augmentations_config = {
    'powerline': {
        'prob': 0.5,                    # 50% probability
        'frequencies': [50, 60],        # Base frequencies (Hz)
        'freq_std': 1.0,               # Frequency variation (Hz)
        'snr_range': [10, 30],         # SNR range (dB)
        'harmonics': True              # Include 2nd/3rd harmonics
    },
    'temporal': {
        'prob': 1.0,                   # Always apply (for length consistency)
        'crop_range': [0.8, 1.0],      # Crop to 80-100% of original
        'shift_range': 0.1             # Shift up to 10% of length
    },
    'general': {
        'sample_rate': 500,            # ECG sampling rate
        'target_length': 5000,         # Output length (10 seconds)
        'amplitude_range': [-5.0, 5.0], # Valid ECG range (mV)
        'verbose': True                # Enable logging
    }
}
```

## Usage Examples

### Basic Usage
```python
from augmentations import ECGAugmentations, create_default_augmentation_config

# Create augmentation pipeline
config = create_default_augmentation_config()
augmenter = ECGAugmentations(config)

# Apply to ECG signal
signal = torch.randn(12, 4800)  # Variable length input
augmented = augmenter(signal)   # Fixed length output [12, 5000]
```

### Integration with Training
```python
# In fm.py main training block
augmentations_config = {
    'powerline': {'prob': 0.5, 'frequencies': [50, 60], ...},
    'temporal': {'prob': 1.0, 'crop_range': [0.8, 1.0], ...},
    'general': {'sample_rate': 500, 'target_length': 5000, ...}
}

# Pass config to data module
data_module = ECGDataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LENGTH,
    windowing_method='entire_recording',
    aug_config=augmentations_config
)

# Training dataset gets augmentation, validation/test do not
data_module.train_dataset = ECGDataset(
    train_records, DATA_DIR, is_training=True, 
    aug_config=augmentations_config
)
data_module.val_dataset = ECGDataset(
    val_records, DATA_DIR, is_training=False
    # No aug_config = no augmentation
)
```

## Key Design Decisions

### 1. Modular Architecture
- **Rationale**: Keep augmentation logic separate from data loading and model code
- **Benefits**: Easy to maintain, test, and modify independently

### 2. Fixed Output Length
- **Rationale**: Ensure consistent tensor shapes for batch processing
- **Implementation**: Temporal augmentation always produces exactly 5000 samples
- **Benefits**: Eliminates need for complex padding logic in training loop

### 3. Training-Only Application
- **Rationale**: Augmentation should only improve training robustness
- **Implementation**: Applied only when `is_training=True` and `aug_config` is provided
- **Benefits**: Clean separation between training and inference behavior

### 4. Comprehensive Validation
- **Rationale**: Medical AI requires robust error handling
- **Implementation**: Input validation, output verification, amplitude clamping
- **Benefits**: Prevents training crashes and ensures data quality

## Testing

### Unit Tests
```bash
# Simple tests (no external dependencies)
python test_simple.py

# Integration tests
python test_dataloader_integration.py

# Full unit test suite (requires pytest)
python -m pytest test_augmentations.py -v
```

### Test Coverage
- ✅ Basic functionality and output shape consistency
- ✅ Powerline interference detection (FFT analysis)
- ✅ Temporal augmentation and padding verification
- ✅ Input validation and error handling
- ✅ Edge cases (short/long signals, extreme values)
- ✅ Configuration options and defaults
- ✅ Integration with dataloader pattern
- ✅ Batch processing compatibility

## Performance Considerations

### Memory Efficiency
- Uses in-place operations where possible
- Minimal memory overhead during augmentation
- Vectorized operations (no loops over time dimension)

### Computational Efficiency
- Leverages PyTorch's optimized tensor operations
- Minimal CPU overhead for powerline noise generation
- Efficient FFT-based frequency domain operations

### GPU Compatibility
- Maintains device consistency (CPU/CUDA)
- All operations support automatic differentiation
- Compatible with mixed-precision training

## Troubleshooting

### Common Issues

1. **Shape Mismatch Errors**
   - Ensure input has exactly 12 leads
   - Check that input is 2D tensor `[leads, samples]`

2. **Memory Issues**
   - Large batch sizes may require memory optimization
   - Consider reducing `batch_size` if GPU memory is limited

3. **Augmentation Not Applied**
   - Verify `is_training=True` and `aug_config` is provided
   - Check configuration probabilities (should be > 0)

4. **Training Instability**
   - Reduce augmentation probability if training is unstable
   - Adjust SNR range or amplitude clamping if needed

### Debug Mode
```python
config = create_default_augmentation_config()
config['general']['verbose'] = True  # Enable detailed logging
```

## Future Enhancements

Potential improvements that could be added:

1. **Additional Augmentations**
   - Gaussian noise addition
   - Baseline wander simulation
   - Lead dropout/masking

2. **Advanced Temporal Techniques**
   - Time warping
   - Elastic deformation
   - Multi-scale temporal augmentation

3. **Adaptive Augmentation**
   - Curriculum learning integration
   - Difficulty-based augmentation scheduling
   - Performance-driven augmentation tuning

4. **Domain-Specific Techniques**
   - Arrhythmia-preserving augmentations
   - Population-specific noise models
   - Multi-lead correlation preservation

## References

- Medical device standards for ECG signal quality
- Powerline interference characteristics in medical equipment
- Data augmentation best practices for time series classification
- PyTorch Lightning framework documentation
