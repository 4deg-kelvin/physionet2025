# Feature EDA Pipeline GPT

This pipeline performs robust, recursive ECG feature extraction and exploratory data analysis (EDA) for the PhysioNet 2025 dataset. It integrates advanced feature extraction logic and comprehensive EDA capabilities.

## Step-by-Step Breakdown

### 1. Setup and Imports
- Essential libraries for data handling, signal processing, parallelization, and visualization are imported.
- Helper functions are imported from `helper_code.py`.

### 2. Metadata Extraction
- Recursively finds all `.hea` files in the training data directory using `find_records`.
- For each record, extracts metadata:
  - **Age**: Parsed from the header file.
  - **Sex**: Encoded as binary (`is_male`).
  - **Chagas Label**: Disease status from the header.
  - **Relative Path**: Preserved for robust file access.
- Records with missing or invalid metadata are skipped.

### 3. Data Balancing
- Balances the dataset by sampling equal numbers of positive and negative Chagas cases.
- Builds the list of records to process from the balanced metadata, using the full relative path for each record.

### 4. Parallel Feature Extraction
- Features are extracted in parallel for each record using `joblib.Parallel`.
- The extraction function processes each record robustly, iterating through all 12 ECG channels to find a valid signal.

### 5. Feature Extraction Details
#### Morphological Features
- Extracted using `neurokit2` and custom logic:
  - **P wave duration**: Time between P onset and offset.
  - **PR interval**: Time from P onset to Q peak.
  - **PR segment**: Time from P offset to Q peak.
  - **QRS duration**: Time from Q peak to S peak.
  - **QT interval**: Time from Q peak to T offset.
  - **ST segment**: Time from S peak to T onset.
- For each, mean, std, min, and max are computed across detected beats.

#### ST Slope
- Calculated as the slope between S peak and T onset using the cleaned ECG signal.

#### Heart Rate Variability (HRV) Features
- Extracted with `neurokit2`:
  - **HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_SDSD, HRV_CVNN, HRV_CVSD, HRV_MedianNN, HRV_MadNN, HRV_MCVNN, HRV_IQRNN, HRV_SDRMSSD, HRV_Prc20NN, HRV_Prc80NN, HRV_pNN50, HRV_pNN20, HRV_MinNN, HRV_MaxNN, HRV_HTI, HRV_TINN**
- Only computed if enough R-peaks are detected and the signal is sufficiently variable.

#### Wavelet Features
- Extracted using `pywt` (Daubechies 4 wavelet, level 4):
  - **Energy and standard deviation** for detail coefficients (d1-d4) and approximation coefficient (a4).

### 6. Error Handling
- Robustly skips records with missing files or failed feature extraction, logging errors and continuing.

### 7. EDA and Visualization
- **Age Distribution**: Histogram by gender.
- **Correlation Heatmap**: All numeric features.
- **Boxplots**: Key features by Chagas status.

### 8. Output
- Final features and error records are saved as CSV files.
- EDA plots are saved as PNG files.

---

## Feature Extraction Summary

| Feature Type         | Features Extracted                                                                 |
|----------------------|-----------------------------------------------------------------------------------|
| Morphological        | P_wave_duration, PR_interval, PR_segment, QRS_duration, QT_interval, ST_segment   |
| ST Slope             | ST_slope                                                                          |
| HRV                  | HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_SDSD, HRV_CVNN, HRV_CVSD, HRV_MedianNN, etc.|
| Wavelet              | wavelet_energy_d1, wavelet_energy_d2, ..., wavelet_std_a4                         |
| Demographic/Clinical | age, is_male, chagas                                                              |

---

## How to Run

**Activate the Python virtual environment before running the pipeline or installing packages:**
```bash
source ../venv/bin/activate
```

Install required packages (after activating the environment):
```bash
pip install wfdb neurokit2 pywt scipy torch pandas numpy matplotlib seaborn scikit-learn joblib tqdm orjson
```

Run the pipeline:
```bash
python feature_eda_pipeline_GPT.py --train /Users/andysmithwick/Documents/GitHub/physionet2025/training_data --output ./EDA_output
```

---

## Notes

- The pipeline is robust to missing files and supports arbitrary folder depth.
- All features are extracted per record and aggregated for analysis.
- Outputs are saved in the specified output directory.
