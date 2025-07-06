# Combined EDA Scripts

This directory contains two equivalent scripts for performing Exploratory Data Analysis (EDA) on the project's ECG data:
1.  `combined_eda.ipynb`: A Jupyter Notebook for interactive analysis.
2.  `combined_eda.py`: A Python script for automated execution.

Both scripts perform the same tasks:
- Load ECG records from the `../training_data/samitrop/` directory.
- Extract a comprehensive set of features, including basic demographics, advanced ECG morphological features, Heart Rate Variability (HRV) metrics, and wavelet-based features.
- Perform EDA by generating and saving several plots.
- Save the final, feature-rich dataset to a CSV file.

## How to Use

### Prerequisites

Ensure you have the necessary Python libraries installed. You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy wfdb neurokit2 joblib tqdm orjson pywavelets jupyter
```

### 1. Jupyter Notebook (`combined_eda.ipynb`)

This is the recommended approach for interactive exploration and visualization.

1.  **Start Jupyter:**
    Navigate to the root directory of this project in your terminal and run:
    ```bash
    jupyter notebook
    ```
2.  **Open the Notebook:**
    In the Jupyter interface that opens in your browser, navigate to the `EDA` folder and click on `combined_eda.ipynb`.
3.  **Run the Cells:**
    You can run the cells one by one to see the output of each step, or you can run all cells at once by selecting "Cell" > "Run All" from the menu.

### 2. Python Script (`combined_eda.py`)

This is ideal for automated runs or if you prefer a non-interactive script.

1.  **Navigate to the Directory:**
    Open your terminal and change the directory to the `EDA` folder:
    ```bash
    cd EDA
    ```
2.  **Run the Script:**
    Execute the script from the terminal:
    ```bash
    python combined_eda.py
    ```
    The script will print its progress and notify you upon completion.

## Outputs

Both the notebook and the script will generate the following files inside the `EDA` directory:

-   `combined_features.csv`: A CSV file containing the full dataset with all the extracted features.
-   `age_distribution.png`: A histogram showing the distribution of patient ages, grouped by gender.
-   `correlation_heatmap.png`: A heatmap visualizing the correlation between all numerical features.
-   `feature_boxplots.png`: Boxplots comparing key features between patients with and without Chagas disease.
-   `rr_interval_length_distribution.png`: A histogram showing the distribution of the number of detected RR-intervals across all signals.
