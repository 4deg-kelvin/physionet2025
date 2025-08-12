import time
from sklearn.model_selection import train_test_split
import numpy as np

# --- Standard Imports ---
import helper_code
import pandas as pd
from tqdm.contrib.concurrent import thread_map

def process_record(record):
    signal, _ = helper_code.load_signals(record) # shape: (num_samples, num_leads)
    assert signal.shape[1] == 12, f"Expected 12 leads, got {signal.shape[1]} for record {record}"
    # Calculate the mean and SD of the signal, per lead (there's 12)
    means = np.mean(signal, axis=0)
    stds = np.std(signal, axis=0)
    # For global stats: sum, sum of squares, count per lead
    sum_per_lead = np.sum(signal, axis=0)
    sumsq_per_lead = np.sum(signal**2, axis=0)
    count_per_lead = signal.shape[0]
    return means, stds, sum_per_lead, sumsq_per_lead, count_per_lead

DATA_DIR = "training_data"
ADDIT_DIRS = [
    r"/juice2/scr2/kelvinkn/other_work/edwards/prna_2020_pooled_inputs",
    r"/juice2/scr2/kelvinkn/other_work/edwards/physionet2021_data",
]
# --- Corrected Data Loading and Splitting ---
# 1. Load ALL records, for pretraining (since this is official code)
records_meta = helper_code.find_records_abs(DATA_DIR)

print(f"Found {len(records_meta)} records in {DATA_DIR}")

# Get additional records
addit_data = []
for dir in ADDIT_DIRS:
    addit_data += helper_code.find_records_abs(dir)


# 2. Combine all records into a single list
print("WARNING: EXCLUDING UNLABELED RECORDS, THIS IS FOR COMPETITION MODEL TRAINING")
# all_records = labeled_records + unlabeled_records
all_records = records_meta + addit_data
print(f"Total records found: {len(all_records)}")
if len(all_records) == 0:
    raise ValueError("No records found in the specified directories. Please check the paths.")

# 3. Split into training and validation sets
# Do train/val/test
train_records, temp_records = train_test_split(all_records, test_size=0.2, random_state=42, shuffle=True)
val_records, test_records = train_test_split(temp_records, test_size=0.5, random_state=42, shuffle=True)

# Aggregate per-lead sums and sum of squares using multithreading and tqdm
results = thread_map(process_record, all_records, max_workers=16, desc="Processing records")
_, _, all_sums, all_sumsqs, all_counts = zip(*results)

# Ensure all_sums and all_sumsqs are stacked as arrays of shape (num_records, num_leads)
all_sums = np.stack(all_sums)      # shape: (num_records, num_leads)
all_sumsqs = np.stack(all_sumsqs)  # shape: (num_records, num_leads)
all_counts = np.array(all_counts)  # shape: (num_records,)

# --- True global mean and std per lead ---
total_sum = np.sum(all_sums, axis=0)         # shape: (num_leads,)
total_sumsq = np.sum(all_sumsqs, axis=0)     # shape: (num_leads,)
total_count = np.sum(all_counts)             # scalar (all leads have same count per record)

global_mean_per_lead = total_sum / total_count
global_var_per_lead = (total_sumsq / total_count) - (global_mean_per_lead ** 2)
global_std_per_lead = np.sqrt(global_var_per_lead)

print("Global mean per lead:", global_mean_per_lead)
print("Global std per lead:", global_std_per_lead)

# Save each mean and sds per lead as ONE df, with rows as each lead, and columns as mean and std
df_stats = pd.DataFrame({
    'mean': global_mean_per_lead,
    'std': global_std_per_lead
}, index=[f'lead_{i+1}' for i in range(len(global_mean_per_lead))])



# Save the statistics to a CSV file
output_file = "ecg_statistics.csv"
df_stats.to_csv(output_file)
print(f"ECG statistics saved to {output_file}")




