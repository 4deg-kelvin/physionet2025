"""
Utility to convert a directory of WFDB (.dat, .hea) files to NumPy (.npy) files.
"""
import os
import numpy as np
import argparse
import sys
from tqdm import tqdm

# Add helper_code to path if it's not in the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_code import find_records, load_signals

def convert_all_wfdb_to_npy(input_dir, output_dir):
    """
    Converts all WFDB records in a directory to .npy files.
    """
    print(f"Finding records in '{input_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    records = find_records(input_dir)
    if not records:
        print(f"No records found in {input_dir}. Exiting.")
        return

    print(f"Found {len(records)} records. Converting to .npy format...")
    for record_name in tqdm(records, desc="Converting records"):
        try:
            record_path = os.path.join(input_dir, record_name)
            signal, fields = load_signals(record_path)
            
            # The model expects (channels, seq_len), but load_signals gives (seq_len, channels)
            ecg_data = signal.T
            
            output_filename = os.path.join(output_dir, f"{os.path.basename(record_name)}.npy")
            np.save(output_filename, ecg_data)
        except Exception as e:
            print(f"Error processing record {record_name}: {e}", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert WFDB files in a directory to NumPy (.npy) files.')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory containing the WFDB (.hea, .dat) files.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save the output .npy files.')
    args = parser.parse_args()
    
    convert_all_wfdb_to_npy(args.input_dir, args.output_dir)