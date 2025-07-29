"""
Utility to locate .hea and .mat files using helper_code.py
"""
import os
from helper_code import find_records, get_header_file, get_signal_files

def locate_mat_and_hea_files(data_folder):
    """
    Locate all .mat and .hea files in the given folder using helper_code.py utilities.
    Returns a list of tuples: (record_name, hea_file, mat_files)
    """
    records = find_records(data_folder, file_extension='.hea')
    result = []
    for record in records:
        hea_file = get_header_file(os.path.join(data_folder, record))
        mat_files = [f for f in get_signal_files(os.path.join(data_folder, record)) if f.endswith('.mat')]
        result.append((record, hea_file, mat_files))
    return result

def save_records_for_semantic_search(files, output_path="located_records.txt"):
    """
    Save the located .hea and .mat files in a text format for semantic search indexing.
    """
    with open(output_path, "w") as f:
        for record, hea, mats in files:
            f.write(f"Record: {record}\n")
            f.write(f"Header: {hea}\n")
            f.write(f"MAT files: {', '.join(mats)}\n")
            f.write("\n")
if __name__ == "__main__":
    data_folder = "training_data/samitrop/samitrop_unzipped"
    files = locate_mat_and_hea_files(data_folder)
    for record, hea, mats in files:
        print(f"Record: {record}")
        print(f"  Header: {hea}")
        print(f"  MAT files: {mats}")
    save_records_for_semantic_search(files)
