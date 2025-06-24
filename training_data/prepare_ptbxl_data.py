#!/usr/bin/env python

# Load libraries.
import argparse
import numpy as np
import os
import os.path
import pandas as pd
import shutil
import sys
import wfdb
from tqdm import tqdm  # <--- ADDED

from helper_code import find_records, get_signal_files, is_integer

# Parse arguments.
def get_parser():
    description = 'Prepare the PTB-XL database for use in the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)  # records100 or records500
    parser.add_argument('-d', '--ptbxl_database_file', type=str, required=True)  # ptbxl_database.csv
    parser.add_argument('-f', '--signal_format', type=str, required=False, default='dat', choices=['dat', 'mat'])
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Suppress stdout for noisy commands.
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout

# Convert .dat files to .mat files (optional).
def convert_dat_to_mat(record, write_dir=None):
    import wfdb.io.convert
    import os

    # Split record into directory + basename
    record_dir, record_basename = os.path.split(record)

    # Save current working directory
    cwd = os.getcwd()

    if write_dir:
        # Preserve subdirectory structure!
        output_subdir = os.path.join(write_dir, record_dir)
        os.makedirs(output_subdir, exist_ok=True)
        os.chdir(output_subdir)
        record_to_convert = record_basename
    else:
        record_to_convert = record


    # print(f'Calling wfdb_to_mat on {record_to_convert} in {os.getcwd()}')
    with suppress_stdout():
        wfdb.io.convert.matlab.wfdb_to_mat(record_to_convert)

    # Remove .dat + .hea — use basename, since we're now in write_dir
    try:
        os.remove(record_to_convert + '.hea')
        os.remove(record_to_convert + '.dat')
    except FileNotFoundError:
        # Sometimes the .dat or .hea may not exist — skip safely
        pass

    # Rename .hea if present — note: PTB-XL often skips writing hrm.hea
    src_hea = record_to_convert + 'm' + '.hea'
    dst_hea = record_to_convert + '.hea'
    if os.path.exists(src_hea):
        os.rename(src_hea, dst_hea)
        # print(f'Renamed {src_hea} → {dst_hea}')
        # Fix header contents
        with open(dst_hea, 'r') as f:
            output_string = ''
            for l in f:
                if l.startswith('#Creator') or l.startswith('#Source'):
                    pass
                else:
                    l = l.replace(record_to_convert + 'm', record_to_convert)
                    output_string += l

        with open(dst_hea, 'w') as f:
            f.write(output_string)
    else:
        print(f'Info: {src_hea} not created — keeping original .hea.')

    # Rename .mat — this should always be created
    src_mat = record_to_convert + 'm' + '.mat'
    dst_mat = record_to_convert + '.mat'
    if os.path.exists(src_mat):
        os.rename(src_mat, dst_mat)
        # print(f'Renamed {src_mat} → {dst_mat}')
    else:
        print(f'WARNING: {src_mat} not created — conversion may have failed.')

    # Restore previous working directory
    if write_dir:
        os.chdir(cwd)


# Fix the checksums from the Python WFDB library.
def fix_checksums(record, checksums=None):
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)

    header_filename = os.path.join(record + '.hea')
    string = ''
    with open(header_filename, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                arrs = l.split(' ')
                num_leads = int(arrs[1])
            if 0 < i <= num_leads and not l.startswith('#'):
                arrs = l.split(' ')
                arrs[6] = str(checksums[i-1])
                l = ' '.join(arrs)
            string += l

    with open(header_filename, 'w') as f:
        f.write(string)

# Run script.
def run(args):
    # Load the demographic information.
    df = pd.read_csv(args.ptbxl_database_file, index_col='ecg_id')

    # Identify the header files.
    records = find_records(args.input_folder)

    # Update the header files to include demographics data and copy the signal files unchanged.
    print(f'Processing {len(records)} records...')
    for record in tqdm(records, desc='Processing records'):
        # Extract the demographics data.
        record_path, record_basename = os.path.split(record)
        ecg_id = int(record_basename.split('_')[0])
        row = df.loc[ecg_id]

        recording_date_string = row['recording_date']
        date_string, time_string = recording_date_string.split(' ')
        yyyy, mm, dd = date_string.split('-')
        date_string = f'{dd}/{mm}/{yyyy}'

        age = row['age']
        age = int(age) if is_integer(age) else float(age)

        sex = row['sex']
        if sex == 0:
            sex = 'Male'
        elif sex == 1:
            sex = 'Female'
        else:
            sex = 'Unknown'

        # Assume that all of the patients are negative for Chagas disease.
        label = False

        # Specify the label.
        source = 'PTB-XL'

        # Update the header file.
        input_header_file = os.path.join(args.input_folder, record + '.hea')
        output_header_file = os.path.join(args.output_folder, record + '.hea')

        input_path = os.path.join(args.input_folder, record_path)
        output_path = os.path.join(args.output_folder, record_path)

        os.makedirs(output_path, exist_ok=True)

        with open(input_header_file, 'r') as f:
            input_header = f.read()

        lines = input_header.split('\n')
        record_line = ' '.join(lines[0].strip().split(' ')[:4]) + '\n'
        signal_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.strip() and not l.startswith('#')) + '\n'
        comment_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.startswith('#') and not any((l.startswith(x) for x in ('# Age:', '# Sex:', '# Height:', '# Weight:', '# Chagas label:', '# Source:')))) + '\n'

        record_line = record_line.strip() + f' {time_string} {date_string} ' + '\n'
        signal_lines = signal_lines.strip() + '\n'
        comment_lines = comment_lines.strip() + f'# Age: {age}\n# Sex: {sex}\n# Chagas label: {label}\n# Source: {source}\n'

        output_header = record_line + signal_lines + comment_lines

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        # Copy the signal files if the input and output folders are different.
        if os.path.normpath(args.input_folder) != os.path.normpath(args.output_folder):
            signal_files = get_signal_files(input_header_file)  
            for input_signal_file in signal_files:
                output_signal_file = os.path.join(args.output_folder, os.path.relpath(input_signal_file, args.input_folder))
                if os.path.isfile(input_signal_file):
                    shutil.copy2(input_signal_file, output_signal_file)
                else:
                    raise FileNotFoundError(f'{input_signal_file} not found.')

        # Convert data from .dat files to .mat files, if requested.
        if args.signal_format in ('mat', '.mat'):
            convert_dat_to_mat(record, write_dir=args.output_folder)

        # Recompute the checksums as needed.
        fix_checksums(os.path.join(args.output_folder, record))

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
