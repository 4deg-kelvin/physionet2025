import argparse
import h5py
import numpy as np
import os
import os.path
import pandas as pd
import sys
import wfdb
import multiprocessing
from tqdm import tqdm

from helper_code import is_integer, is_boolean, sanitize_boolean_value
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Convert .dat files to .mat files (optional).
def convert_dat_to_mat(record, write_dir=None):
    import wfdb.io.convert
    if write_dir:
        cwd = os.getcwd()
        os.chdir(write_dir)
    with suppress_stdout():
        wfdb.io.convert.matlab.wfdb_to_mat(record)
    os.remove(record + '.hea')
    os.remove(record + '.dat')
    os.rename(record + 'm.hea', record + '.hea')
    os.rename(record + 'm.mat', record + '.mat')
    out = ''
    with open(record + '.hea', 'r') as f:
        for l in f:
            if l.startswith('#Creator') or l.startswith('#Source'):
                continue
            out += l.replace(record + 'm', record)
    with open(record + '.hea', 'w') as f:
        f.write(out)
    if write_dir:
        os.chdir(cwd)

# Fix checksums for Python WFDB library.
def fix_checksums(record, checksums=None):
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)
    header = record + '.hea'
    out = ''
    with open(header, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                num_leads = int(l.split()[1])
            if 0 < i <= num_leads and not l.startswith('#'):
                parts = l.split()
                parts[6] = str(checksums[i-1])
                l = ' '.join(parts) + '\n'
            out += l
    with open(header, 'w') as f:
        f.write(out)

# Worker: process one signal-file group.
def process_signal_group(args):
    signal_file, output_path, demographics, labels, signal_format = args
    try:
        f = h5py.File(signal_file, 'r')
    except OSError:
        return signal_file

    with f:
        os.makedirs(output_path, exist_ok=True)
        exam_to_age, exam_to_sex = demographics
        exam_to_chagas = labels

        exam_ids = list(f['exam_id'])
        tracings = f['tracings']

        for idx, exam_id in enumerate(tqdm(exam_ids,
                                            desc=f"File {os.path.basename(signal_file)}",
                                            unit="exam")):
            if exam_id not in exam_to_chagas:
                continue
            sig = np.asarray(tracings[idx], dtype=np.float32)
            # trim zero padding
            start, end = 0, sig.shape[0]
            while start < end and np.all(sig[start] == 0): start += 1
            while end > start and np.all(sig[end-1] == 0): end -= 1
            if start >= end:
                continue
            sig = sig[start:end]
            # to digital
            gain, bits = 1000, 16
            digi = np.round(gain * sig)
            digi = np.clip(digi, -2**(bits-1)+1, 2**(bits-1)-1)
            digi[~np.isfinite(digi)] = -2**(bits-1)
            digi = digi.astype(np.int32)
            # metadata
            age = exam_to_age[exam_id]
            sex = exam_to_sex[exam_id]
            chagas = exam_to_chagas[exam_id]
            comments = [f'Age: {age}', f'Sex: {sex}', f'Chagas label: {chagas}', 'Source: CODE-15%']
            record = str(exam_id)
            wfdb.wrsamp(
                record,
                fs=400,
                units=['mV'] * digi.shape[1],
                sig_name=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6'],
                d_signal=digi,
                fmt=[str(bits)] * digi.shape[1],
                adc_gain=[gain] * digi.shape[1],
                baseline=[0] * digi.shape[1],
                write_dir=output_path,
                comments=comments,
            )
            if signal_format == 'mat':
                convert_dat_to_mat(record, write_dir=output_path)
            chk = np.sum(digi, axis=0, dtype=np.int16)
            fix_checksums(os.path.join(output_path, record), chk)
    return None

# Main: parse args and launch pool with progress, collecting errors.
def get_parser():
    parser = argparse.ArgumentParser(description='Prepare CODE-15% dataset with multiprocessing')
    parser.add_argument('-i', '--signal_files', nargs='+', required=True,
                        help='List of input HDF5 signal files')
    parser.add_argument('-d', '--demographics_file', required=True,
                        help='CSV of demographics')
    parser.add_argument('-l', '--labels_file', required=True,
                        help='CSV of Chagas labels')
    parser.add_argument('-f', '--signal_format', type=str, default='dat', choices=['dat','mat'],
                        help='Output signal format: dat (WFDB) or mat')
    parser.add_argument('-o', '--output_paths', nargs='+', required=True,
                        help='Output directory paths (one per signal file or same for all)')
    parser.add_argument('-e', '--error_file', type=str, default='failed_hdf5_files.txt',
                        help='Path to write names of HDF5 files that failed to open')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    print("Starting CODE-15% dataset preparation")
    df_demo = pd.read_csv(args.demographics_file)
    exam_to_age = {int(r['exam_id']): int(r['age']) for _, r in df_demo.iterrows()}
    exam_to_sex = {int(r['exam_id']): ('Male' if sanitize_boolean_value(r['is_male']) else 'Female')
                   for _, r in df_demo.iterrows()}
    df_lab = pd.read_csv(args.labels_file)
    exam_to_chagas = {int(r['exam_id']): bool(sanitize_boolean_value(r['chagas']))
                      for _, r in df_lab.iterrows()}
    sf = args.signal_files
    op = args.output_paths if len(args.output_paths) == len(sf) else [args.output_paths[0]] * len(sf)
    demographics = (exam_to_age, exam_to_sex)
    labels = exam_to_chagas
    fmt = args.signal_format
    tasks = [(sf[i], op[i], demographics, labels, fmt) for i in range(len(sf))]
    failed = []
    print(f"Processing {len(tasks)} signal files with {multiprocessing.cpu_count()} CPU(s)")
    with multiprocessing.Pool(processes=min(len(tasks), multiprocessing.cpu_count())) as pool:
        for result in tqdm(pool.imap_unordered(process_signal_group, tasks),
                           total=len(tasks), desc="Signal files", unit="file"): 
            if result:
                failed.append(result)
    if failed:
        with open(args.error_file, 'w') as f:
            for path in failed:
                f.write(path + '\n')
        print(f"Wrote {len(failed)} failed file(s) to {args.error_file}")
