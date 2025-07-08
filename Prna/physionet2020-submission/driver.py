#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier, run_12ECG_classifier_batch
from tqdm import tqdm

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data


def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(os.path.basename(filename))[0]
    new_file = recording + '.csv'
    output_file = os.path.join(output_directory, new_file)

    # Include the filename as the recording number
    recording_string = '{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        # print('(in save_challenge_preds) Saving results for file: {} to {}'.format(filename, output_file))
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')



if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) < 4 or len(sys.argv) > 7:
        raise Exception('Usage: python driver.py model_input input_directory output_directory [batch_size] [eval_fraction] [eval_split]')
    print("Starting driver.py")

    model_input = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]

    print('Model input:', model_input)
    print('Input directory:', input_directory)
    print('Output directory:', output_directory)

    batch_size = None
    if len(sys.argv) >= 5:
        try:
            batch_size = int(sys.argv[4])
            if batch_size <= 0:
                print('Warning: batch_size must be a positive integer. Defaulting to non-batch mode.')
                batch_size = None
        except ValueError:
            print('Warning: Invalid batch_size. Defaulting to non-batch mode.')
            batch_size = None

    eval_fraction = 1.0
    eval_split = 0
    if len(sys.argv) >= 6:
        try:
            eval_fraction = float(sys.argv[5])
            if eval_fraction <= 0 or eval_fraction > 1.0:
                print('Warning: eval_fraction must be between 0 and 1. Defaulting to 1.0')
                eval_fraction = 1.0
        except ValueError:
            print('Warning: Invalid eval_fraction. Defaulting to 1.0')
            eval_fraction = 1.0

    if len(sys.argv) >= 7:
        try:
            eval_split = int(sys.argv[6])
            if eval_split < 0:
                print('Warning: eval_split must be >= 0. Defaulting to 0.')
                eval_split = 0
        except ValueError:
            print('Warning: Invalid eval_split. Defaulting to 0.')
            eval_split = 0

    # Find files.
    print('Finding input files!...')
    input_files = []
    modified_input_directory = os.getcwd()
    modified_input_directory = os.path.abspath(modified_input_directory)
    modified_input_directory = os.path.join(modified_input_directory, input_directory) 
    print('Searching in directory:', modified_input_directory)
    for root, _, files in os.walk(modified_input_directory):
        print('    Searching in directory:', root)
        for f in files:
            if os.path.isfile(os.path.join(root, f)) and not f.lower().startswith('.') and f.lower().endswith('.mat'):
                input_files.append(os.path.join(root, f))

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print('Found {} input files.'.format(len(input_files)))

    # Select subset of files based on eval_fraction and eval_split
    num_files_total = len(input_files)
    files_per_split = int(np.ceil(num_files_total * eval_fraction))
    start_idx = eval_split * files_per_split
    end_idx = min(start_idx + files_per_split, num_files_total)
    selected_input_files = input_files[start_idx:end_idx]

    print(f'Evaluating split {eval_split} with fraction {eval_fraction:.2f} -> {len(selected_input_files)} files out of {num_files_total} total.')

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model(model_input)

    print('Extracting 12ECG features...')

    if batch_size:
        # Iterate over files in batches.
        for i in tqdm(range(0, len(selected_input_files), batch_size), desc="Processing batches", unit="batch"):
            batch_input_files = selected_input_files[i:i+batch_size]
            data_batch = []
            header_data_batch = []

            for f_idx, f in enumerate(batch_input_files):
                tmp_input_file = os.path.join(input_directory, f)
                data, header_data = load_challenge_data(tmp_input_file)
                data_batch.append(data)
                header_data_batch.append(header_data)

            if not data_batch:
                continue

            batch_labels, batch_scores, classes_batch, skipped_files_indexes = run_12ECG_classifier_batch(data_batch, header_data_batch, model)
            skipped_files_indexes = set(skipped_files_indexes)
            processed_item_idx = 0

            for j, f in enumerate(batch_input_files):
                if j in skipped_files_indexes:
                    tqdm.write(f"Skipping file {f} due to preprocessing error.")
                    continue

                if processed_item_idx >= len(batch_labels):
                    tqdm.write(f"Warning: Mismatch for file {f}. Preprocessing OK (original index {j}), but no corresponding model output at processed index {processed_item_idx}. batch_labels length: {len(batch_labels)}")
                    continue

                current_label = batch_labels[processed_item_idx]
                current_score = batch_scores[processed_item_idx]
                save_challenge_predictions(output_directory, f, current_score, current_label, classes_batch)
                processed_item_idx += 1
    else:
        # Iterate over files individually.
        for f in tqdm(selected_input_files, desc="Processing files", unit="file"):
            tmp_input_file = f
            data, header_data = load_challenge_data(tmp_input_file)
            current_label, current_score, classes = run_12ECG_classifier(data, header_data, model)
            save_challenge_predictions(output_directory, f, current_score, current_label, classes)

    print('Done.')

