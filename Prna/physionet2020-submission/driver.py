#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier, run_12ECG_classifier_batch

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data


def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')



if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        raise Exception('Usage: python driver.py model_input input_directory output_directory [batch_size]')

    model_input = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]

    batch_size = None
    if len(sys.argv) == 5:
        try:
            batch_size = int(sys.argv[4])
            if batch_size <= 0:
                print('Warning: batch_size must be a positive integer. Defaulting to non-batch mode.')
                batch_size = None
        except ValueError:
            print('Warning: Invalid batch_size. Defaulting to non-batch mode.')
            batch_size = None

    # Find files.
    input_files = []
    for _, _, f in os.walk(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model(model_input)

    print('Extracting 12ECG features...')
    num_files = len(input_files)

    if batch_size:
        # Iterate over files in batches.
        for i in range(0, num_files, batch_size):
            batch_input_files = input_files[i:i+batch_size]
            data_batch = []
            header_data_batch = []
            
            print(f'Processing batch {i//batch_size + 1}/{(num_files + batch_size - 1)//batch_size}...')

            for f_idx, f in enumerate(batch_input_files):
                print('    {}/{} (File: {})... '.format(i + f_idx + 1, num_files, f))
                tmp_input_file = os.path.join(input_directory,f)
                data, header_data = load_challenge_data(tmp_input_file)
                data_batch.append(data)
                header_data_batch.append(header_data)
            
            if not data_batch:
                continue

            batch_labels, batch_scores, classes_batch = run_12ECG_classifier_batch(data_batch, header_data_batch, model)
            
            # Save results for each file in the batch
            for j, f in enumerate(batch_input_files):
                current_label = batch_labels[j]
                current_score = batch_scores[j]
                save_challenge_predictions(output_directory, f, current_score, current_label, classes_batch)
    else:
        # Iterate over files individually.
        for i, f in enumerate(input_files):
            print('    {}/{}...'.format(i+1, num_files))
            tmp_input_file = os.path.join(input_directory,f)
            data,header_data = load_challenge_data(tmp_input_file)
            current_label, current_score,classes = run_12ECG_classifier(data,header_data, model)
            # Save results.
            save_challenge_predictions(output_directory,f,current_score,current_label,classes)

    print('Done.')
