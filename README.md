# Python Code for Physionet 2025, Edwards Lifesciences

# Important Links
- [Physionet Website Link](https://moody-challenge.physionet.org/2025/)
- [Data Download for PERSONAL Computers](https://drive.google.com/drive/folders/1s_EGSV_s6p2dUVeAnGcUHoXwb5F8EQyp?usp=drive_link)
- [Data Download for EDWARDS Computers](https://edwardslifesciences-my.sharepoint.com/:f:/g/personal/kelvin_nguyen_edwards_com/EiHr9xaX9A1IgaZWOjTohuoBhPT38WC3j919QldOntRQ0Q?e=N7Tgwa)

# Environment
For development using conda, use the `physionet_dev.yml` environment. For submission, the submission has its own requirements_linux.txt for pip and a dockerfile

### Mac Users:

Set up a conda or `venv` environment, then do the following `pip install` commands:

```
pip3 install torch torchvision torchaudio # double check the official pytorch website for the exact command
pip install numpy
pip install neurokit2
pip install h5py
pip install wfdb
pip install tqdm
pip install pytorch-lightning
```

# Preparing Data
First, note that we have three data sources: samitrop, ptb, and code15. First, proceed to the `training_data` directory. You'll need to download the datasets. 
- For Edwards employees, access it [here:](https://edwardslifesciences-my.sharepoint.com/:f:/g/personal/kelvin_nguyen_edwards_com/EiHr9xaX9A1IgaZWOjTohuoBhPT38WC3j919QldOntRQ0Q?e=N7Tgwa),
- If working on a personal laptop,use this gdrive [link:](https://drive.google.com/drive/folders/1s_EGSV_s6p2dUVeAnGcUHoXwb5F8EQyp?usp=sharing)
- I have the files on OneDrive. Unzip them, (including the inner exams_part<x> zip files for code15) then run the corresponding `prepare_<dataset>` file with the right dataset, with these parameters:

**FOR SAMITROP AND CODE15 (NOT PTB!!!!)**
*MAKE SURE TO HAVE -F AS MAT, NOT DAT*
```
python prepare_<samitrop or code15>_data.py \
    -i /path/to/signal_file_1.hdf5 /path/to/signal_file_2.hdf5 \
    -d /path/to/demographics.csv \
    -l /path/to/code15_chagas_labels.csv \
    -f mat \
    -o /path/to/output_dir/
```
**FOR PTB**
```
python prepare_ptbxl_data.py \
    -i /path/to/your/ptbxl/records500 \
    -d /path/to/your/ptbxl/ptbxl_database.csv \
    -f mat \
    -o /path/to/your/output_directory
```
Note that this will a while, and your computer will be unusuable because I set it to use multiprocessing, so it'll use all you CPU resources. go take a break. 

# Directory Format

## `official`
This is the directory containing all the official code that will be uploaded to the Physionet servers for it to run. *Keep in mind that `official` may have duplicated files from other directories (including Prna, the feature model), as there might be dev versions of the files that we may want to submit.* 

## `training_data`
This directory is also a dev directory, and is meant to store you physionet 2025 training dataset. Note that we have three data sources this year: ptb, samitrop, and code15. See the preparing data section for more information on what to do. 

## `Prna` 
Prna is our feature extractor -- however, this folder contains only NON-OFFICIAL code, ie, code that we won't send to submission. This is meant for dev work. It also contains training code to train Prna, as well as the singularity/docker containers required to train it. 

### How to run inference on PRNA (the feature model)
#### Loading the Model
The first thing you need to do is to download the `trained_model` directory, which has the pretrained weights. To do this, download the file [here](https://drive.google.com/drive/folders/1HBmXWUlKHxds9CUnM_hF-5n5XAcYlfEI?usp=sharing). When passing in the path of the model for training, just pass the path to `trained_model`. 
#### Performing Inference
Prna inference can be done either through the `driver.py` file, or the `run_12ECG_classifier.py` file. The `driver.py` file is meant for running predictions on a whole folder, and we also have batched inputs/ to speed things up. `run_12ECG_classifier` is what `driver.py` calls repeatedly to perform inference, but is more suited for individual record-by-record inference. 

# Creating your own models/transformer models etc
## To create your own transformer models, you have a lot of freedom as to what to use. However, I provide pretty good datset code in the `transformers.py` file in custom_transformer, especially in the __get_item__ function of ECGDataset, which will retrieve a ECG file for you and generate the required numpy 12 lead array (although it also does some preprocessing, which you can ignore in your code)

# Todo 
- look into component aware transformer, a transformer that uses multiple CNN's of diff kernel sizes all at once per lead, and averages the vector at the end to be passed into a transformer (see CardioPatternFormer paper)
- augument the transformer code
- work on getting more features
- use random forest classifier to see what features we can use

# Known Issues
- Some signal files are actually really short (this is mostly an issue in the code15 dataset). Make sure any feature selection you do which depends on a minimum input length doesn't break!
-     All of the data is either in 400 or 500 Hz. However, ignore the length guarantees given on the datasets' website -- they're wrong
- The environment file I made does NOT compile cleanly on macbooks/windows. It was meant for ubuntu (Linux) with a CUDA backend. I'm working on making a separate env file to handle it
- You CANNOT download the datasets from the ONedrive link if you're not on a edwards computer. I am working on adding a google drive link as well
