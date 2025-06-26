# How to use this Directory
This directory contains template code that you can structure your transformer/machine learning models around. It should abstract most of the boilerplate code needed to train and evaluate a model, allowing you to focus on the specific details of your model architecture and training process. It also includes utilities for loading data, creating optimizers, etc

# `dataloader.py`
This file contains the `DataLoader` class, which is responsible for loading and preprocessing your dataset. 
## `__getitem__` Method
The `__getitem__` method is where the data loading and preprocessing happens. It includes:
- Loading the ECG signal from a file. (REQUIRED)
- Resampling the signal to a unified frequency. (REQUIRED)
- Cleaning the signal and correcting polarity. (OPTIONAL)
- Normalizing the signal. (OPTIONAL)
- Handling different windowing methods (e.g., QRS detection). (REQUIRED)
- Returning the signal and its corresponding label.

All you need to do is pass in the list of records and the directory where the data is stored when you initialize the `ECGDataset` class. The `__getitem__` method will handle the rest.

# Train/val/test splits
To keep the challenge score consistent across our experiments, use the provided "train_val_test_sets.csv" file to split your dataset into training, validation, and test sets. This file contains the record IDs for each set, which you can use to filter your dataset. The provided code already includes logic to read this file and create the splits.

# `utils.py`
This file contains certain utility functions as well as constants that can be used in your model. The utility functions include computing challenge score, and a windowing function, which creates patches from the ECG signal according to different windowing methods. 

# `transformer.py`
This holds your model. You will most likely change this file the most, as this is what determines the architecture of your model.  

# `lightning_classifier.py`
This file contains a pytorch lightning module, that is responsible for handling the training and validation steps of your model, as well as the optimizers. It also controls the final loss (BCE loss) and the challenge score computation. You shouldn't need to change this file much, besides swapping out what model it uses. 

# tensorboard 
The given code already outputs a tensorboard log file, which you can use to visualize the training process. Note that each run, failed or not, creates a tensorboard log file, so you might accidentally create many files. No biggie, just a bit annoying. You can run the following command to start tensorboard (when in the correct directory where the `lightning_logs/` directory is located):

`tensorboard --logdir lightning_logs/ --bind_all`