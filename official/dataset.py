import torch
from torch.utils.data import Dataset, DataLoader

import sys
import helper_code
import team_code
import os 
import numpy as np

class SamitropDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.records = helper_code.find_records(data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        signal = np.array(helper_code.load_signals(os.path.join(self.data_folder, record))[0]).T # load_signals returns a tuple of signals and extra info
        # Pad with an equal number of zeros on either size to make signal length 4096
        signal = np.pad(signal, ((0, 0), (0, 4096 - signal.shape[1])), mode='constant', constant_values=0)

        assert signal.shape == (12, 4096), f"Signal shape is {signal.shape} for record {record}, which it should be (12x4096)"

        label = helper_code.load_label(os.path.join(self.data_folder, record))

        if self.transform:
            sample = self.transform(sample)

        return signal, label