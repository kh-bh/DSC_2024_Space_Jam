import torch
from torch.utils.data import Dataset
import numpy as np
from cardiac_ml_tools import get_standard_leads, read_data_dirs, get_activation_time
import glob, re, os
import matplotlib.pyplot as plt


class ECGDataset(Dataset):
    def __init__(self, file_pairs):
        self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        ecg_file, vm_file = self.file_pairs[idx]

        # Load ECG data
        ecg_data = np.load(ecg_file)
        ecg_data = get_standard_leads(ecg_data)  # Convert to 12-lead
        ecg_data = self.normalize_ecg(ecg_data)  # Gives tensor of dimensions [500,12]

        ecg_data = ecg_data.T  # Gives tensor of dimensions [12, 500]

        # Load and process Vm data
        vm_data = np.load(vm_file)

        activation_time = get_activation_time(vm_data)
        #activation_time = activation_time.T

        return torch.FloatTensor(ecg_data), torch.FloatTensor(activation_time)

    def normalize_ecg(self, ecg):
        """
        Normalize each lead of the ECG data so that max(lead) - min(lead) = 1

        :param ecg: numpy array of shape (12, 500)
        :return: normalized numpy array of shape (12, 500)
        """
        normalized_ecg = np.zeros_like(ecg)
        for i in range(ecg.shape[0]):
            lead_min = np.min(ecg[i])
            lead_max = np.max(ecg[i])
            lead_range = lead_max - lead_min
            if lead_range != 0:
                normalized_ecg[i] = (ecg[i] - lead_min) / lead_range
            else:
                normalized_ecg[i] = ecg[i] - lead_min  # If the lead is constant, just center it at 0
        return normalized_ecg


# Example usage
if __name__ == "__main__":
    # Assuming you have a list of file pairs
    data_dirs = []
    regex = r'data_hearts_dd_0p2*'
    DIR = '../../intracardiac_dataset/'  # This should be the path to the intracardiac_dataset, it can be downloaded using data_science_challenge_2023/download_intracardiac_dataset.sh
    for x in os.listdir(DIR):
        if re.match(regex, x):
            data_dirs.append(DIR + x)
    file_pairs = read_data_dirs(data_dirs)

    dataset = ECGDataset(file_pairs)

    # Get a sample
    ecg_sample, vm_sample = dataset[0]
    print("ECG shape:", ecg_sample.shape)
    print("VM shape:", vm_sample.shape)

    # Verify normalization
    for i in range(12):
        lead_range = torch.max(ecg_sample[i]) - torch.min(ecg_sample[i])
        print(f"Lead {i + 1} range: {lead_range.item():.6f}")
