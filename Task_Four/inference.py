import os
import re

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from dataloader import ECGDataset
from model import SqueezeNet1D
from cardiac_ml_tools import read_data_dirs


def inference(test_file_pairs, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SqueezeNet1D().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataset = ECGDataset(test_file_pairs)

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=20, pin_memory=True)

    criterion = nn.L1Loss()

    all_errors = []
    predictions = []

    with torch.no_grad():
        for X, Y in test_loader:
            X_gpu, Y_gt_gpu = X.to(device), Y.to(device)
            Y_pred = model(X_gpu)

            predictions.append(Y_pred.cpu().numpy())

            errors = criterion(Y_pred, Y_gt_gpu)
            all_errors.append(float(errors.cpu().numpy()))

    all_predictions = np.array(predictions)

    np.save('predictions.npy', all_predictions)

    # Calculate average error and standard deviation across all samples and recording points
    avg_error = np.mean(all_errors)
    std_error = np.std(all_errors)

    print(f"Average error: {avg_error:.2f} msec")
    print(f"Standard deviation of error: {std_error:.2f} msec")

    return avg_error, std_error


if __name__ == "__main__":
    test_file_pairs = np.load('test.npy')
    inference(test_file_pairs=test_file_pairs, model_path="model_20240731_171505_1000.pth")
