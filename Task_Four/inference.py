import os
import re

import numpy as np
from scipy import stats

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


def calculate_pearson_correlation(predictions, ground_truth_pairs):
    """
    Calculate the Pearson correlation coefficient between predicted
    transmembrane potentials and ground truth potentials loaded from paired files.

    :param predictions: numpy array of shape (num_samples, num_time_steps)
                        containing predicted potentials
    :param ground_truth_pairs: numpy array of shape (num_samples, 2) containing pairs of
                               [pECG_filename, VM_filename] for each sample
    :return: numpy array of shape (num_samples,) containing Pearson
             correlation coefficients for each sample
    """
    num_samples = len(ground_truth_pairs)
    correlations = np.zeros(num_samples)

    for i, (_, vm_file) in enumerate(ground_truth_pairs):
        # Load ground truth VM data
        ground_truth = np.load(vm_file).T

        # Ensure both arrays are 1D
        pred = predictions[i].squeeze(0)
        pred_flat = pred.flatten()

        truth_flat = ground_truth.flatten()

        correlation, _ = stats.pearsonr(pred_flat, truth_flat)
        correlations[i] = correlation

    return correlations

def calculate_rmse(predictions, ground_truth_pairs):
    errors = []
    for i, (_, vm_file) in enumerate(ground_truth_pairs):
        ground_truth = np.load(vm_file).T  # (75, 500)
        pred = predictions[i].squeeze(0)    # (75, 500)
        rmse = np.sqrt(np.mean((pred - ground_truth)**2))
        errors.append(rmse)
    return np.array(errors)


if __name__ == "__main__":
    test_file_pairs = np.load('./data/test.npy')
    #inference(test_file_pairs=test_file_pairs, model_path="model_20240731_171505_1000.pth")
    c = calculate_pearson_correlation(np.load('./data/predictions.npy'), np.load('./data/test.npy'))
    print(np.mean(c))

    rmse = calculate_rmse(np.load('./data/predictions.npy'), np.load('./data/test.npy'))
    print(np.mean(rmse))
