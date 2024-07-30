import os
import re

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from dataloader import ECGDataset
from model import SqueezeNet1D
from cardiac_ml_tools import read_data_dirs


def inference(test_file_pairs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SqueezeNet1D().to(device)
    model.load_state_dict(torch.load('best_model_StepLR.pth'))
    model.eval()

    # data_dirs = []
    # regex = r'data_hearts_dd_0p2*'
    # DIR = '../intracardiac_dataset/'
    # for x in os.listdir(DIR):
    #     if re.match(regex, x):
    #         data_dirs.append(DIR + x)
    # file_pairs = read_data_dirs(data_dirs)
    # print('Number of file pairs: {}'.format(len(file_pairs)))
    # # example of file pair
    # print("Example of file pair:")
    # print("{}\n{}".format(file_pairs[0][0], file_pairs[0][1]))
    # print("{}".format(file_pairs[1]))
    #test_file_pairs = file_pairs[14505:]

    #test_file_pairs = np.load('test.npy')
    test_dataset = ECGDataset(test_file_pairs)

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=20, shuffle=True, pin_memory=True)

    criterion = nn.L1Loss()

    all_errors = []

    with torch.no_grad():
        for X, Y in test_loader:
            X_gpu, Y_gt_gpu = X.to(device), Y.to(device)
            Y_pred = model(X_gpu)

            errors = criterion(Y_pred, Y_gt_gpu)
            all_errors.append(float(errors.cpu().numpy()))

    # Concatenate all errors
    #all_errors = np.concatenate(all_errors, axis=0)

    # Calculate average error and standard deviation across all samples and recording points
    avg_error = np.mean(all_errors)
    std_error = np.std(all_errors)

    print(f"Average error: {avg_error:.2f} msec")
    print(f"Standard deviation of error: {std_error:.2f} msec")

    return avg_error, std_error


if __name__ == "__main__":
    test_file_pairs = np.load('test.npy')
    inference(test_file_pairs=test_file_pairs)
