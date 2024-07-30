import os
import re

import torch
from torch.utils.data import DataLoader
from torch import nn

from dataloader import ECGDataset
from model import SqueezeNet1D
from cardiac_ml_tools import read_data_dirs

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SqueezeNet1D().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    data_dirs = []
    regex = r'data_hearts_dd_0p2*'
    DIR = '../intracardiac_dataset/'
    for x in os.listdir(DIR):
        if re.match(regex, x):
            data_dirs.append(DIR + x)
    file_pairs = read_data_dirs(data_dirs)
    print('Number of file pairs: {}'.format(len(file_pairs)))
    # example of file pair
    print("Example of file pair:")
    print("{}\n{}".format(file_pairs[0][0], file_pairs[0][1]))
    print("{}".format(file_pairs[1]))

    test_file_pairs = file_pairs[14505:]
    test_dataset = ECGDataset(test_file_pairs)

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=20, shuffle=True, pin_memory=True)

    loss = nn.L1Loss()

    errors = []
    for X, Y in test_loader:
        X_gpu, Y_gt_gpu = X.to(device), Y.to(device)
        Y_pred = model(X_gpu)
        loss_per_item = loss(Y_pred, Y_gt_gpu)
        errors.append(loss_per_item)
    print(sum(errors)/len(errors))
