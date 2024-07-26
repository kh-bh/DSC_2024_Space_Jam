import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch._dynamo


from cardiac_ml_tools import read_data_dirs
from dataloader import ECGDataset
from model import SqueezeNet1D

torch._dynamo.config.suppress_errors = True

def train(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for batch_ecg, batch_activation in train_loader:
            batch_ecg_gpu, batch_activation_gpu = batch_ecg.to(device), batch_activation.to(device)
            optimizer.zero_grad()
            with torch.enable_grad():
                outputs = model(batch_ecg_gpu)
                loss = criterion(outputs, batch_activation_gpu)
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_ecg, batch_activation in val_loader:
                batch_ecg_gpu_one, batch_activation_gpu_one = batch_ecg.to(device), batch_activation.to(device)
                outputs = model(batch_ecg_gpu_one)
                val_loss += criterion(outputs, batch_activation_gpu_one).item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SqueezeNet1D().to(device)
    model = torch.compile(model)

    data_dirs = []
    regex = r'data_hearts_dd_0p2*'
    DIR = '../../intracardiac_dataset/'  # This should be the path to the intracardiac_dataset, it can be downloaded using data_science_challenge_2023/download_intracardiac_dataset.sh
    for x in os.listdir(DIR):
        if re.match(regex, x):
            data_dirs.append(DIR + x)
    file_pairs = read_data_dirs(data_dirs)
    print('Number of file pairs: {}'.format(len(file_pairs)))
    # example of file pair
    print("Example of file pair:")
    print("{}\n{}".format(file_pairs[0][0], file_pairs[0][1]))
    print("{}".format(file_pairs[1]))

    train_file_pairs = file_pairs[0:15312]
    val_file_pairs = file_pairs[15312:16117]

    train_dataset = ECGDataset(train_file_pairs)
    val_dataset = ECGDataset(val_file_pairs)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001)
