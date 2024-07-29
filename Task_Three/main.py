import os
import re

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cardiac_ml_tools import read_data_dirs
from dataloader import ECGDataset
from model import SqueezeNet1D
from trainer import train

if __name__ == "__main__":
    writer = SummaryWriter("log/experiment4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SqueezeNet1D().to(device)
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

    train_file_pairs = file_pairs[0:12893]
    val_file_pairs = file_pairs[12893:14505]
    test_file_pairs = file_pairs[14505:]

    train_dataset = ECGDataset(train_file_pairs)
    val_dataset = ECGDataset(val_file_pairs)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=20, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=20, pin_memory=True)

    trained_model = train(writer, device, model, train_loader, val_loader, num_epochs=1000, learning_rate=0.001)

    writer.close()
