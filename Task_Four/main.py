import os
import random
import re
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cardiac_ml_tools import read_data_dirs
from dataloader import ECGDataset
from model import SqueezeNet1D
from trainer import train


if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join("log", f"experiment_{run_id}")
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    # random.shuffle(file_pairs)
    #
    # train_file_pairs = np.array(file_pairs[0:12893])
    # np.save('train.npy', train_file_pairs)
    #
    # val_file_pairs = np.array(file_pairs[12893:14505])
    # np.save('val.npy', val_file_pairs)
    #
    # test_file_pairs = np.array(file_pairs[14505:])
    # np.save('test.npy', test_file_pairs)

    train_dataset = ECGDataset(np.load('train.npy'))
    val_dataset = ECGDataset(np.load('val.npy'))

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=20, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=20, pin_memory=True)

    trained_model = train(writer, device, model, train_loader, val_loader, num_epochs=1000, learning_rate=0.001)

    writer.close()
