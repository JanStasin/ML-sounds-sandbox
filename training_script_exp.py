import os
import librosa
import librosa.display
import numpy as np
# plotting
import matplotlib.pyplot as plt
#import seaborn as sns
from PIL import Image

import torch
#import torchaudio
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix
import random

## import custom made datset class:
from audio_ds_model import AudioDataset, AudioClassifNetXAI
## and the external trainig function:
from training_func_gcam import run_training, gradCAMS_saver

inp_dir = '/opt/ml/input/data/'
dir_ = '/opt/ml/model/'
out_dir = '/opt/ml/output/'

os.makedirs(inp_dir, exist_ok=True)
os.makedirs(dir_, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# load preprocessed spectrograms data:
dict_mats = np.load('dict_mats_dB.npy', allow_pickle=True).item()
# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
all_labels = list(dict_mats['A'].keys())

transform = transforms.Compose(
    [transforms.Resize((64,431)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])


NUM_EPOCHS = 5000
#LR = 0.00085

#choose_labels:
for LR in [0.001, 0.0005, 0.0002, 0.00008, 0.00005, 0.00001]:
    chosen_labels = all_labels

    print(f'Epochs  {NUM_EPOCHS} learning rate {LR}')
    encoded_labels = {}
    for i, label in enumerate(chosen_labels):
        encoded_labels[label] = i

    ## Create an  instance of the model:
    n_classes = len(chosen_labels)
    model = AudioClassifNetXAI(n_classes)

    ## Run training:
    print(f'Running training with {model.n_classes} classes')
    out = run_training(model, train_loader, val_loader, encoded_labels, rate_l=LR, NUM_EPOCHS=NUM_EPOCHS, save=True)
    print(f'Done for {LR} learning rate')
