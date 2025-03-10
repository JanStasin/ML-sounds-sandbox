import os
import librosa
import librosa.display
import numpy as np
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torchaudio
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

# load preprocessed spectrograms data:
dict_mats = np.load('/Users/jansta/learn/acoustics/dict_mats_dB.npy', allow_pickle=True).item()
all_labels = list(dict_mats['A'].keys())

transform = transforms.Compose(
    [transforms.Resize((64,431)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])


NUM_EPOCHS = 1200
#LR = 0.00085

#choose_labels:
for LR in [0.0005,0.0002, 0.00008,0.00005, 0.00001]:
    chosen_labels = all_labels[:]
    #print(f'Number of labels: {n} --> {chosen_labels}')
    print(f'Epochs  {NUM_EPOCHS} learning rate {LR}')
    encoded_labels = {}
    for i, label in enumerate(chosen_labels):
        encoded_labels[label] = i

    # Create dataset with transform
    dataset = AudioDataset(dict_mats['A'], chosen_labels, encoded_labels, transform=transform)
    datasetB = AudioDataset(dict_mats['B'], chosen_labels, encoded_labels, transform=transform)

    # Create dataloaders
    batch_size = 5
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(datasetB, batch_size=batch_size, shuffle=True)

    ## Create an  instance of the model:
    n_classes = len(chosen_labels)
    model = AudioClassifNetXAI(n_classes)

    ## Run training:
    print(f'Running training with {model.n_classes} classes')
    out = run_training(model, train_loader, val_loader, n_classes, rate_l=LR, NUM_EPOCHS=NUM_EPOCHS, save=True)

    ## Plot the confusion matrix:
    #sns.heatmap(out[2], annot=True, xticklabels=chosen_labels, yticklabels=chosen_labels)
    #sns.savefig(f'confusion_matrix_{LR}.png')