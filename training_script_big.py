import librosa
import librosa.display
import numpy as np
# plotting

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

## import custom made datset class:
from audio_ds_model import AudioDataset, AudioClassifNetBig
## and the external trainig function:
from training_func import run_training

# load preprocessed spectrograms data:
dict_mats = np.load('/Users/jansta/learn/acoustics/dict_mats_dB.npy', allow_pickle=True).item()
all_labels = list(dict_mats['A'].keys())

transform = transforms.Compose(
    [transforms.Resize((64,431)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])


NUM_EPOCHS = 1000
LR = 0.001

# #choose_labels:
# for n in range(10,50,5):
chosen_labels = all_labels
print(f'Number of labels: {len(chosen_labels)} --> {chosen_labels}')
print(f'Epochs  {NUM_EPOCHS} learning rate {LR}')
encoded_labels = {}
for i, label in enumerate(chosen_labels):
    encoded_labels[label] = i

# Create dataset with transform
dataset = AudioDataset(dict_mats['A'], chosen_labels, encoded_labels, transform=transform)
datasetB = AudioDataset(dict_mats['B'], chosen_labels, encoded_labels, transform=transform)

# Create dataloaders
batch_size = 4
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(datasetB, batch_size=batch_size, shuffle=True)

## Create an  instance of the model:
n_classes = len(chosen_labels)
model = AudioClassifNetBig(n_classes)

## Run training:
print(f'Running training with {n_classes} classes')
out = run_training(model, train_loader, val_loader, n_classes, rate_l=LR, NUM_EPOCHS=NUM_EPOCHS, save=True)

## Plot the confusion matrix:
#sns.heatmap(out[2], annot=True, xticklabels=chosen_labels, yticklabels=chosen_labels)