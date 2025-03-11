import librosa
import librosa.display
import numpy as np
import sys
import os
# plotting
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

## import custom made datset class:
from audio_ds_model import AudioDataset, AudioClassifNetXAI
## and the external trainig function:
from training_func_gcam import run_training, gradCAMS_saver

n_epochs = 5000

inp_dir = '/opt/ml/input/data/'
dir_ = '/opt/ml/model/'
out_dir = '/opt/ml/output/'

os.makedirs(inp_dir, exist_ok=True)
os.makedirs(dir_, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# load preprocessed spectrograms data:
dict_mats = np.load(inp_dir+'dict_mats_dB.npy', allow_pickle=True).item()
all_labels = list(dict_mats['A'].keys())

transform = transforms.Compose(
    [transforms.Resize((64,431)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

NUM_EPOCHS = n_epochs
LR = 0.000075

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
batch_size = 5
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(datasetB, batch_size=batch_size, shuffle=True)

## Create an  instance of the model:
n_classes = len(chosen_labels)
model = AudioClassifNetXAI(n_classes)

## Run training:
print(f'Running training with {n_classes} classes')
out = run_training(model, train_loader, val_loader, encoded_labels, rate_l=LR, NUM_EPOCHS=NUM_EPOCHS, save=True, thresh=0)

## Plot the confusion matrix:
#sns.heatmap(out[2], annot=True, xticklabels=chosen_labels, yticklabels=chosen_labels)
