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

print('All libraries imported successfully.')

# Get the directory containing input data from SageMaker environment variable
input_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')

# Verify the input directory exists and contains the expected file
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory not found: {input_dir}")

expected_file = os.path.join(input_dir, 'dict_mats_dB.npy')
if not os.path.exists(expected_file):
    raise FileNotFoundError(f"Input file not found: {expected_file}")

try:
    dict_mats = np.load(expected_file, allow_pickle=True).item()

except Exception as e:
    print(f"Error loading data from {expected_file}: {str(e)}")
    raise e

#inp_dir = '/opt/ml/input/data/'
dir_ = '/opt/ml/model/'
out_dir = '/opt/ml/output/'

#os.makedirs(inp_dir, exist_ok=True)
os.makedirs(dir_, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

for directory in [dir_, out_dir]:
    if os.path.exists(directory):
        print(f'Directory {directory} exists or was successfully created.')
    else:
        print(f'Failed to create directory {directory}.')

# # process the labels:
all_labels = list(dict_mats['A'].keys())
chosen_labels = all_labels
encoded_labels = {}
for i, label in enumerate(all_labels):
        encoded_labels[label] = i

transform = transforms.Compose(
    [transforms.Resize((64,431)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

dataset = AudioDataset(dict_mats['A'], chosen_labels, encoded_labels, transform=transform)

print('Dataset created successfully.')

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


NUM_EPOCHS = 5000
#LR = 0.00085

## Create an  instance of the model:
n_classes = len(chosen_labels)
model = AudioClassifNetXAI(n_classes)

## Run training:
print(f'Running training with {n_classes} classes')
out = run_training(model, train_loader, val_loader, encoded_labels, rate_l=LR, NUM_EPOCHS=NUM_EPOCHS, save=True, thresh=0)

## Plot the confusion matrix:
#sns.heatmap(out[2], annot=True, xticklabels=chosen_labels, yticklabels=chosen_labels)
