import pydub
import librosa
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

def get_mel_spect(file_path, in_dB=True, SR=22050, FRAME=512, n_mels=128):
    data = pydub.AudioSegment.silent(duration=5000)
    data = data.overlay(pydub.AudioSegment.from_file(file_path)[0:5000])
    y = (np.frombuffer(data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)
    
    mel_spect = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=FRAME, n_mels=n_mels)
    
    if in_dB:
        mel_spect_dB = librosa.power_to_db(mel_spect, ref=np.max)
        return mel_spect_dB
    else:
        return mel_spect

class AudioDataset(Dataset):
    def __init__(self, dict_mats, chosen_labels, encoded_labels, transform=None):
        self.X = []
        self.y = []
        self.transform = transform
        for key in dict_mats.keys():
            if key in chosen_labels:
                for i in range(len(dict_mats[key])):
                    self.X.append(dict_mats[key][i])
                    self.y.append(encoded_labels[key])
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        # Add a channel dimension
        sample = np.expand_dims(sample, axis=0)
        
        # Convert to tensor
        sample = torch.FloatTensor(sample)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

class AudioClassifNet(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        #self.fc1 = nn.Linear(64 * 4 * 53, 256)  # Adjusted based on pooling layers
        self.fc1 = nn.Linear(64 * 4 * 26, 256)  # Adjusted based on pooling layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)  # classes
        self.fc4 = nn.Linear(64, n_classes)  # classes
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)  # out: (BS, 16, 64, 431)
        x = self.relu(x)
        x = self.pool(x)   # out: (BS, 16, 32, 215)
        x = self.conv2(x)  # out: (BS, 32, 32, 215)
        x = self.relu(x)
        x = self.pool(x)   # out: (BS, 32, 16, 107)
        x = self.conv3(x)  # out: (BS, 64, 8, 107)
        x = self.relu(x)
        x = self.pool(x)   # out: (BS, 64, 8, 53)
        x = self.pool(x)   # additional pooling layer - out: (BS, 64, 4, 26) 
        x = self.flatten(x) # out: (4, 3328)
        x = self.fc1(x)  # out: (BS, 256)
        x = self.relu(x)
        x = self.fc2(x)  
        x = self.relu(x)
        x = self.fc3(x)  
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

class AudioClassifNetBig(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        #  batch normalization layers to stabilize and accelerate training.
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        # dropout layers to prevent overfitting.
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 26, 512)  # Adjusted based on pooling layers
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)  # out: (BS, 32, 64, 431)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)   # out: (BS, 32, 32, 215)
        
        x = self.conv2(x)  # out: (BS, 64, 32, 215)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)   # out: (BS, 64, 16, 107)
        
        x = self.conv3(x)  # out: (BS, 128, 16, 107)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)   # out: (BS, 128, 8, 53)
        
        x = self.conv4(x)  # out: (BS, 256, 8, 53)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)   # out: (BS, 256, 4, 26)
        
        x = self.flatten(x) # out: (BS, 256 * 4 * 26)
        x = self.dropout(x)
        x = self.fc1(x)  # out: (BS, 512)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # out: (BS, 256)
        x = self.relu(x)
        x = self.fc3(x)  # out: (BS, n_classes)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x