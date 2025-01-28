import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


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
        self.fc3 = nn.Linear(128, n_classes)  # classes
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
        x = self.fc2(x)  # out: (BS, 128)
        x = self.relu(x)
        x = self.fc3(x)  # out: (BS, n_classes)
        x = self.softmax(x)
        return x