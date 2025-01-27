import torch
import torch.nn as nn

class AudioClassifNet(nn.Module):
    def __init__(self) -> None:
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
        self.fc3 = nn.Linear(256, n_classes)
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
        x = self.softmax(x)
        return x