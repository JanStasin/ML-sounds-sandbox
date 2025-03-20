import pydub
import librosa
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

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
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 53, 256)  # Adjusted based on pooling layers
        #self.fc1 = nn.Linear(64 * 4 * 26, 256)  # Adjusted based on pooling layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)  # classes
        self.fc4 = nn.Linear(64, self.n_classes)  # classes
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
        #x = self.pool(x)   # additional pooling layer - out: (BS, 64, 4, 26) 
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
        self.n_classes = n_classes
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
        self.fc1 = nn.Linear(256 * 4 * 26, 1024)  # Adjusted based on pooling layers
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.feature_maps = None
    
    def forward(self, x):
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)   
        x = self.conv2(x)  
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)   
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)   
        x = self.conv4(x)  
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)  
        self.feature_maps = x.detach()
        x = self.flatten(x) 
        x = self.dropout(x)
        x = self.fc1(x) 
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  
        x = self.relu(x)
        x = self.fc3(x)  # out: (BS, n_classes)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

## model for vanillaCAM

class AudioClassifNetCAM(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 256)  # Simplified linear layer
        #self.fc1 = nn.Linear(64 * 8 * 53, 512)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.n_classes)  # classes
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        # Store feature maps for CAM
        self.feature_maps = None
    
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
        # Store the feature maps after final convolution
        self.feature_maps = x.detach()
        x = self.global_avg_pool(x)
        x = self.flatten(x) # out: (4, 3328)
        x = self.fc1(x)  # out: (BS, 256)
        x = self.relu(x)
        x = self.fc2(x)  # out: (BS, 128)
        x = self.relu(x)
        x = self.fc3(x)  # out: (BS, n_classes)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

    def getCAM(self, class_idx):
        if self.feature_maps is None:
            raise ValueError("Feature maps are not set. Run a forward pass first.")
        
        if class_idx >= self.n_classes:
            raise ValueError(f"Class index {class_idx} is out of bounds for {self.n_classes} classes.")
        
        # Get the feature maps from the last convolutional layer
        feature_maps = self.feature_maps.squeeze(0)
        
        # Get the weights for the final fully connected layer
        weights = self.fc4.weight[class_idx].detach()
        #print(f"Weights shape before reshape: {weights.shape}")
        weights = weights.view(1,-1, 1, 1)
        #print(f"Weights shape after reshape: {weights.shape}")
    
        cam = torch.sum(feature_maps * weights, dim=1, keepdim=True)
        #print(f"CAM shape after computation: {cam.shape}")
        
        # Normalize the CAM
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam                 

    def generate_cam_visualization(self, cam, test_inp):
        """
        Generate CAM visualization for a given input
        Args:
            input_tensor (Tensor): Input audio spectrogram tensor
            class_idx (int): Index of the target class   
        Returns:
            visualization (ndarray): CAM visualization overlaid on input
        """
        #cam = self.getCAM(class_idx)

        # First squeeze to remove singleton dimensions
        cam = cam.squeeze(0)  # Remove batch dimension
        cam = cam.squeeze(0)  # Remove channel dimension

        # Now resize to match input dimensions
        # Get the height and width from input tensor
        height, width = test_inp.shape[1:]

        # Resize using both dimensions
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),  # Add back dimensions for interpolation
            size=(height, width),           # Use both dimensions
            mode='bilinear',
            align_corners=True
        )

        # Remove added dimensions and convert to numpy
        visualization = cam.squeeze().cpu().numpy()
        
        return visualization


class AudioClassifNetXAI_old(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.n_classes = n_classes
        
        # First block: Keep original (unchanged)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second block: Fixed channel progression
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13312, 512),  # Changed from 256 to 6272
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x):
        # Store feature maps after second convolutional block
        x = self.conv_block1(x)
        #print(x.size())
        x = self.conv_block2(x)
        #print(x.size())
        
        # Store feature maps for CAM visualization
        self.feature_maps = x
        
        x = self.fc_block(x)
        return x
        
    def forward(self, x):
        # Store feature maps after second convolutional block
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # Store feature maps for CAM visualization
        self.feature_maps = x.detach()
        
        x = self.fc_block(x)
        return x

    def getCAM(self, class_idx):
        if self.feature_maps is None:
            raise ValueError("Feature maps are not set. Run a forward pass first.")
        
        if class_idx >= n_classes:
            raise ValueError(f"Class index {class_idx} is out of bounds for {n_classes} classes.")
        
        # Get the feature maps from the last convolutional layer
        feature_maps = self.feature_maps.squeeze(0)
        
        # Get the weights for the final fully connected layer
        weights = self.fc_block[7].weight[class_idx].detach()
        #print(f"Weights shape before reshape: {weights.shape}")
        weights = weights.view(1,-1, 1, 1)
        #print(f"Weights shape after reshape: {weights.shape}")
    
        cam = torch.sum(feature_maps * weights, dim=1, keepdim=True)
        #print(f"CAM shape after computation: {cam.shape}")
        
        # Normalize the CAM
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam                 

    def generate_cam_visualization(self, cam, test_inp):
        """
        Generate CAM visualization for a given input
        Args:
            input_tensor (Tensor): Input audio spectrogram tensor
            class_idx (int): Index of the target class   
        Returns:
            visualization (ndarray): CAM visualization overlaid on input
        """
        #cam = self.getCAM(class_idx)

        # First squeeze to remove singleton dimensions
        cam = cam.squeeze(0)  # Remove batch dimension
        cam = cam.squeeze(0)  # Remove channel dimension

        # Now resize to match input dimensions
        # Get the height and width from input tensor
        height, width = test_inp.shape[1:]

        # Resize using both dimensions
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),  # Add back dimensions for interpolation
            size=(height, width),           # Use both dimensions
            mode='bilinear',
            align_corners=True
        )

        # Remove added dimensions and convert to numpy
        visualization = cam.squeeze().cpu().numpy()
        
        return visualization


class AudioClassifNetXAI(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        
        # First Convolutional Block
        self.conv_block1 = nn.Sequential(
            # First convolution: increase number of channels to 16
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # reduces height and width by 2
            
            # Second convolution: further increase channels to 32, also add BatchNorm
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second Convolutional Block with a fixed channel progression
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Global average pooling to collapse the spatial dimensions to 1x1.
        # This avoids having to hard-code the flattened size.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected block, now starting from a known feature size (128)
        self.fc_block = nn.Sequential(
            nn.Flatten(),              # Flattens (B, 128, 1, 1) into (B, 128)
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x: torch.Tensor, store_feature_maps: bool = False) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        if store_feature_maps:
            # Detaching feature maps for visualization (e.g., Grad-CAM)
            self.feature_maps = x.detach()
        
        # Global average pooling: converts (B, 128, H, W) to (B, 128, 1, 1)
        x = self.global_pool(x)
        x = self.fc_block(x)
        # Note: Do not apply an activation like softmax here if you're using CrossEntropyLoss
      
        return x

