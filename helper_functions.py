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


def resize_cam_to_input(cam, input_shape):
    """
    Resize CAM to match input dimensions while preserving aspect ratio
    """
    # Get target dimensions
    target_height = input_shape[1]  # Height of input
    target_width = input_shape[2]   # Width of input
    
    # Resize CAM using bilinear interpolation
    cam_resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=True
    )
    return cam_resized.squeeze(0).squeeze(0)  # Remove batch and channel dims
