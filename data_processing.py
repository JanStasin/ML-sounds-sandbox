import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pydub

raw_data_loc = '/Users/jansta/learn/acoustics/ESC-50-master/audio/'

out_loc = '/Users/jansta/learn/acoustics/spects/'


# List of supported audio file extensions
supported_formats = ('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a')

# Get all audio files in the directory
audio_files = [
    f for f in os.listdir(raw_data_loc)
    if f.lower().endswith(supported_formats)
]

SR = 44100
FRAME = 512
n_mels = 32


for idx, file_name in enumerate(audio_files[:10]):
    file_path = os.path.join(raw_data_loc, file_name)

    name = file_name.split('.')[0]
    
    try:
        # Load the audio file
        # y, sr = librosa.load(file_path, sr=SR, mono=True)
        # print(sr)
        # # Save the 1D time series as a NumPy array
        # timeseries_path = f'ts/ts_{name}.npy'
        # np.save(timeseries_path, y)
        # Actual recordings are sometimes not frame accurate, so we trim/overlay to exactly 5 seconds
        data = pydub.AudioSegment.silent(duration=5000)
        data = data.overlay(pydub.AudioSegment.from_file(file_path)[0:5000])
        y = (np.fromstring(data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  
        print(y.shape)

        #y_dB = librosa.amplitude_to_db(np.abs(y), ref=np.max)

        mel_spect = librosa.feature.melspectrogram(y, sr=SR, hop_length=FRAME)
        #mel_spect_dB = librosa.feature.melspectrogram(y, sr=SR, hop_length=FRAME)
        # # Convert amplitude to decibels for better visualization
        # S_db = librosa.amplitude_to_db(S_abs, ref=np.max)

        # power_spect = librosa.db_to_power(spect)
        # Number of Mel bands

        # mel_basis = librosa.filters.mel(sr=44100, n_fft=n_fft, n_mels=n_mels)
        # mel_spect = np.dot(mel_basis, power_spect)

         # Optionally, save the magnitude spectrogram as a matrix
        spectrogram_matrix_path = f'spects/spect_mat_{name}.npy'
        np.save(spectrogram_matrix_path, mel_spect)
        
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spect, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {file_name}')
        plt.tight_layout()
        
        # Save the spectrogram image
        spectrogram_path = f'spects/spectrogram_{name}.png'
        plt.savefig(spectrogram_path)
        plt.close()
        
        print(f'Processed {file_name}:')
        #print(f'- Time series saved to {timeseries_path}')
        print(f'- Spectrogram saved to {spectrogram_matrix_path}\n')
        
    except Exception as e:
        print(f'Error processing {file_name}: {e}\n')

