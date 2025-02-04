import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pydub

from timing_decor import timing_decorator
from audio_ds_model import get_mel_spect

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
n_mels = 64

@timing_decorator
def process_audio(audio_files, SR = 44100, FRAME = 512, n_mels = 64):
    for idx, file_name in enumerate(audio_files):
        file_path = os.path.join(raw_data_loc, file_name)

        name = file_name.split('.')[0]

        try:
            mel_spect = get_mel_spect(file_path, in_dB=True, SR=SR, FRAME=FRAME, n_mels=n_mels)

            # Optionally, save the magnitude spectrogram as a matrix
            spectrogram_matrix_path = f'spects/spect_dB_mat_{name}.npy'
            #np.save(spectrogram_matrix_path, mel_spect)

            # Plot the spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_spect, sr=SR, x_axis='frames', y_axis='linear')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram of {file_name}')
            plt.tight_layout()
            # Save the spectrogram image
            spectrogram_path = f'spects/dB_spectrogram_{name}.png'
            plt.savefig(spectrogram_path)
            plt.close()

            print(f'Processed {file_name}:')
            #print(f'- Time series saved to {timeseries_path}')
            print(f'- Spectrogram saved to {spectrogram_matrix_path}\n')

        except Exception as e:
            print(f'Error processing {file_name}: {e}\n')
    print('processing done')

if __name__ == '__main__':
    process_audio(audio_files, SR, FRAME, n_mels)
    
    # for idx, file_name in enumerate(audio_files):

    #     file_path = os.path.join(raw_data_loc, file_name)

    #     name = file_name.split('.')[0]
    
    # try:
    #     mel_spect = get_mel_spect(file_path, in_dB=True, SR=SR, FRAME=FRAME, n_mels=n_mels)

    #     # Optionally, save the magnitude spectrogram as a matrix
    #     spectrogram_matrix_path = f'spects/spect_dB_mat_{name}.npy'
    #     #np.save(spectrogram_matrix_path, mel_spect)
        
    #     # Plot the spectrogram
    #     plt.figure(figsize=(10, 4))
    #     librosa.display.specshow(mel_spect, sr=SR, x_axis='frames', y_axis='linear')
    #     plt.colorbar(format='%+2.0f dB')
    #     plt.title(f'Spectrogram of {file_name}')
    #     plt.tight_layout()
    #     # Save the spectrogram image
    #     spectrogram_path = f'spects/dB_spectrogram_{name}.png'
    #     plt.savefig(spectrogram_path)
    #     plt.close()

    #     print(f'Processed {file_name}:')
    #     #print(f'- Time series saved to {timeseries_path}')
    #     print(f'- Spectrogram saved to {spectrogram_matrix_path}\n')
        
    # except Exception as e:
    #     print(f'Error processing {file_name}: {e}\n')
    

