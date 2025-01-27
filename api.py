import AudioClassifNet
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import os

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioClassifNet()
model.load_state_dict(torch.load('model/model.pth', map_location=device))
model.eval()



def audio_to_image(audio_data, name, SR=22050, FRAME=512, n_mels=128):
    # Load and preprocess audio
    data = pydub.AudioSegment.silent(duration=5000)
    audio = pydub.AudioSegment.from_file(io.BytesIO(audio_data))
    data = data.overlay(audio[0:5000])
    y = (np.frombuffer(data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  

    # Convert amplitude to dB
    y_dB = librosa.amplitude_to_db(np.abs(y), ref=np.max)

    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=FRAME, n_mels=n_mels)

    # Plot and save the spectrogram image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spect, sr=SR, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram of {name}')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return buf


@app.route('/predict', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        spectrogram_image = audio_to_image(file.read(), file.filename)
        
        # Make prediction - call model

        return jsonify({'message': 'Spectrogram created successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500    