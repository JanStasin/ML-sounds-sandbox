from audio_ds_model import *
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import os
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = np.load('working_models/results_and_model_acc_78.3_LR_0.001_nclasses_8.npy', allow_pickle=True).item()['model']
print(type(m), m)
model = AudioClassifNet(n_classes=8)
#model = AudioClassifNetBig()
model.load_state_dict(torch.load(m, map_location=device))
model.eval()

transform = transforms.Compose(
    [transforms.Resize((64,431)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

    try:
        spectrogram_image = get_mel_spect(file_path)
        # Add a channel dimension
        sample = np.expand_dims(spectrogram_image, axis=0)
        # Convert to tensor
        sample = torch.FloatTensor(sample)
        input_img = transform(sample)

        # Make prediction - call model
        with torch.no_grad():
            output = model(input_img.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            predicted_label = chosen_labels[predicted.item()]
            print(f'Predicted label: {predicted_label}')
            return jsonify({'predicted_label': predicted_label}), 200
        #return jsonify({'message': 'Spectrogram created successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500    