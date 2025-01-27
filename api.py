from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.load_state_dict(torch.load('model/model.pth', map_location=device))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# API endpoint for predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

        # Load and preprocess the image
        image = Image.open(file.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            class_names = ["cat", "dog"]
            prediction = class_names[predicted.item()]

        return {"prediction": prediction}

    except HTTPException as http_exc:
        # Return specific error for invalid file types
        return JSONResponse(status_code=http_exc.status_code, content={"error": http_exc.detail})

    except Exception as e:
        # Handle generic errors
        return {"error": f"An error occurred: {str(e)}"}