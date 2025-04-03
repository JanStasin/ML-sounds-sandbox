# Use a Python base image
FROM python:3.11-slim AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    build-essential \
    ffmpeg \
    libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy files to the container
COPY app.py ./ 
COPY working_models/results_and_model_acc_83.5_LR_0.00085_nclasses_15.npy .
COPY audio_ds_model.py ./
COPY encoded_labels.npy ./ 
COPY helper_functions.py ./ 
COPY requirements.txt ./ 

COPY dict_mats_dB.npy ./
    
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the app
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
