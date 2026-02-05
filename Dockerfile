FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV ASR_MODEL_PATH=/app/models/best_model.pt
ENV N_MEL_CHANNELS=80
ENV D_MODEL=256
ENV N_HEADS=4
ENV N_BLOCKS=8

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "serve/api.py"]
