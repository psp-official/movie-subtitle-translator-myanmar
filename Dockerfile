FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for FFmpeg and OpenCV (PaddleOCR)
RUN apt-get update && apt-get install -y     ffmpeg     libgl1-mesa-glx     libglib2.0-0     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "1800"]
