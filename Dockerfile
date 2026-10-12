FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y     ffmpeg     libgl1     libglib2.0-0     libgomp1     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install No-AVX version of paddlepaddle to fix SIGILL (Code 132) on Cloud CPUs
RUN pip install paddlepaddle==2.6.2 -f https://www.paddlepaddle.org.cn/whl/linux/openblas/noavx/stable.html

# 2. Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "1800"]
