# See https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1

ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /usr/share/fonts/truetype/

WORKDIR /app

ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt \
    /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENTRYPOINT ["python", "main.py"]
