ARG TARGETARCH=${TARGETARCH:-amd64}

# Source at https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-cpu
FROM ultralytics/ultralytics:latest-cpu@sha256:d24ce7bbc999e4733c89f089e7a4987dcb1444eb228792f668ad654afb006a4f AS base_amd64
# Source at https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-jetson-jetpack5
FROM ultralytics/ultralytics:latest-jetson-jetpack5@sha256:5c76b71c3d4348ffd559ca7d3f93e50ac0356becceecebd953be074beab551da AS base_arm64

FROM base_${TARGETARCH} AS final

WORKDIR /app/weights

RUN ln -fs /ultralytics/yolo11n.pt yolo11n.pt

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8000
ENTRYPOINT ["python", "main.py", "--http-host", "0.0.0.0", "--http-port", "8000"]
