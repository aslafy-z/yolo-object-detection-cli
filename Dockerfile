ARG TARGETARCH=${TARGETARCH:-amd64}

# renovate: datasource=pypi depName=ultralytics
ARG ULTRALYTICS_VERSION=8.3.58

# Source at https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-cpu
FROM ultralytics/ultralytics:${ULTRALYTICS_VERSION}-cpu AS base_amd64
# Source at https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-jetson-jetpack5
FROM ultralytics/ultralytics:${ULTRALYTICS_VERSION}-jetson-jetpack5 AS base_arm64

FROM base_${TARGETARCH} AS final

ARG TARGETARCH

WORKDIR /app/weights

RUN ln -fs /ultralytics/yolo11n.pt yolo11n.pt

WORKDIR /app

COPY requirements.docker.txt .
RUN uv pip install --system -r requirements.docker.txt
RUN uv pip freeze --system
RUN ldconfig -p
COPY . .

EXPOSE 8000
ENTRYPOINT ["python3", "main.py", "--http-host", "0.0.0.0", "--http-port", "8000"]
