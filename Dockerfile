ARG TARGETARCH=${TARGETARCH:-amd64}

# renovate: datasource=pypi depName=ultralytics
ARG ULTRALYTICS_VERSION=8.3.49

# Source at https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-cpu
FROM ultralytics/ultralytics:${ULTRALYTICS_VERSION}-cpu@sha256:89a194b0e555ee8999ebc0e9d12e622d2d2b2026fd709c15fd62ad51c0d92ec8 AS base_amd64
# Source at https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-jetson-jetpack5
FROM ultralytics/ultralytics:${ULTRALYTICS_VERSION}-jetson-jetpack5@sha256:5c76b71c3d4348ffd559ca7d3f93e50ac0356becceecebd953be074beab551da AS base_arm64

FROM base_${TARGETARCH} AS final

WORKDIR /app/weights

RUN ln -fs /ultralytics/yolo11n.pt yolo11n.pt

WORKDIR /app

COPY requirements.docker.txt .
RUN uv pip install --system -r requirements.docker.txt

COPY . .

EXPOSE 8000
ENTRYPOINT ["python", "main.py", "--http-host", "0.0.0.0", "--http-port", "8000"]
