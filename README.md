# yolo-object-detection-cli

## Usage

### Docker

Pull dependencies

```shell
git lfs clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples
docker pull ghcr.io/aslafy-z/yolo-object-detection-cli:main
```

Run with CPU support:

```shell
docker run -it --rm \
  -v $PWD/samples/data:/samples \
  -p 8000:8000 \
  ghcr.io/aslafy-z/yolo-object-detection-cli:main \
  --source=/samples/shop.mp4
```

Run with GPU support:

```shell
docker run -it --rm \
  -v $PWD/samples/data:/samples \
  -p 8000:8000 \
  --ipc=host --gpus=all --runtime=nvidia \
  ghcr.io/aslafy-z/yolo-object-detection-cli:main \
  --source=/samples/shop.mp4 \
  --device=cuda:0
```

### System

```shell
sudo apt update && sudo apt install -y ffmpeg libsm6 libxext6

git clone https://github.com/aslafy-z/yolo-object-detection-cli && cd ./yolo-object-detection-cli
git lfs clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples

python -m virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt

python main.py --source=samples/data/shop.mp4
python main.py --source=rtsp://user:pass@10.0.0.5/Src/MediaInput/stream_1
```

## Development

### Lock dependencies

```shell
uv lock
uv pip compile pyproject.toml -o requirements.txt
```

> Note: Docker image uses a subset of the dependencies, see in `requirements.docker.txt`.

### Run

```shell
git lfs clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples
uv run main.py --source=samples/data/shop.mp4
```
