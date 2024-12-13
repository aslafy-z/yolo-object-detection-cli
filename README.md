# yolo-object-detection-cli

## Usage

### Docker

```shell
git clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples
docker pull ghcr.io/aslafy-z/yolo-object-detection-cli:main
docker run -it --rm \
  -v $PWD/samples/data:/samples \
  -p 8000:8000 \
  ghcr.io/aslafy-z/yolo-object-detection-cli:main \
  --source=/samples/shop.mp4
```

> Note: To run with the nvidia runtime, add `--ipc=host --gpus all` to the `docker run` command.

### System

```shell
sudo apt update && sudo apt install -y ffmpeg libsm6 libxext6

git clone https://github.com/aslafy-z/yolo-object-detection-cli && cd ./yolo-object-detection-cli
git lfs clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples

python -m virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt

python main.py --source=samples/data/shop.mp4
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
